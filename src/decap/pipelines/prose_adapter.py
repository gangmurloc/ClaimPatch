import csv
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator
from scipy.optimize import linear_sum_assignment

from decap.evaluation.patch_metrics import patch_metrics
from decap.impact.propagation import rule_based_impact
from decap.models.llm_client import build_llm_client
from decap.models.structured_generation import extract_json_object, render_prompt
from decap.patch.generator import build_patch_from_impact
from decap.schemas.claims import ClaimNode, ClaimType
from decap.schemas.graph import DependencyEdge, DependencyType
from decap.schemas.results import AnswerVersion, SyntheticInstance
from decap.schemas.updates import ImpactLabel
from decap.data.synthetic_generator import generate_synthetic_dataset


_CLAIM_TYPES = {
    "factual",
    "numeric",
    "comparative",
    "interpretive",
    "temporal",
    "recommendation",
    "citation_only",
}
_DEPENDENCY_TYPES = {
    "numeric",
    "logical",
    "comparative",
    "temporal",
    "causal",
    "citation",
    "scope",
    "other",
}
_STYLE_HINTS = [
    "Lead with the overall comparison, then give supporting details.",
    "Begin with the evaluation setting, then report the results.",
    "Lead with the two metric values and close with the source reference.",
    "Use a compact report style with varied sentence lengths.",
]
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


class ExtractedClaim(BaseModel):
    local_id: str
    text: str
    claim_type: ClaimType

    @field_validator("local_id", "text")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("extracted claim fields must be non-empty")
        return value


class ExtractedDependency(BaseModel):
    source_claim_ids: List[str]
    target_claim_id: str
    dependency_type: DependencyType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rule: Optional[str] = None

    @field_validator("source_claim_ids")
    @classmethod
    def non_empty_sources(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("dependency sources must be non-empty")
        return value


class ExtractedClaimGraph(BaseModel):
    claims: List[ExtractedClaim]
    dependencies: List[ExtractedDependency]


class ProseAdapterGenerationError(RuntimeError):
    def __init__(self, stage: str, reason: str, raw_text: str = ""):
        super().__init__(f"{stage} failed: {reason}")
        self.stage = stage
        self.reason = reason
        self.raw_text = raw_text


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_path(name: str) -> Path:
    return Path("prompts") / name


def _prompt_text(name: str) -> str:
    path = _prompt_path(name)
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _manifest_payload(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    prompt_config = config.get("prompts", {})
    prose_version = prompt_config.get("prose_generation_version", "v1")
    extraction_version = prompt_config.get("graph_extraction_version", "v1")
    prompt_names = [
        f"prose_generation/explicit_{prose_version}.txt",
        f"prose_generation/implicit_{prose_version}.txt",
        f"prose_graph_extraction/{extraction_version}.txt",
    ]
    return {
        "run_id": config["run_id"],
        "config_path": str(config_path),
        "config_sha256": _sha256_text(config_path.read_text(encoding="utf-8")),
        "prompt_sha256": {
            name: _sha256_text(_prompt_text(name))
            for name in prompt_names
        },
        "models": {
            key: value.get("model_name", "")
            for key, value in config.get("models", {}).items()
        },
        "scope": "synthetic_prose_adapter_only",
        "real_world_external_validity": False,
    }


def _prepare_output(config_path: Path, config: Dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated").mkdir(exist_ok=True)
    (output_dir / "extracted").mkdir(exist_ok=True)
    (output_dir / "stage_failures").mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    expected = _manifest_payload(config_path, config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"stale output manifest at {manifest_path}; use a new output_dir "
                "rather than mixing changed config/prompts with existing artifacts"
            )
    else:
        manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "config.yaml").write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return output_dir


def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _instances(config: Dict[str, Any]) -> List[SyntheticInstance]:
    dataset = config["dataset"]
    return generate_synthetic_dataset(
        size=int(dataset["size"]),
        seed=int(config["seed"]),
        sequential_steps=int(dataset.get("sequential_steps", 1)),
        profile=dataset.get("profile", "standard"),
    )


class ProseGenerator:
    def __init__(self, client: Any, prompt_version: str = "v1"):
        self.client = client
        self.prompt_version = prompt_version

    def generate(
        self,
        question: str,
        claim_texts: Sequence[str],
        condition: str,
        style_hint: str,
    ) -> Tuple[str, str, bool]:
        if condition not in {"explicit", "implicit"}:
            raise ValueError(f"unknown prose condition: {condition}")
        template = _prompt_text(f"prose_generation/{condition}_{self.prompt_version}.txt")
        prompt = render_prompt(
            template,
            question=question,
            claims=list(claim_texts),
            style_hint=style_hint,
        )
        raw = self.client.generate_text(prompt, task=f"prose_generation_{condition}")
        try:
            parsed = extract_json_object(raw)
            prose = parsed["prose"]
            if not isinstance(prose, str) or not prose.strip():
                raise ValueError("prose must be a non-empty string")
            return prose.strip(), raw, True
        except Exception as exc:
            # This is a data-construction stage rather than the structured
            # graph interface under evaluation. Some instruction models return
            # the requested paragraph directly despite the JSON wrapper. Keep
            # a non-empty plain paragraph while recording format noncompliance.
            stripped = raw.strip()
            if stripped and not stripped.startswith(("{", "[")):
                return stripped, raw, False
            raise ProseAdapterGenerationError("prose_generation", str(exc), raw) from exc


class ProseGraphExtractor:
    def __init__(self, client: Any, prompt_version: str = "v1"):
        self.client = client
        self.prompt_version = prompt_version

    def extract(self, question: str, prose: str) -> Tuple[ExtractedClaimGraph, str]:
        template = _prompt_text(f"prose_graph_extraction/{self.prompt_version}.txt")
        prompt = render_prompt(template, question=question, prose=prose)
        raw = self.client.generate_text(prompt, task="prose_graph_extraction")
        try:
            parsed = extract_json_object(raw)
            graph = ExtractedClaimGraph.model_validate(parsed)
            _validate_extracted_graph(graph)
        except Exception as exc:
            raise ProseAdapterGenerationError("prose_graph_extraction", str(exc), raw) from exc
        return graph, raw


def _validate_extracted_graph(graph: ExtractedClaimGraph) -> None:
    ids = [claim.local_id for claim in graph.claims]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate extracted claim local_id")
    id_set = set(ids)
    for claim in graph.claims:
        if claim.claim_type not in _CLAIM_TYPES:
            raise ValueError(f"invalid claim type: {claim.claim_type}")
    for dependency in graph.dependencies:
        if dependency.dependency_type not in _DEPENDENCY_TYPES:
            raise ValueError(f"invalid dependency type: {dependency.dependency_type}")
        refs = set(dependency.source_claim_ids + [dependency.target_claim_id])
        missing = refs - id_set
        if missing:
            raise ValueError(f"dependency references unknown local IDs: {sorted(missing)}")
        if dependency.target_claim_id in dependency.source_claim_ids:
            raise ValueError("self dependency is invalid")


def run_prose_generation(config_path: Path, model_key: str, force: bool = False) -> Path:
    config = _load_config(config_path)
    output_dir = _prepare_output(config_path, config)
    if model_key not in config["models"]:
        raise KeyError(f"unknown model key: {model_key}")
    destination = output_dir / "generated" / f"{model_key}.jsonl"
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} exists; pass --force only for an intentional same-protocol rerun")

    instances = _instances(config)
    conditions = list(config.get("conditions", ["explicit", "implicit"]))
    client = build_llm_client(config["models"][model_key])
    generator = ProseGenerator(
        client,
        prompt_version=config.get("prompts", {}).get("prose_generation_version", "v1"),
    )
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()
    total = len(instances) * len(conditions)
    completed = 0
    for index, instance in enumerate(instances):
        for condition_index, condition in enumerate(conditions):
            claim_texts = [
                claim.text
                for claim in instance.answer_v0.claims
                if claim.status == "active"
            ]
            shuffle_seed = int(config["seed"]) * 1000003 + index * 101 + condition_index
            random.Random(shuffle_seed).shuffle(claim_texts)
            style_hint = _STYLE_HINTS[(index + condition_index) % len(_STYLE_HINTS)]
            row: Dict[str, Any] = {
                "instance_id": instance.instance_id,
                "condition": condition,
                "generator_model_key": model_key,
                "generator_model_name": client.model_name,
                "question": instance.question,
                "success": False,
                "format_success": False,
                "prose": "",
                "raw_text": "",
            }
            try:
                prose, raw, format_success = generator.generate(
                    instance.question,
                    claim_texts,
                    condition,
                    style_hint,
                )
                row.update(
                    {
                        "success": True,
                        "format_success": format_success,
                        "prose": prose,
                        "raw_text": raw,
                    }
                )
            except ProseAdapterGenerationError as exc:
                row["raw_text"] = exc.raw_text
                row["error"] = exc.reason
                failures.append(dict(row))
            rows.append(row)
            completed += 1
            print(f"Generated prose {completed}/{total}: {instance.instance_id} {condition}", flush=True)

    _write_jsonl_atomic(destination, rows)
    _write_jsonl_atomic(output_dir / "stage_failures" / f"generate_{model_key}.jsonl", failures)
    elapsed = time.time() - started
    print(f"Prose generation complete for {model_key}: {destination} ({elapsed:.1f}s)")
    return destination


def run_prose_extraction(config_path: Path, model_key: str, force: bool = False) -> Path:
    config = _load_config(config_path)
    output_dir = _prepare_output(config_path, config)
    if model_key not in config["models"]:
        raise KeyError(f"unknown model key: {model_key}")
    destination = output_dir / "extracted" / f"{model_key}.jsonl"
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} exists; pass --force only for an intentional same-protocol rerun")

    generated_rows: List[Dict[str, Any]] = []
    for generator_key in config["models"]:
        path = output_dir / "generated" / f"{generator_key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing generation artifact: {path}")
        generated_rows.extend(_read_jsonl(path))

    client = build_llm_client(config["models"][model_key])
    extractor = ProseGraphExtractor(
        client,
        prompt_version=config.get("prompts", {}).get("graph_extraction_version", "v1"),
    )
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()
    total = len(generated_rows)
    for index, source in enumerate(generated_rows, 1):
        row: Dict[str, Any] = {
            "instance_id": source["instance_id"],
            "condition": source["condition"],
            "generator_model_key": source["generator_model_key"],
            "generator_model_name": source["generator_model_name"],
            "extractor_model_key": model_key,
            "extractor_model_name": client.model_name,
            "question": source["question"],
            "prose": source.get("prose", ""),
            "generation_success": bool(source.get("success")),
            "generation_format_success": bool(source.get("format_success")),
            "success": False,
            "graph": {"claims": [], "dependencies": []},
            "raw_text": "",
        }
        if not source.get("success"):
            row["error"] = "upstream prose generation failed"
            failures.append(dict(row))
            rows.append(row)
            print(f"Extracted graph {index}/{total}: upstream failure", flush=True)
            continue
        try:
            graph, raw = extractor.extract(source["question"], source["prose"])
            row.update({"success": True, "graph": graph.model_dump(), "raw_text": raw})
        except ProseAdapterGenerationError as exc:
            row["raw_text"] = exc.raw_text
            row["error"] = exc.reason
            failures.append(dict(row))
        rows.append(row)
        print(
            f"Extracted graph {index}/{total}: "
            f"{source['generator_model_key']}->{model_key} "
            f"{source['instance_id']} {source['condition']}",
            flush=True,
        )

    _write_jsonl_atomic(destination, rows)
    _write_jsonl_atomic(output_dir / "stage_failures" / f"extract_{model_key}.jsonl", failures)
    elapsed = time.time() - started
    print(f"Graph extraction complete for {model_key}: {destination} ({elapsed:.1f}s)")
    return destination


def build_prose_audit(config_path: Path) -> Path:
    config = _load_config(config_path)
    output_dir = _prepare_output(config_path, config)
    instances = {instance.instance_id: instance for instance in _instances(config)}
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for generator_key in config["models"]:
        path = output_dir / "generated" / f"{generator_key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing generation artifact: {path}")
        for row in _read_jsonl(path):
            groups[(generator_key, row["condition"])].append(row)

    requested = int(config.get("audit", {}).get("sample_count", 12))
    group_count = max(1, len(groups))
    per_group = max(1, int(math.ceil(requested / group_count)))
    selected: List[Dict[str, Any]] = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda item: item["instance_id"])[:per_group])
    selected = selected[: max(requested, group_count)]

    lines = [
        f"# Prose fidelity and leakage audit: {config['run_id']}",
        "",
        "This audit is generated before graph extraction scoring. Mark each item",
        "for fact omission, hallucinated addition, dependency leakage, unnatural",
        "templating, and condition compliance. Automatic results must not replace",
        "this inspection.",
        "",
    ]
    for audit_index, row in enumerate(selected, 1):
        instance = instances[row["instance_id"]]
        lines.extend(
            [
                f"## Sample {audit_index}: {row['generator_model_key']} / {row['condition']} / {row['instance_id']}",
                "",
                f"Generation success: `{row.get('success', False)}`",
                "",
                "Gold atomic facts:",
                "",
            ]
        )
        for claim in instance.answer_v0.claims:
            lines.append(f"- `{claim.claim_id}` ({claim.claim_type}): {claim.text}")
        lines.extend(["", "Gold dependencies:", ""])
        for edge in instance.answer_v0.dependencies:
            lines.append(
                f"- `{'+'.join(edge.source_claim_ids)} -> {edge.target_claim_id}` "
                f"({edge.dependency_type})"
            )
        lines.extend(
            [
                "",
                "Generated prose:",
                "",
                row.get("prose", "") or f"[FAILED: {row.get('error', 'unknown')}]",
                "",
                "Human audit fields:",
                "",
                "- Fact omission: [ ] no [ ] yes",
                "- Unsupported addition: [ ] no [ ] yes",
                "- Dependency leakage: [ ] no [ ] yes",
                "- Unnatural/template artifact: [ ] no [ ] yes",
                "- Condition compliant: [ ] yes [ ] no",
                "- Notes:",
                "",
            ]
        )
    destination = output_dir / "prose_audit.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Prose audit written: {destination}")
    return destination


class SemanticNodeMatcher:
    def __init__(self, config: Dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for semantic node alignment") from exc

        device = config.get("device", "cpu")
        self.model = SentenceTransformer(
            config.get("model_name", "BAAI/bge-m3"),
            device=device,
            local_files_only=bool(config.get("local_files_only", True)),
        )

    def encode(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        unique = list(dict.fromkeys(texts))
        vectors = self.model.encode(
            unique,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return {text: vector for text, vector in zip(unique, vectors)}


class NLIFidelityAuditor:
    """Independent argmax NLI audit of gold-claim coverage in generated prose."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers/torch are required for the NLI fidelity audit") from exc

        self.torch = torch
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model_name = config.get("model_name", "cross-encoder/nli-deberta-v3-base")
        local_files_only = bool(config.get("local_files_only", True))
        self.batch_size = int(config.get("batch_size", 32))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        label2id = {
            str(label).lower(): int(index)
            for label, index in self.model.config.label2id.items()
        }
        if "entailment" not in label2id:
            raise ValueError(f"NLI model has no entailment label: {self.model.config.label2id}")
        self.entailment_id = label2id["entailment"]

    def score(
        self,
        pairs: Sequence[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], Tuple[bool, float]]:
        unique = list(dict.fromkeys(pairs))
        results: Dict[Tuple[str, str], Tuple[bool, float]] = {}
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
                probabilities = self.torch.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1)
            for pair, label, probability in zip(
                batch,
                predicted.detach().cpu().tolist(),
                probabilities[:, self.entailment_id].detach().cpu().tolist(),
            ):
                results[pair] = (int(label) == self.entailment_id, float(probability))
        return results


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_f1(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _numeric_agreement(left: str, right: str) -> float:
    left_numbers = set(_NUMBER_RE.findall(left.lower()))
    right_numbers = set(_NUMBER_RE.findall(right.lower()))
    if not left_numbers and not right_numbers:
        return 1.0
    if not left_numbers or not right_numbers:
        return 0.0
    return len(left_numbers & right_numbers) / len(left_numbers | right_numbers)


def _similarity(left: str, right: str, embeddings: Dict[str, np.ndarray]) -> float:
    cosine = float(np.dot(embeddings[left], embeddings[right]))
    cosine = max(0.0, min(1.0, cosine))
    return (
        0.75 * cosine
        + 0.15 * _token_f1(left, right)
        + 0.10 * _numeric_agreement(left, right)
    )


def _match_nodes(
    predicted: Sequence[ExtractedClaim],
    gold: Sequence[ClaimNode],
    embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    if not predicted or not gold:
        return {}, []
    matrix = np.zeros((len(predicted), len(gold)), dtype=np.float64)
    for pred_index, pred in enumerate(predicted):
        for gold_index, ref in enumerate(gold):
            matrix[pred_index, gold_index] = _similarity(pred.text, ref.text, embeddings)
    pred_indices, gold_indices = linear_sum_assignment(-matrix)
    mapping: Dict[str, str] = {}
    details: List[Dict[str, Any]] = []
    for pred_index, gold_index in zip(pred_indices.tolist(), gold_indices.tolist()):
        score = float(matrix[pred_index, gold_index])
        accepted = score >= threshold
        pred = predicted[pred_index]
        ref = gold[gold_index]
        details.append(
            {
                "predicted_local_id": pred.local_id,
                "predicted_text": pred.text,
                "gold_claim_id": ref.claim_id,
                "gold_text": ref.text,
                "similarity": score,
                "accepted": accepted,
                "type_match": pred.claim_type == ref.claim_type,
            }
        )
        if accepted:
            mapping[pred.local_id] = ref.claim_id
    return mapping, details


def _prf(tp: int, predicted: int, gold: int) -> Tuple[float, float, float]:
    precision = tp / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = tp / gold if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _edge_key(edge: DependencyEdge, typed: bool = True) -> Tuple[Any, ...]:
    base: Tuple[Any, ...] = (tuple(sorted(edge.source_claim_ids)), edge.target_claim_id)
    return base + (edge.dependency_type,) if typed else base


def _graph_metrics(
    graph: ExtractedClaimGraph,
    gold_answer: AnswerVersion,
    mapping: Dict[str, str],
) -> Tuple[Dict[str, float], List[DependencyEdge]]:
    gold_typed = {_edge_key(edge, typed=True) for edge in gold_answer.dependencies}
    gold_untyped = {_edge_key(edge, typed=False) for edge in gold_answer.dependencies}
    pred_local_keys = {
        (
            tuple(sorted(edge.source_claim_ids)),
            edge.target_claim_id,
            edge.dependency_type,
        )
        for edge in graph.dependencies
    }
    mapped_edges: List[DependencyEdge] = []
    for edge in graph.dependencies:
        ids = list(edge.source_claim_ids) + [edge.target_claim_id]
        if not all(claim_id in mapping for claim_id in ids):
            continue
        sources = [mapping[claim_id] for claim_id in edge.source_claim_ids]
        target = mapping[edge.target_claim_id]
        if target in sources:
            continue
        mapped_edges.append(
            DependencyEdge(
                source_claim_ids=sources,
                target_claim_id=target,
                dependency_type=edge.dependency_type,
                confidence=edge.confidence,
                rule=edge.rule,
                metadata={"source": "prose_adapter"},
            )
        )
    mapped_typed = {_edge_key(edge, typed=True) for edge in mapped_edges}
    mapped_untyped = {_edge_key(edge, typed=False) for edge in mapped_edges}
    typed_tp = len(mapped_typed & gold_typed)
    untyped_tp = len(mapped_untyped & gold_untyped)
    typed_precision, typed_recall, typed_f1 = _prf(typed_tp, len(pred_local_keys), len(gold_typed))
    untyped_precision, untyped_recall, untyped_f1 = _prf(
        untyped_tp,
        len(pred_local_keys),
        len(gold_untyped),
    )
    type_correct = 0
    for mapped in mapped_edges:
        if _edge_key(mapped, typed=False) not in gold_untyped:
            continue
        if _edge_key(mapped, typed=True) in gold_typed:
            type_correct += 1
    multi_gold = {key for key in gold_untyped if len(key[0]) > 1}
    multi_tp = len(mapped_untyped & multi_gold)

    digraph = nx.DiGraph()
    digraph.add_nodes_from(claim.local_id for claim in graph.claims)
    for edge in graph.dependencies:
        for source in edge.source_claim_ids:
            digraph.add_edge(source, edge.target_claim_id)
    dag_valid = 1.0 if nx.is_directed_acyclic_graph(digraph) else 0.0
    return (
        {
            "edge_typed_tp": float(typed_tp),
            "edge_untyped_tp": float(untyped_tp),
            "edge_pred_count": float(len(pred_local_keys)),
            "edge_gold_count": float(len(gold_typed)),
            "edge_typed_precision": typed_precision,
            "edge_typed_recall": typed_recall,
            "edge_typed_f1": typed_f1,
            "edge_untyped_precision": untyped_precision,
            "edge_untyped_recall": untyped_recall,
            "edge_untyped_f1": untyped_f1,
            "edge_type_correct": float(type_correct),
            "edge_type_denominator": float(untyped_tp),
            "multi_parent_tp": float(multi_tp),
            "multi_parent_gold": float(len(multi_gold)),
            "multi_parent_recall": multi_tp / len(multi_gold) if multi_gold else 1.0,
            "dag_valid": dag_valid,
        },
        mapped_edges,
    )


def _aligned_downstream_probe(
    instance: SyntheticInstance,
    mapping: Dict[str, str],
    mapped_edges: List[DependencyEdge],
) -> Dict[str, float]:
    current = instance.answer_v0
    update = instance.updates[0]
    fresh = instance.fresh_answers[0]
    gold_labels = instance.gold_impact_labels[0]
    matched_gold_ids = set(mapping.values())
    aligned_claims = [
        claim.model_copy(deep=True)
        for claim in current.claims
        if claim.claim_id in matched_gold_ids and claim.status == "active"
    ]
    aligned_edge_keys = set()
    aligned_edges: List[DependencyEdge] = []
    for edge in mapped_edges:
        key = _edge_key(edge, typed=True)
        if key in aligned_edge_keys:
            continue
        aligned_edge_keys.add(key)
        aligned_edges.append(edge)
    aligned_answer = AnswerVersion(
        answer_id=current.answer_id,
        version=current.version,
        question=current.question,
        rendered_text=current.rendered_text,
        claims=aligned_claims,
        dependencies=aligned_edges,
        evidence=current.evidence,
        parent_version=current.parent_version,
        applied_patch_id=current.applied_patch_id,
    )
    aligned_labels = rule_based_impact(aligned_answer, update)
    by_id = {label.claim_id: label for label in aligned_labels}
    completed_labels: List[ImpactLabel] = []
    for claim in current.claims:
        if claim.claim_id in by_id:
            completed_labels.append(by_id[claim.claim_id])
        else:
            completed_labels.append(
                ImpactLabel(
                    claim_id=claim.claim_id,
                    state="STILL_VALID",
                    direct=False,
                    reason="The prose adapter omitted or failed to align this claim.",
                    gold_operation=None,
                )
            )

    extracted_patch = build_patch_from_impact(current, fresh, completed_labels)
    gold_patch = build_patch_from_impact(current, fresh, gold_labels)
    all_ids = {claim.claim_id for claim in current.claims if claim.status == "active"}
    extracted_metrics = patch_metrics(extracted_patch, gold_labels, all_ids)
    gold_metrics = patch_metrics(gold_patch, gold_labels, all_ids)
    return {
        "e2e_dcs": extracted_metrics["dependency_complete_success"],
        "e2e_patch_precision": extracted_metrics["patch_precision"],
        "e2e_patch_recall": extracted_metrics["patch_recall"],
        "e2e_collateral": extracted_metrics["collateral_edit_rate"],
        "e2e_residual": extracted_metrics["residual_stale_rate"],
        "gold_dcs": gold_metrics["dependency_complete_success"],
        "gold_collateral": gold_metrics["collateral_edit_rate"],
        "gold_residual": gold_metrics["residual_stale_rate"],
        "dcs_loss": gold_metrics["dependency_complete_success"]
        - extracted_metrics["dependency_complete_success"],
        "collateral_increase": extracted_metrics["collateral_edit_rate"]
        - gold_metrics["collateral_edit_rate"],
        "residual_increase": extracted_metrics["residual_stale_rate"]
        - gold_metrics["residual_stale_rate"],
    }


def _bootstrap_mean_ci(values: Sequence[float], seed: int, samples: int) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return mean, float(low), float(high)


def _aggregate_cell(
    rows: List[Dict[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    node_tp = int(sum(row["metrics"]["node_tp"] for row in rows))
    node_pred = int(sum(row["metrics"]["node_pred_count"] for row in rows))
    node_gold = int(sum(row["metrics"]["node_gold_count"] for row in rows))
    node_precision, node_recall, node_f1 = _prf(node_tp, node_pred, node_gold)
    edge_typed_tp = int(sum(row["metrics"]["edge_typed_tp"] for row in rows))
    edge_untyped_tp = int(sum(row["metrics"]["edge_untyped_tp"] for row in rows))
    edge_pred = int(sum(row["metrics"]["edge_pred_count"] for row in rows))
    edge_gold = int(sum(row["metrics"]["edge_gold_count"] for row in rows))
    edge_typed_precision, edge_typed_recall, edge_typed_f1 = _prf(
        edge_typed_tp,
        edge_pred,
        edge_gold,
    )
    edge_untyped_precision, edge_untyped_recall, edge_untyped_f1 = _prf(
        edge_untyped_tp,
        edge_pred,
        edge_gold,
    )
    type_correct = sum(row["metrics"]["node_type_correct"] for row in rows)
    type_denominator = sum(row["metrics"]["node_type_denominator"] for row in rows)
    edge_type_correct = sum(row["metrics"]["edge_type_correct"] for row in rows)
    edge_type_denominator = sum(row["metrics"]["edge_type_denominator"] for row in rows)
    multi_tp = sum(row["metrics"]["multi_parent_tp"] for row in rows)
    multi_gold = sum(row["metrics"]["multi_parent_gold"] for row in rows)
    average_names = [
        "parse_success",
        "generation_success",
        "generation_format_success",
        "prose_claim_coverage",
        "prose_entailment_probability",
        "dag_valid",
        "e2e_dcs",
        "e2e_patch_precision",
        "e2e_patch_recall",
        "e2e_collateral",
        "e2e_residual",
        "gold_dcs",
        "gold_collateral",
        "gold_residual",
        "dcs_loss",
        "collateral_increase",
        "residual_increase",
    ]
    result: Dict[str, Any] = {
        "n": len(rows),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "node_type_accuracy": type_correct / type_denominator if type_denominator else 0.0,
        "edge_typed_precision": edge_typed_precision,
        "edge_typed_recall": edge_typed_recall,
        "edge_typed_f1": edge_typed_f1,
        "edge_untyped_precision": edge_untyped_precision,
        "edge_untyped_recall": edge_untyped_recall,
        "edge_untyped_f1": edge_untyped_f1,
        "edge_type_accuracy": (
            edge_type_correct / edge_type_denominator
            if edge_type_denominator
            else 0.0
        ),
        "multi_parent_recall": multi_tp / multi_gold if multi_gold else 1.0,
    }
    for name in average_names:
        result[name] = sum(row["metrics"][name] for row in rows) / len(rows) if rows else 0.0
    for offset, name in enumerate(["dcs_loss", "collateral_increase", "residual_increase"]):
        mean, low, high = _bootstrap_mean_ci(
            [row["metrics"][name] for row in rows],
            seed=seed + offset,
            samples=bootstrap_samples,
        )
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
        result[name] = mean
    return result


def _gate_result(metrics: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "node_f1": metrics["node_f1"] >= 0.85,
        "edge_typed_f1": metrics["edge_typed_f1"] >= 0.75,
        "multi_parent_recall": metrics["multi_parent_recall"] >= 0.70,
        "parse_success": metrics["parse_success"] >= 0.98,
        "dcs_loss": metrics["dcs_loss"] <= 0.10,
        "collateral_increase": metrics["collateral_increase"] <= 0.05,
        "residual_increase": metrics["residual_increase"] <= 0.05,
    }


def _smoke_go_no_go(cells: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Operational lower bound fixed for the final v2 development smoke."""

    all_cells = list(cells.values())
    total_n = sum(int(cell["n"]) for cell in all_cells)
    overall_parse = (
        sum(float(cell["parse_success"]) * int(cell["n"]) for cell in all_cells) / total_n
        if total_n
        else 0.0
    )
    primary: Dict[str, Dict[str, bool]] = {}
    for cell_name, cell in cells.items():
        if not cell["off_diagonal"]:
            continue
        primary[cell_name] = {
            "parse_success": cell["parse_success"] >= 0.90,
            "edge_typed_f1": cell["edge_typed_f1"] >= 0.50,
            "multi_parent_recall": cell["multi_parent_recall"] >= 0.50,
            "dcs_loss": cell["dcs_loss"] <= 0.30,
        }
    overall_parse_pass = overall_parse >= 0.95
    primary_pass = bool(primary) and all(all(checks.values()) for checks in primary.values())
    return {
        "pass": overall_parse_pass and primary_pass,
        "overall_parse_success": overall_parse,
        "overall_parse_pass": overall_parse_pass,
        "primary_cells": primary,
        "criteria": {
            "overall_parse_success_min": 0.95,
            "off_diagonal_parse_success_min": 0.90,
            "off_diagonal_edge_typed_f1_min": 0.50,
            "off_diagonal_multi_parent_recall_min": 0.50,
            "off_diagonal_dcs_loss_max": 0.30,
        },
    }


def run_prose_evaluation(config_path: Path) -> Path:
    config = _load_config(config_path)
    output_dir = _prepare_output(config_path, config)
    instances = {instance.instance_id: instance for instance in _instances(config)}
    extracted_rows: List[Dict[str, Any]] = []
    for extractor_key in config["models"]:
        path = output_dir / "extracted" / f"{extractor_key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing extraction artifact: {path}")
        extracted_rows.extend(_read_jsonl(path))

    all_texts: List[str] = []
    for instance in instances.values():
        all_texts.extend(claim.text for claim in instance.answer_v0.claims)
    for row in extracted_rows:
        all_texts.extend(
            claim.get("text", "")
            for claim in row.get("graph", {}).get("claims", [])
            if claim.get("text")
        )
    matcher = SemanticNodeMatcher(config.get("matching", {}))
    embeddings = matcher.encode(all_texts)
    del matcher
    _release_torch_memory()

    fidelity_pairs: List[Tuple[str, str]] = []
    for row in extracted_rows:
        instance = instances[row["instance_id"]]
        prose = row.get("prose", "")
        fidelity_pairs.extend((prose, claim.text) for claim in instance.answer_v0.claims)
    fidelity_auditor = NLIFidelityAuditor(config.get("fidelity_audit", {}))
    fidelity_scores = fidelity_auditor.score(fidelity_pairs)
    del fidelity_auditor
    _release_torch_memory()

    main_threshold = float(config.get("matching", {}).get("main_threshold", 0.70))
    sensitivity_thresholds = [
        float(value)
        for value in config.get("matching", {}).get("sensitivity_thresholds", [0.65, 0.75])
    ]

    evaluated: List[Dict[str, Any]] = []
    for index, row in enumerate(extracted_rows, 1):
        instance = instances[row["instance_id"]]
        graph = ExtractedClaimGraph.model_validate(row["graph"])
        mapping, match_details = _match_nodes(
            graph.claims,
            instance.answer_v0.claims,
            embeddings,
            main_threshold,
        )
        node_tp = len(mapping)
        node_precision, node_recall, node_f1 = _prf(
            node_tp,
            len(graph.claims),
            len(instance.answer_v0.claims),
        )
        accepted = [detail for detail in match_details if detail["accepted"]]
        node_type_correct = sum(1 for detail in accepted if detail["type_match"])
        prose_claim_scores = [
            fidelity_scores[(row.get("prose", ""), claim.text)]
            for claim in instance.answer_v0.claims
        ]
        prose_claim_coverage = (
            sum(1 for entailed, _ in prose_claim_scores if entailed)
            / len(prose_claim_scores)
            if prose_claim_scores
            else 0.0
        )
        prose_entailment_probability = (
            sum(probability for _, probability in prose_claim_scores)
            / len(prose_claim_scores)
            if prose_claim_scores
            else 0.0
        )
        edge_metrics, mapped_edges = _graph_metrics(graph, instance.answer_v0, mapping)
        downstream = _aligned_downstream_probe(instance, mapping, mapped_edges)
        metrics: Dict[str, float] = {
            "parse_success": 1.0 if row.get("success") else 0.0,
            "generation_success": 1.0 if row.get("generation_success") else 0.0,
            "generation_format_success": 1.0 if row.get("generation_format_success") else 0.0,
            "prose_claim_coverage": prose_claim_coverage,
            "prose_entailment_probability": prose_entailment_probability,
            "node_tp": float(node_tp),
            "node_pred_count": float(len(graph.claims)),
            "node_gold_count": float(len(instance.answer_v0.claims)),
            "node_precision": node_precision,
            "node_recall": node_recall,
            "node_f1": node_f1,
            "node_type_correct": float(node_type_correct),
            "node_type_denominator": float(len(accepted)),
            **edge_metrics,
            **downstream,
        }
        sensitivity: Dict[str, Dict[str, float]] = {}
        for threshold in sensitivity_thresholds:
            threshold_mapping, _ = _match_nodes(
                graph.claims,
                instance.answer_v0.claims,
                embeddings,
                threshold,
            )
            threshold_edge, _ = _graph_metrics(graph, instance.answer_v0, threshold_mapping)
            _, _, threshold_node_f1 = _prf(
                len(threshold_mapping),
                len(graph.claims),
                len(instance.answer_v0.claims),
            )
            sensitivity[f"{threshold:.2f}"] = {
                "node_tp": float(len(threshold_mapping)),
                "node_pred_count": float(len(graph.claims)),
                "node_gold_count": float(len(instance.answer_v0.claims)),
                "node_f1": threshold_node_f1,
                "edge_typed_tp": threshold_edge["edge_typed_tp"],
                "edge_pred_count": threshold_edge["edge_pred_count"],
                "edge_gold_count": threshold_edge["edge_gold_count"],
                "edge_typed_f1": threshold_edge["edge_typed_f1"],
            }
        evaluated.append(
            {
                "instance_id": row["instance_id"],
                "condition": row["condition"],
                "generator_model_key": row["generator_model_key"],
                "extractor_model_key": row["extractor_model_key"],
                "success": row.get("success", False),
                "mapping": mapping,
                "match_details": match_details,
                "predicted_graph": graph.model_dump(),
                "metrics": metrics,
                "sensitivity": sensitivity,
            }
        )
        print(f"Evaluated graph {index}/{len(extracted_rows)}", flush=True)

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        grouped[
            (
                row["generator_model_key"],
                row["extractor_model_key"],
                row["condition"],
            )
        ].append(row)
    bootstrap_samples = int(config.get("bootstrap", {}).get("samples", 2000))
    cells: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []
    for cell_index, (key, rows) in enumerate(sorted(grouped.items())):
        generator_key, extractor_key, condition = key
        cell_name = f"{generator_key}->{extractor_key}|{condition}"
        aggregate = _aggregate_cell(
            rows,
            seed=int(config["seed"]) + cell_index * 17,
            bootstrap_samples=bootstrap_samples,
        )
        aggregate["generator_model_key"] = generator_key
        aggregate["extractor_model_key"] = extractor_key
        aggregate["condition"] = condition
        aggregate["off_diagonal"] = generator_key != extractor_key
        aggregate["gates"] = _gate_result(aggregate)
        aggregate["all_gates_pass"] = all(aggregate["gates"].values())
        cells[cell_name] = aggregate
        table_rows.append({"cell": cell_name, **{k: v for k, v in aggregate.items() if k != "gates"}})

    sensitivity_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cell_name, aggregate in cells.items():
        generator_key = aggregate["generator_model_key"]
        extractor_key = aggregate["extractor_model_key"]
        condition = aggregate["condition"]
        rows = grouped[(generator_key, extractor_key, condition)]
        sensitivity_summary[cell_name] = {}
        for threshold in sensitivity_thresholds:
            label = f"{threshold:.2f}"
            node_tp = int(sum(row["sensitivity"][label]["node_tp"] for row in rows))
            node_pred = int(sum(row["sensitivity"][label]["node_pred_count"] for row in rows))
            node_gold = int(sum(row["sensitivity"][label]["node_gold_count"] for row in rows))
            edge_tp = int(sum(row["sensitivity"][label]["edge_typed_tp"] for row in rows))
            edge_pred = int(sum(row["sensitivity"][label]["edge_pred_count"] for row in rows))
            edge_gold = int(sum(row["sensitivity"][label]["edge_gold_count"] for row in rows))
            _, _, node_f1 = _prf(node_tp, node_pred, node_gold)
            _, _, edge_f1 = _prf(edge_tp, edge_pred, edge_gold)
            sensitivity_summary[cell_name][label] = {
                "node_f1": node_f1,
                "edge_typed_f1": edge_f1,
            }

    gate_applicable = int(config["dataset"]["size"]) >= 100
    primary_cells = [value for value in cells.values() if value["off_diagonal"]]
    cross_model_pass = gate_applicable and bool(primary_cells) and all(
        value["all_gates_pass"] for value in primary_cells
    )
    result = {
        "run_id": config["run_id"],
        "scope": "synthetic_prose_adapter_only",
        "oracle_aligned_downstream_probe": True,
        "real_world_external_validity": False,
        "main_alignment_threshold": main_threshold,
        "gate_applicable": gate_applicable,
        "cross_model_pass": cross_model_pass,
        "cells": cells,
        "sensitivity": sensitivity_summary,
    }
    if int(config["dataset"]["size"]) < 100:
        result["smoke_go_no_go"] = _smoke_go_no_go(cells)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl_atomic(output_dir / "predictions.jsonl", evaluated)
    _write_csv(output_dir / "metrics_by_cell.csv", table_rows)
    _write_summary(output_dir / "summary.md", config, result)
    print(f"Prose adapter evaluation complete: {output_dir / 'summary.md'}")
    return output_dir / "summary.md"


def _write_summary(
    path: Path,
    config: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    lines = [
        f"# {config['stage_title']}",
        "",
        "Scope: synthetic LLM-prose adapter pilot only. This is not real-world",
        "external validation. Downstream results use post-hoc oracle node alignment",
        "to isolate graph-extraction loss.",
        "",
        f"- run: `{config['run_id']}`",
        f"- base instances: {config['dataset']['size']}",
        f"- main node-match threshold: {result['main_alignment_threshold']:.2f}",
        f"- preregistered gate applicable: {result['gate_applicable']}",
        f"- all off-diagonal primary cells pass: {result['cross_model_pass']}",
        "",
        "## Main cell metrics",
        "",
        "| cell | prose coverage | parse | node F1 | typed edge F1 | multi-parent R | DCS loss | collateral inc. | residual inc. | gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if "smoke_go_no_go" in result:
        smoke_gate = result["smoke_go_no_go"]
        lines[9:9] = [
            f"- smoke go/no-go pass: {smoke_gate['pass']}",
            f"- smoke overall parse success: {smoke_gate['overall_parse_success']:.3f}",
        ]
    for cell_name, metrics in sorted(result["cells"].items()):
        lines.append(
            f"| {cell_name} | {metrics['prose_claim_coverage']:.3f} | "
            f"{metrics['parse_success']:.3f} | "
            f"{metrics['node_f1']:.3f} | {metrics['edge_typed_f1']:.3f} | "
            f"{metrics['multi_parent_recall']:.3f} | {metrics['dcs_loss']:.3f} | "
            f"{metrics['collateral_increase']:.3f} | {metrics['residual_increase']:.3f} | "
            f"{'PASS' if metrics['all_gates_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Paired base-instance bootstrap intervals",
            "",
            "| cell | DCS loss 95% CI | collateral increase 95% CI | residual increase 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for cell_name, metrics in sorted(result["cells"].items()):
        lines.append(
            f"| {cell_name} | "
            f"[{metrics['dcs_loss_ci_low']:.3f}, {metrics['dcs_loss_ci_high']:.3f}] | "
            f"[{metrics['collateral_increase_ci_low']:.3f}, {metrics['collateral_increase_ci_high']:.3f}] | "
            f"[{metrics['residual_increase_ci_low']:.3f}, {metrics['residual_increase_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "For the 10-instance smoke, gates are displayed only to diagnose the",
            "pipeline and must not be treated as manuscript evidence. For the frozen",
            "100-instance pilot, both off-diagonal model directions must pass under",
            "both discourse conditions. Diagonal success cannot rescue off-diagonal",
            "failure.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
