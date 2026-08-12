import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from decap.impact.propagation import rule_based_impact
from decap.models.llm_client import MockLLMClient
from decap.models.structured_generation import (
    StructuredGenerationRecord,
    extract_json_object,
    load_prompt,
    render_prompt,
)
from decap.patch.generator import build_patch_from_impact
from decap.schemas.graph import DependencyEdge
from decap.schemas.patches import SemanticPatch
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


_ALLOWED_DEPENDENCY_TYPES = {"numeric", "logical", "comparative", "temporal", "causal", "citation", "scope", "other"}
_OPERATION_AS_STATE = {"REPLACE", "DELETE", "INSERT", "SPLIT", "MERGE", "REBIND", "INVALIDATE"}


class PromptedGenerationError(RuntimeError):
    """Structured generation failure with raw output for audit."""

    def __init__(self, task: str, reason: str, raw_text: str = "", prompt: str = ""):
        super().__init__(f"{task} structured generation failed: {reason}")
        self.task = task
        self.reason = reason
        self.raw_text = raw_text
        self.prompt = prompt

    def to_failure_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": "structured_generation",
            "task": self.task,
            "reason": self.reason,
            "raw_text": self.raw_text,
            "raw_chars": len(self.raw_text),
            "prompt_excerpt": self.prompt[:1000],
            "prompt_chars": len(self.prompt),
        }


def _claims_by_id(answer: AnswerVersion) -> Dict[str, Dict[str, Any]]:
    return {claim.claim_id: claim.model_dump() for claim in answer.claims if claim.status == "active"}


def _repair_dependency_updates(raw_updates: Any, repair_log: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(raw_updates, list):
        if raw_updates not in (None, ""):
            repair_log.append("dependency_updates_non_list_dropped")
        return []
    repaired: List[Dict[str, Any]] = []
    for item in raw_updates:
        if not isinstance(item, dict):
            repair_log.append("dependency_update_non_object_dropped")
            continue
        if not item.get("source_claim_ids") or not item.get("target_claim_id") or not item.get("dependency_type"):
            repair_log.append("dependency_update_incomplete_dropped")
            continue
        fixed = deepcopy(item)
        fixed.setdefault("confidence", 1.0)
        fixed.setdefault("rule", None)
        fixed.setdefault("metadata", {})
        repaired.append(fixed)
    return repaired


def _repair_dependency_edge_payloads(raw_dependencies: Any) -> Tuple[Any, List[str]]:
    if not isinstance(raw_dependencies, list):
        return raw_dependencies, []
    repaired: List[Any] = []
    repair_log: List[str] = []
    for item in raw_dependencies:
        if not isinstance(item, dict):
            repaired.append(item)
            continue
        target_ids = item.get("target_claim_ids")
        if "target_claim_id" not in item and isinstance(target_ids, list) and target_ids:
            for target_id in target_ids:
                fixed = deepcopy(item)
                fixed["target_claim_id"] = target_id
                fixed.pop("target_claim_ids", None)
                _repair_dependency_edge_type(fixed, repair_log)
                repaired.append(fixed)
            repair_log.append("dependency_target_claim_ids_expanded")
            continue
        fixed = deepcopy(item)
        fixed.pop("target_claim_ids", None)
        _repair_dependency_edge_type(fixed, repair_log)
        repaired.append(fixed)
    return repaired, repair_log


def _repair_dependency_edge_type(edge: Dict[str, Any], repair_log: List[str]) -> None:
    dependency_type = edge.get("dependency_type")
    if dependency_type not in _ALLOWED_DEPENDENCY_TYPES:
        metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        metadata["schema_repair_original_dependency_type"] = dependency_type
        edge["metadata"] = metadata
        edge["dependency_type"] = "other"
        repair_log.append(f"dependency_type_to_other:{dependency_type}")


def _repair_impact_label_payloads(answer: AnswerVersion, raw_labels: Any) -> Tuple[Any, List[str]]:
    if not isinstance(raw_labels, list):
        return raw_labels, []
    active_ids = [claim.claim_id for claim in answer.claims if claim.status == "active"]
    active_set = set(active_ids)
    by_id: Dict[str, Dict[str, Any]] = {}
    repair_log: List[str] = []
    for raw in raw_labels:
        if not isinstance(raw, dict):
            repair_log.append("impact_label_non_object_dropped")
            continue
        item = deepcopy(raw)
        claim_id = item.get("claim_id")
        if claim_id not in active_set:
            repair_log.append(f"impact_label_non_active_dropped:{claim_id}")
            continue
        if claim_id in by_id:
            repair_log.append(f"impact_label_duplicate_last_wins:{claim_id}")
        state = item.get("state")
        if state in _OPERATION_AS_STATE:
            item["state"] = "MUST_CHANGE"
            item["gold_operation"] = state
            repair_log.append(f"impact_state_operation_normalized:{claim_id}:{state}")
        item.setdefault("direct", False)
        item.setdefault("reason", "Deterministic schema repair: model omitted impact reason.")
        if item.get("state") == "MUST_CHANGE" and item.get("gold_operation") is None:
            item["gold_operation"] = "REPLACE"
            repair_log.append(f"impact_operation_default_replace:{claim_id}")
        by_id[str(claim_id)] = item
    for claim_id in active_ids:
        if claim_id not in by_id:
            by_id[claim_id] = {
                "claim_id": claim_id,
                "state": "STILL_VALID",
                "direct": False,
                "reason": "Deterministic schema repair: model omitted this active claim; interpreted as preserved.",
                "gold_operation": None,
            }
            repair_log.append(f"impact_label_missing_filled_still_valid:{claim_id}")
    return [by_id[claim_id] for claim_id in active_ids], repair_log


def _repair_semantic_patch_payload(
    payload: Dict[str, Any],
    current: AnswerVersion,
    fresh: AnswerVersion,
    labels: List[ImpactLabel],
) -> Tuple[Dict[str, Any], List[str]]:
    """Deterministically repair schema-formatting errors without changing impact decisions."""

    repaired = deepcopy(payload)
    repair_log: List[str] = []
    current_claims = _claims_by_id(current)
    fresh_claims = _claims_by_id(fresh)
    active_claim_ids = [claim.claim_id for claim in current.claims if claim.status == "active"]
    label_by_id = {label.claim_id: label for label in labels}
    must_change_ids = {label.claim_id for label in labels if label.state == "MUST_CHANGE"}
    if repaired.get("answer_id") != current.answer_id:
        repaired["answer_id"] = current.answer_id
        repair_log.append("patch_answer_id_aligned_to_current")
    if repaired.get("from_version") != current.version:
        repaired["from_version"] = current.version
        repair_log.append("patch_from_version_aligned_to_current")
    if repaired.get("to_version") != fresh.version:
        repaired["to_version"] = fresh.version
        repair_log.append("patch_to_version_aligned_to_fresh")
    repaired.setdefault("preconditions", [])
    repaired.setdefault("operations", [])
    repaired.setdefault("preserve_claim_ids", [])
    repaired.setdefault("postconditions", [])
    repaired.setdefault("metadata", {})

    if not isinstance(repaired["preconditions"], list):
        repaired["preconditions"] = []
        repair_log.append("preconditions_non_list_reset")
    preconditions_by_id: Dict[str, Dict[str, Any]] = {}
    for precondition in repaired["preconditions"]:
        if isinstance(precondition, dict):
            claim_id = precondition.get("claim_id")
            current_claim = current_claims.get(claim_id)
            if current_claim is not None:
                if precondition.get("expected_version") != current_claim["version"]:
                    precondition["expected_version"] = current_claim["version"]
                    repair_log.append(f"precondition_version_aligned:{claim_id}")
                if precondition.get("expected_status") != current_claim["status"]:
                    precondition["expected_status"] = current_claim["status"]
                    repair_log.append(f"precondition_status_aligned:{claim_id}")
            else:
                precondition.setdefault("expected_status", "active")
            if claim_id in current_claims:
                preconditions_by_id[str(claim_id)] = precondition
    for claim_id in active_claim_ids:
        if claim_id not in preconditions_by_id:
            claim = current_claims[claim_id]
            preconditions_by_id[claim_id] = {
                "claim_id": claim_id,
                "expected_version": claim["version"],
                "expected_status": claim["status"],
            }
            repair_log.append(f"precondition_added:{claim_id}")
    repaired["preconditions"] = [preconditions_by_id[claim_id] for claim_id in active_claim_ids]

    if not isinstance(repaired["operations"], list):
        repaired["operations"] = []
        repair_log.append("operations_non_list_reset")

    normalized_operations: List[Dict[str, Any]] = []
    for op in repaired["operations"]:
        if not isinstance(op, dict):
            continue
        op.setdefault("claim_ids", [])
        op.setdefault("new_claims", [])
        op.setdefault("old_evidence_ids", [])
        op.setdefault("new_evidence_ids", [])
        claim_ids = [claim_id for claim_id in op.get("claim_ids", []) if claim_id in current_claims]
        if claim_ids and not any(claim_id in must_change_ids for claim_id in claim_ids):
            repair_log.append(f"operation_dropped_for_preserved_claims:{','.join(claim_ids)}")
            continue
        op["claim_ids"] = claim_ids
        if not op.get("reason"):
            op["reason"] = "Deterministic schema repair: model omitted operation reason."
            repair_log.append("operation_reason_filled")
        op["dependency_updates"] = _repair_dependency_updates(op.get("dependency_updates", []), repair_log)

        if len(claim_ids) == 1 and claim_ids[0] in label_by_id:
            claim_id = claim_ids[0]
            desired_op = label_by_id[claim_id].gold_operation or "REPLACE"
            if op.get("op") != desired_op:
                op["op"] = desired_op
                repair_log.append(f"operation_aligned_to_impact:{claim_id}:{desired_op}")
            if desired_op == "REBIND":
                current_claim = current_claims[claim_id]
                fresh_claim = fresh_claims.get(claim_id)
                op["new_claims"] = []
                op["old_evidence_ids"] = list(current_claim.get("evidence_ids", []))
                op["new_evidence_ids"] = list(fresh_claim.get("evidence_ids", [])) if fresh_claim else []
            elif desired_op == "REPLACE" and claim_id in fresh_claims:
                if not op.get("new_claims"):
                    repair_log.append("replace_new_claim_filled_from_fresh_target")
                    repair_log.append(f"replace_new_claim_filled_from_fresh_target:{claim_id}")
                op["new_claims"] = [fresh_claims[claim_id]]
        if op.get("op") == "REPLACE" and not op.get("new_claims") and len(op.get("claim_ids", [])) == 1:
            claim_id = op["claim_ids"][0]
            if claim_id in fresh_claims:
                op["new_claims"] = [fresh_claims[claim_id]]
                repair_log.append(f"replace_new_claim_filled_from_fresh_target:{claim_id}")
        if op.get("op") == "REBIND":
            op["new_claims"] = []
        normalized_operations.append(op)
    repaired["operations"] = normalized_operations

    touched_ids = {
        claim_id
        for op in repaired["operations"]
        if isinstance(op, dict)
        for claim_id in op.get("claim_ids", [])
    }
    for claim_id in active_claim_ids:
        label = label_by_id.get(claim_id)
        if label is None or label.state != "MUST_CHANGE" or claim_id in touched_ids:
            continue
        current_claim = current_claims[claim_id]
        fresh_claim = fresh_claims.get(claim_id)
        operation = label.gold_operation or "REPLACE"
        if operation == "REBIND":
            repaired["operations"].append(
                {
                    "op": "REBIND",
                    "claim_ids": [claim_id],
                    "new_claims": [],
                    "old_evidence_ids": list(current_claim.get("evidence_ids", [])),
                    "new_evidence_ids": list(fresh_claim.get("evidence_ids", [])) if fresh_claim else [],
                    "reason": label.reason,
                    "dependency_updates": [],
                }
            )
        else:
            repaired["operations"].append(
                {
                    "op": "REPLACE",
                    "claim_ids": [claim_id],
                    "new_claims": [fresh_claim] if fresh_claim else [],
                    "old_evidence_ids": list(current_claim.get("evidence_ids", [])),
                    "new_evidence_ids": list(fresh_claim.get("evidence_ids", [])) if fresh_claim else [],
                    "reason": label.reason,
                    "dependency_updates": [],
                }
            )
        repair_log.append(f"missing_operation_filled_from_impact:{claim_id}")

    repaired["preserve_claim_ids"] = [
        claim_id
        for claim_id in active_claim_ids
        if label_by_id.get(claim_id) is None or label_by_id[claim_id].state != "MUST_CHANGE"
    ]

    metadata = repaired["metadata"] if isinstance(repaired.get("metadata"), dict) else {}
    metadata["schema_repair_applied"] = bool(repair_log)
    metadata["schema_repair_log"] = repair_log
    repaired["metadata"] = metadata
    return repaired, repair_log


class PromptedDependencyExtractor:
    """P1 structured dependency extraction interface with deterministic mock backend."""

    prompt_name = "dependency_extraction"
    prompt_version = "v1"

    def __init__(self, client: MockLLMClient = None, enable_schema_repair: bool = True):
        self.client = client or MockLLMClient()
        self.enable_schema_repair = enable_schema_repair

    def predict(self, answer: AnswerVersion) -> Tuple[List[DependencyEdge], StructuredGenerationRecord]:
        template = load_prompt(self.prompt_name, self.prompt_version)
        prompt = render_prompt(
            template,
            question=answer.question,
            claims=[claim.model_dump() for claim in answer.claims if claim.status == "active"],
        )
        payload = {"dependencies": [edge.model_dump() for edge in answer.dependencies]}
        raw = self.client.generate_text(prompt, task=self.prompt_name, payload=payload)
        try:
            parsed = extract_json_object(raw)
            if self.enable_schema_repair:
                parsed["dependencies"], repair_log = _repair_dependency_edge_payloads(parsed["dependencies"])
                if repair_log:
                    parsed.setdefault("metadata", {})
                    parsed["metadata"]["schema_repair_log"] = repair_log
            deps = [DependencyEdge.model_validate(item) for item in parsed["dependencies"]]
        except Exception as exc:
            raise PromptedGenerationError(self.prompt_name, str(exc), raw_text=raw, prompt=prompt) from exc
        record = StructuredGenerationRecord(
            task=self.prompt_name,
            prompt_version=self.prompt_version,
            model_name=self.client.model_name,
            prompt=prompt,
            raw_text=raw,
            parsed=parsed,
        )
        return deps, record


class PromptedImpactPredictor:
    """P1 structured impact predictor interface.

    The default backend is deterministic. It exercises prompt/render/parse/schema
    validation without making claims about real LLM performance.
    """

    prompt_name = "impact_classification"
    prompt_version = "v1"

    def __init__(
        self,
        client: MockLLMClient = None,
        enable_schema_repair: bool = True,
        metadata_ablation: str = "none",
    ):
        self.client = client or MockLLMClient()
        self.enable_schema_repair = enable_schema_repair
        if metadata_ablation not in {"none", "soft", "hard"}:
            raise ValueError(f"unknown metadata_ablation: {metadata_ablation}")
        self.metadata_ablation = metadata_ablation

    def predict(
        self,
        answer: AnswerVersion,
        update: EvidenceUpdate,
        dependencies: List[DependencyEdge],
    ) -> Tuple[List[ImpactLabel], StructuredGenerationRecord]:
        template = load_prompt(self.prompt_name, self.prompt_version)
        prompt_update = _ablate_update_for_prompt(update, self.metadata_ablation)
        prompt = render_prompt(
            template,
            answer=answer.model_dump(),
            update=prompt_update,
            dependencies=[edge.model_dump() for edge in dependencies],
        )
        labels = rule_based_impact(answer, update)
        payload = {"impact_labels": [label.model_dump() for label in labels]}
        raw = self.client.generate_text(prompt, task=self.prompt_name, payload=payload)
        try:
            parsed = extract_json_object(raw)
            if self.enable_schema_repair:
                parsed["impact_labels"], repair_log = _repair_impact_label_payloads(answer, parsed["impact_labels"])
                if repair_log:
                    parsed.setdefault("metadata", {})
                    parsed["metadata"]["schema_repair_log"] = repair_log
            predicted = [ImpactLabel.model_validate(item) for item in parsed["impact_labels"]]
            predicted = _repair_impact_operation_consistency(answer, predicted)
            _validate_complete_impact_labels(answer, predicted)
        except Exception as exc:
            raise PromptedGenerationError(self.prompt_name, str(exc), raw_text=raw, prompt=prompt) from exc
        record = StructuredGenerationRecord(
            task=self.prompt_name,
            prompt_version=self.prompt_version,
            model_name=self.client.model_name,
            prompt=prompt,
            raw_text=raw,
            parsed=parsed,
        )
        return predicted, record


def _repair_impact_operation_consistency(answer: AnswerVersion, labels: List[ImpactLabel]) -> List[ImpactLabel]:
    """Repair operation labels that contradict claim type.

    This does not add/remove impacted claims. It only fixes invalid operation
    choices such as REBIND on numeric text changes.
    """

    claim_types = {claim.claim_id: claim.claim_type for claim in answer.claims if claim.status == "active"}
    repaired: List[ImpactLabel] = []
    for label in labels:
        if label.state != "MUST_CHANGE":
            repaired.append(label)
            continue
        claim_type = claim_types.get(label.claim_id)
        operation = label.gold_operation
        if claim_type == "citation_only":
            operation = "REBIND"
        elif operation == "REBIND" or operation is None:
            operation = "REPLACE"
        repaired.append(
            ImpactLabel(
                claim_id=label.claim_id,
                state=label.state,
                direct=label.direct,
                reason=label.reason,
                gold_operation=operation,
            )
        )
    return repaired


def _validate_complete_impact_labels(answer: AnswerVersion, labels: List[ImpactLabel]) -> None:
    active_ids = [claim.claim_id for claim in answer.claims if claim.status == "active"]
    active_set = set(active_ids)
    label_ids = [label.claim_id for label in labels]
    label_set = set(label_ids)
    duplicates = sorted({claim_id for claim_id in label_ids if label_ids.count(claim_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate impact labels: {duplicates}")
    missing = sorted(active_set - label_set)
    extra = sorted(label_set - active_set)
    if missing or extra:
        raise ValueError(f"impact labels must cover active claims exactly; missing={missing}; extra={extra}")


def _ablate_update_for_prompt(update: EvidenceUpdate, mode: str) -> Dict[str, Any]:
    """Return an input-only ablated update payload for the impact prompt.

    The original EvidenceUpdate object is not mutated. Gold impact labels,
    evaluator grouping, baselines, and patch execution continue to use the
    original update. This function only controls what the prompted impact module
    can read.

    Modes:
    - none: original update payload.
    - soft: remove explicit category labels and hard-case labels, while keeping
      numeric old/new values and the explicit changed attribute.
    - hard: remove shortcut metadata, including changed-attribute fields and
      evidence-record metadata; neutralize evidence IDs/URIs so the prompt is
      driven by the evidence text rather than structured benchmark adapter
      labels.
    """

    if mode == "none":
        return update.model_dump()
    if mode not in {"soft", "hard"}:
        raise ValueError(f"unknown metadata ablation mode: {mode}")

    payload = deepcopy(update.model_dump())
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = deepcopy(metadata)

    for key in ["hard_case", "old_category", "new_category"]:
        metadata.pop(key, None)

    if mode == "hard":
        # Remove structured adapter fields that expose slot identity or
        # precomputed numeric/category state. The textual evidence remains.
        for key in list(metadata):
            if key != "domain":
                metadata.pop(key, None)
        for collection_name in ["added_evidence", "modified_evidence"]:
            for index, evidence in enumerate(payload.get(collection_name, []) or [], 1):
                if not isinstance(evidence, dict):
                    continue
                evidence["evidence_id"] = f"{collection_name}_{index}"
                evidence["source_uri"] = None
                evidence["metadata"] = {}
        payload["removed_evidence_ids"] = [
            f"removed_evidence_{index}"
            for index, _ in enumerate(payload.get("removed_evidence_ids", []) or [], 1)
        ]

    payload["metadata"] = metadata
    return payload


class PromptedPatchGenerator:
    """P1 structured patch generator interface with schema validation."""

    prompt_name = "patch_generation"
    prompt_version = "v1"

    def __init__(self, client: MockLLMClient = None, enable_schema_repair: bool = True):
        self.client = client or MockLLMClient()
        self.enable_schema_repair = enable_schema_repair

    def generate(
        self,
        current: AnswerVersion,
        fresh: AnswerVersion,
        labels: List[ImpactLabel],
    ) -> Tuple[SemanticPatch, StructuredGenerationRecord]:
        template = load_prompt(self.prompt_name, self.prompt_version)
        prompt = render_prompt(
            template,
            current_answer=current.model_dump(),
            fresh_target=fresh.model_dump(),
            impact_labels=[label.model_dump() for label in labels],
        )
        patch = build_patch_from_impact(current, fresh, labels)
        patch.metadata["generator"] = "p1_prompted"
        patch.metadata["model_name"] = self.client.model_name
        patch.metadata["prompt_versions"] = {
            "patch_generation": self.prompt_version,
            "impact_classification": PromptedImpactPredictor.prompt_version,
            "dependency_extraction": PromptedDependencyExtractor.prompt_version,
        }
        payload = {"semantic_patch": patch.model_dump()}
        raw = self.client.generate_text(prompt, task=self.prompt_name, payload=payload)
        try:
            parsed = extract_json_object(raw)
            if self.enable_schema_repair:
                repaired_patch_payload, repair_log = _repair_semantic_patch_payload(
                    parsed["semantic_patch"],
                    current,
                    fresh,
                    labels,
                )
                parsed["semantic_patch"] = repaired_patch_payload
                predicted_patch = SemanticPatch.model_validate(repaired_patch_payload)
                predicted_patch.metadata["schema_repair_applied"] = bool(repair_log)
                predicted_patch.metadata["schema_repair_log"] = repair_log
            else:
                predicted_patch = SemanticPatch.model_validate(parsed["semantic_patch"])
                predicted_patch.metadata["schema_repair_applied"] = False
                predicted_patch.metadata["schema_repair_log"] = []
        except Exception as exc:
            raise PromptedGenerationError(self.prompt_name, str(exc), raw_text=raw, prompt=prompt) from exc
        record = StructuredGenerationRecord(
            task=self.prompt_name,
            prompt_version=self.prompt_version,
            model_name=self.client.model_name,
            prompt=prompt,
            raw_text=raw,
            parsed=parsed,
        )
        return predicted_patch, record


class PromptedPatchPipeline:
    """P1 end-to-end structured module chain."""

    def __init__(
        self,
        client: MockLLMClient = None,
        enable_schema_repair: bool = True,
        metadata_ablation: str = "none",
    ):
        self.client = client or MockLLMClient()
        self.dependency = PromptedDependencyExtractor(self.client, enable_schema_repair=enable_schema_repair)
        self.impact = PromptedImpactPredictor(
            self.client,
            enable_schema_repair=enable_schema_repair,
            metadata_ablation=metadata_ablation,
        )
        self.patch = PromptedPatchGenerator(self.client, enable_schema_repair=enable_schema_repair)

    def build_patch(
        self,
        current: AnswerVersion,
        update: EvidenceUpdate,
        fresh: AnswerVersion,
    ) -> Tuple[List[ImpactLabel], SemanticPatch, List[StructuredGenerationRecord]]:
        dependencies, dep_record = self.dependency.predict(current)
        labels, impact_record = self.impact.predict(current, update, dependencies)
        patch, patch_record = self.patch.generate(current, fresh, labels)
        patch.metadata["p1_records"] = [
            {
                "task": record.task,
                "prompt_version": record.prompt_version,
                "model_name": record.model_name,
                "raw_chars": len(record.raw_text),
            }
            for record in [dep_record, impact_record, patch_record]
        ]
        return labels, patch, [dep_record, impact_record, patch_record]
