import csv
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from claimpatch.baselines.descendant_all import descendant_all_labels, descendant_all_patch
from claimpatch.baselines.attribute_no_graph import attribute_no_graph_labels, attribute_no_graph_patch
from claimpatch.baselines.full_regeneration import full_regeneration_patch
from claimpatch.baselines.independent_claim_revision import direct_only_labels, direct_only_patch
from claimpatch.baselines.minimal_edit_prompt import unstructured_selective_edit_patch
from claimpatch.data.loaders import write_jsonl
from claimpatch.data.synthetic_generator import generate_synthetic_dataset, iter_jsonl
from claimpatch.data.validators import validate_instances
from claimpatch.evaluation.bootstrap import paired_bootstrap_delta
from claimpatch.evaluation.citation_metrics import citation_metrics
from claimpatch.evaluation.cost_metrics import approximate_cost
from claimpatch.evaluation.impact_metrics import impact_classification_metrics
from claimpatch.evaluation.patch_metrics import patch_metrics
from claimpatch.evaluation.preservation_metrics import preservation_metrics
from claimpatch.impact.propagation import rule_based_impact
from claimpatch.models.llm_client import build_llm_client
from claimpatch.patch.executor import PatchExecutionError, apply_patch_transaction
from claimpatch.patch.generator import build_patch_from_impact
from claimpatch.pipelines.prompted_modules import PromptedGenerationError, PromptedPatchPipeline
from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.results import AnswerVersion, SyntheticInstance
from claimpatch.schemas.updates import ImpactLabel


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"


def _environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "model": "none-rule-based-p0",
    }


def _make_context(system: str, config: Dict[str, Any], shared_client: Any = None) -> Dict[str, Any]:
    if system == "unstructured_selective_edit":
        client = shared_client
        if client is None:
            model_config = dict(config.get("model", {}))
            client = build_llm_client(model_config)
        return {"client": client}
    if system in {"p1_prompted_mock", "p1_prompted_local"}:
        client = shared_client
        if client is None:
            model_config = dict(config.get("model", {}))
            if system == "p1_prompted_mock":
                model_config["backend"] = "mock"
            client = build_llm_client(model_config)
        prompted_config = dict(config.get("prompted", {}))
        return {
            "pipeline": PromptedPatchPipeline(
                client,
                enable_schema_repair=bool(prompted_config.get("enable_schema_repair", True)),
                metadata_ablation=str(prompted_config.get("metadata_ablation", "none")),
            )
        }
    return {}


def _active_claim_ids(answer: AnswerVersion) -> set:
    return {claim.claim_id for claim in answer.claims if claim.status == "active"}


def _failure_metrics(gold_labels: List[ImpactLabel]) -> Dict[str, float]:
    impacted = [label for label in gold_labels if label.state == "MUST_CHANGE"]
    return {
        "impact_macro_f1": 0.0,
        "must_change_f1": 0.0,
        "must_change_precision": 0.0,
        "must_change_recall": 0.0,
        "still_valid_f1": 0.0,
        "still_valid_precision": 0.0,
        "still_valid_recall": 0.0,
        "uncertain_f1": 0.0,
        "uncertain_precision": 0.0,
        "uncertain_recall": 0.0,
        "patch_precision": 0.0,
        "patch_recall": 0.0,
        "dependency_complete_success": 0.0,
        "collateral_edit_rate": 0.0,
        "residual_stale_rate": 1.0 if impacted else 0.0,
        "broken_correct_rate": 0.0,
        "patch_footprint_claims": 0.0,
        "operation_count": 0.0,
        "preserve_precision": 0.0,
        "preserve_recall": 0.0,
        "unsupported_preservation_rate": 0.0,
        "citation_claim_count": 0.0,
        "citation_orphan_rate": 0.0,
        "approx_answer_tokens": 0.0,
        "approx_patch_tokens": 0.0,
        "token_reduction_vs_full": 0.0,
        "transaction_success": 0.0,
        "schema_valid": 0.0,
    }


def _system_patch(
    system: str,
    current: AnswerVersion,
    update,
    fresh: AnswerVersion,
    gold_labels: List[ImpactLabel],
    context: Dict[str, Any],
) -> Tuple[List[ImpactLabel], SemanticPatch]:
    if system == "claimpatch_rule":
        predicted = rule_based_impact(current, update)
        return predicted, build_patch_from_impact(current, fresh, predicted)
    if system == "p1_prompted_mock":
        predicted, patch, _records = context["pipeline"].build_patch(current, update, fresh)
        return predicted, patch
    if system == "p1_prompted_local":
        predicted, patch, _records = context["pipeline"].build_patch(current, update, fresh)
        return predicted, patch
    if system == "direct_only":
        predicted = direct_only_labels(gold_labels)
        return predicted, direct_only_patch(current, fresh, gold_labels)
    if system == "attribute_no_graph":
        predicted = attribute_no_graph_labels(gold_labels, update)
        return predicted, attribute_no_graph_patch(current, update, fresh, gold_labels)
    if system == "descendant_all":
        predicted = descendant_all_labels(current, update, gold_labels)
        return predicted, descendant_all_patch(current, update, fresh, gold_labels)
    if system == "full_regeneration":
        predicted = [
            ImpactLabel(
                claim_id=claim.claim_id,
                state="MUST_CHANGE",
                direct=False,
                reason="Full regeneration rewrites the answer.",
                gold_operation="REPLACE",
            )
            for claim in current.claims
            if claim.status == "active"
        ]
        return predicted, full_regeneration_patch(current, fresh, gold_labels)
    if system == "unstructured_selective_edit":
        return unstructured_selective_edit_patch(current, update, fresh, gold_labels, context.get("client"))
    raise ValueError(f"unknown system: {system}")


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _shared_llm_client_for_run(systems: List[str], config: Dict[str, Any]) -> Any:
    """Reuse one local LLM for ClaimPatch and graph-free LLM baselines.

    Loading Qwen once per prompted system can exceed 24GB GPUs. When the main
    P1 system and the unstructured LLM baseline are evaluated in one process,
    they should share the same client and differ only in prompts/context.
    """

    prompt_systems = {"p1_prompted_local", "unstructured_selective_edit"}
    if len(prompt_systems.intersection(systems)) < 2:
        return None
    model_config = dict(config.get("model", {}))
    if model_config.get("backend", "mock") == "mock":
        return None
    return build_llm_client(model_config)


def _run_one_system(
    system: str,
    instances: List[SyntheticInstance],
    config: Dict[str, Any],
    shared_client: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    patches: List[Dict[str, Any]] = []
    updated_answers: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    context = _make_context(system, config, shared_client=shared_client)
    for instance in instances:
        current = instance.answer_v0
        for step, update in enumerate(instance.updates):
            gold_labels = instance.gold_impact_labels[step]
            fresh = instance.fresh_answers[step]
            predicted = None
            patch = None
            try:
                predicted, patch = _system_patch(system, current, update, fresh, gold_labels, context)
                updated, log = apply_patch_transaction(current, patch, available_evidence=fresh.evidence)
                all_ids = _active_claim_ids(current)
                metrics = {}
                metrics.update(impact_classification_metrics(predicted, gold_labels))
                metrics.update(patch_metrics(patch, gold_labels, all_ids))
                metrics.update(preservation_metrics(patch, gold_labels))
                metrics.update(citation_metrics(updated))
                metrics.update(approximate_cost(current, patch))
                metrics["transaction_success"] = 1.0
                metrics["schema_valid"] = 1.0
                row = {
                    "system": system,
                    "instance_id": instance.instance_id,
                    "step": step + 1,
                    "update_id": update.update_id,
                    **metrics,
                }
                predictions.append(
                    {
                        "system": system,
                        "instance_id": instance.instance_id,
                        "step": step + 1,
                        "update_id": update.update_id,
                        "change_type": update.change_type,
                        "hard_case": update.metadata.get("hard_case", update.change_type),
                        "predicted_impact": [label.model_dump() for label in predicted],
                        "gold_impact": [label.model_dump() for label in gold_labels],
                        "metrics": metrics,
                    }
                )
                patches.append(
                    {
                        "system": system,
                        "instance_id": instance.instance_id,
                        "step": step + 1,
                        "patch": patch.model_dump(),
                        "execution_log": log,
                    }
                )
                updated_answers.append(
                    {
                        "system": system,
                        "instance_id": instance.instance_id,
                        "step": step + 1,
                        "answer": updated.model_dump(),
                    }
                )
                current = updated
            except Exception as exc:
                metrics = _failure_metrics(gold_labels)
                failure = {
                    "system": system,
                    "instance_id": instance.instance_id,
                    "step": step + 1,
                    "update_id": update.update_id,
                    "change_type": update.change_type,
                    "hard_case": update.metadata.get("hard_case", update.change_type),
                    "error": str(exc),
                    "metrics": metrics,
                }
                if isinstance(exc, PromptedGenerationError):
                    failure.update(exc.to_failure_dict())
                if predicted is not None:
                    failure["predicted_impact"] = [label.model_dump() for label in predicted]
                if patch is not None:
                    failure["candidate_patch"] = patch.model_dump()
                failures.append(failure)
                for skipped_step in range(step + 1, len(instance.updates)):
                    skipped_update = instance.updates[skipped_step]
                    skipped_labels = instance.gold_impact_labels[skipped_step]
                    failures.append(
                        {
                            "system": system,
                            "instance_id": instance.instance_id,
                            "step": skipped_step + 1,
                            "update_id": skipped_update.update_id,
                            "change_type": skipped_update.change_type,
                            "hard_case": skipped_update.metadata.get("hard_case", skipped_update.change_type),
                            "error": f"skipped after prior failure at step {step + 1}",
                            "failure_type": "skipped_after_failure",
                            "metrics": _failure_metrics(skipped_labels),
                        }
                    )
                break
    return predictions, patches, updated_answers, failures


def _aggregate(rows_with_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_system: Dict[str, List[Dict[str, float]]] = {}
    for row in rows_with_metrics:
        if "metrics" not in row:
            continue
        by_system.setdefault(row["system"], []).append(row["metrics"])
    summary: Dict[str, Dict[str, float]] = {}
    for system, rows in by_system.items():
        keys = sorted({key for row in rows for key in row.keys() if isinstance(row.get(key), (int, float))})
        summary[system] = {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}
        summary[system]["n_steps"] = float(len(rows))
    return summary


def _aggregate_by_hard_case(rows_with_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in rows_with_metrics:
        if "metrics" not in row:
            continue
        grouped.setdefault((row["system"], row["hard_case"]), []).append(row["metrics"])
    rows: List[Dict[str, Any]] = []
    for (system, hard_case), metric_rows in sorted(grouped.items()):
        keys = sorted({key for row in metric_rows for key in row.keys() if isinstance(row.get(key), (int, float))})
        out: Dict[str, Any] = {"system": system, "hard_case": hard_case, "n_steps": len(metric_rows)}
        for key in keys:
            out[key] = sum(float(row.get(key, 0.0)) for row in metric_rows) / len(metric_rows)
        rows.append(out)
    return rows


def _failure_summary(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], int] = {}
    for failure in failures:
        counts[(failure.get("system", ""), failure.get("failure_type", "runtime"))] = (
            counts.get((failure.get("system", ""), failure.get("failure_type", "runtime")), 0) + 1
        )
    return [
        {"system": system, "failure_type": failure_type, "count": count}
        for (system, failure_type), count in sorted(counts.items())
    ]


def _step_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (str(row["instance_id"]), int(row["step"]))


def _paired_metric_values(
    predictions: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    system: str,
    metric: str,
) -> Dict[Tuple[str, int], float]:
    values: Dict[Tuple[str, int], float] = {}
    for row in predictions:
        if row["system"] == system:
            values[_step_key(row)] = float(row["metrics"].get(metric, 0.0))
    for failure in failures:
        if failure.get("system") == system:
            values[(str(failure["instance_id"]), int(failure["step"]))] = float(
                failure.get("metrics", {}).get(metric, 0.0)
            )
    return values


def _paired_bootstrap_rows(
    predictions: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    primary_system: str,
    baselines: List[str],
    metrics: List[str],
    seed: int,
    samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        rows.extend(
            _paired_bootstrap_rows_for_metric(
                predictions,
                failures,
                primary_system,
                baselines,
                metric,
                seed,
                samples,
            )
        )
    return rows


def _paired_bootstrap_rows_for_metric(
    predictions: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    primary_system: str,
    baselines: List[str],
    metric: str,
    seed: int,
    samples: int,
) -> List[Dict[str, Any]]:
    primary = _paired_metric_values(predictions, failures, primary_system, metric)
    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        other = _paired_metric_values(predictions, failures, baseline, metric)
        common_keys = sorted(set(primary) & set(other))
        if not common_keys:
            rows.append(
                {
                    "comparison": f"{primary_system}-{baseline}",
                    "metric": metric,
                    "mean_delta": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_pairs": 0,
                    "n_primary": len(primary),
                    "n_baseline": len(other),
                    "note": "no_common_pairs",
                }
            )
            continue
        primary_values = [primary[key] for key in common_keys]
        baseline_values = [other[key] for key in common_keys]
        ci = paired_bootstrap_delta(primary_values, baseline_values, seed=seed, samples=samples)
        rows.append(
            {
                "comparison": f"{primary_system}-{baseline}",
                "metric": metric,
                **ci,
                "n_pairs": len(common_keys),
                "n_primary": len(primary),
                "n_baseline": len(other),
                "note": "common_pairs_only" if len(primary) != len(other) else "",
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_p0(config_path: Path, limit: int = None) -> Path:
    start = time.time()
    config = _load_config(config_path)
    seed = int(config.get("seed", 13))
    dataset_config = dict(config.get("dataset", {}))
    size = int(dataset_config.get("size", 100))
    if limit is not None:
        size = int(limit)
    steps = int(dataset_config.get("sequential_steps", 3))
    profile = str(dataset_config.get("profile", "standard"))
    systems = list(
        config.get(
            "systems",
            [
                "claimpatch_rule",
                "direct_only",
                "attribute_no_graph",
                "descendant_all",
                "full_regeneration",
                "unstructured_selective_edit",
            ],
        )
    )
    output_dir = Path(config.get("output_dir", "outputs/p0"))
    stage_title = str(config.get("stage_title", "ClaimPatch P0 Rule-Based Synthetic Experiment"))
    interpretation = str(
        config.get(
            "interpretation",
            "P0 is deterministic. It validates the executable patch language, graph closure, semantic revalidation, preservation accounting, citation rebinding, and transaction rollback/idempotency tests. It is not evidence for LLM robustness or external validity.",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = generate_synthetic_dataset(size=size, seed=seed, sequential_steps=steps, profile=profile)
    errors = validate_instances(instances)
    if errors:
        raise RuntimeError("synthetic validation failed: " + "; ".join(errors[:10]))
    dataset_slug = str(dataset_config.get("name", f"benchmark_results_{profile}_{size}_seed{seed}.jsonl"))
    write_jsonl(Path("data/synthetic") / dataset_slug, iter_jsonl(instances))

    all_predictions: List[Dict[str, Any]] = []
    all_patches: List[Dict[str, Any]] = []
    all_updated: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []
    shared_client = _shared_llm_client_for_run(systems, config)
    for system in systems:
        preds, patches, updated, failures = _run_one_system(system, instances, config, shared_client=shared_client)
        all_predictions.extend(preds)
        all_patches.extend(patches)
        all_updated.extend(updated)
        all_failures.extend(failures)
        if shared_client is None:
            _release_accelerator_memory()

    metric_source_rows = all_predictions + all_failures
    summary = _aggregate(metric_source_rows)
    hard_case_rows = _aggregate_by_hard_case(metric_source_rows)
    bootstrap_samples = int(config.get("bootstrap", {}).get("samples", 1000))
    primary_system = str(config.get("primary_system", "claimpatch_rule" if "claimpatch_rule" in systems else systems[0]))
    bootstrap_against = list(config.get("bootstrap_against", [s for s in ["direct_only", "attribute_no_graph"] if s in systems]))
    bootstrap_rows = _paired_bootstrap_rows(
        all_predictions,
        all_failures,
        primary_system,
        bootstrap_against,
        metrics=list(
            config.get(
                "bootstrap_metrics",
                [
                    "dependency_complete_success",
                    "collateral_edit_rate",
                    "residual_stale_rate",
                    "patch_precision",
                    "patch_recall",
                ],
            )
        ),
        seed=seed,
        samples=bootstrap_samples,
    )

    metric_rows = []
    for system, metrics in summary.items():
        metric_rows.append({"system": system, **metrics})

    _write_jsonl(output_dir / "predictions.jsonl", all_predictions)
    _write_jsonl(output_dir / "patches.jsonl", all_patches)
    _write_jsonl(output_dir / "updated_answers.jsonl", all_updated)
    _write_jsonl(output_dir / "failures.jsonl", all_failures)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "hard_case_metrics.csv", hard_case_rows)
    _write_csv(output_dir / "failure_summary.csv", _failure_summary(all_failures))
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    bootstrap_text = "comparison,metric,mean_delta,ci_low,ci_high,n_pairs,n_primary,n_baseline,note\n"
    for row in bootstrap_rows:
        bootstrap_text += (
            f"{row['comparison']},{row['metric']},{row['mean_delta']},{row['ci_low']},{row['ci_high']},"
            f"{row['n_pairs']},{row['n_primary']},{row['n_baseline']},{row.get('note', '')}\n"
        )
    (output_dir / "bootstrap_ci.csv").write_text(bootstrap_text, encoding="utf-8")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
    elapsed = time.time() - start
    lines = [
        f"# {stage_title}",
        "",
        f"- instances: {size}",
        f"- sequential steps per instance: {steps}",
        f"- dataset profile: {profile}",
        f"- total evaluated update steps: {size * steps}",
        f"- elapsed seconds: {elapsed:.2f}",
        f"- failures: {len(all_failures)}",
        "",
        "## Main metrics",
        "",
        "| system | DCS | patch_precision | patch_recall | collateral_edit_rate | residual_stale_rate | preserve_precision | token_reduction_vs_full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        metrics = summary.get(system, {})
        lines.append(
            f"| {system} | {metrics.get('dependency_complete_success', 0):.3f} | "
            f"{metrics.get('patch_precision', 0):.3f} | {metrics.get('patch_recall', 0):.3f} | "
            f"{metrics.get('collateral_edit_rate', 0):.3f} | {metrics.get('residual_stale_rate', 0):.3f} | "
            f"{metrics.get('preserve_precision', 0):.3f} | {metrics.get('token_reduction_vs_full', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Hard-case breakdown",
            "",
            "| hard_case | system | DCS | patch_precision | collateral_edit_rate | residual_stale_rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    selected_systems = {
        primary_system,
        "claimpatch_rule",
        "p1_prompted_mock",
        "p1_prompted_local",
        "attribute_no_graph",
        "descendant_all",
        "full_regeneration",
        "unstructured_selective_edit",
    }
    for row in hard_case_rows:
        if row["system"] not in selected_systems:
            continue
        lines.append(
            f"| {row['hard_case']} | {row['system']} | "
            f"{row.get('dependency_complete_success', 0):.3f} | "
            f"{row.get('patch_precision', 0):.3f} | "
            f"{row.get('collateral_edit_rate', 0):.3f} | "
            f"{row.get('residual_stale_rate', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            *[
                f"- {row['metric']} delta, {row['comparison']}: {row['mean_delta']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                f"(paired n={row['n_pairs']}, primary n={row['n_primary']}, baseline n={row['n_baseline']})"
                for row in bootstrap_rows
            ],
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )
    if all_failures:
        lines.extend(
            [
                "",
                "## Failure summary",
                "",
                "| system | failure_type | count |",
                "|---|---|---:|",
            ]
        )
        for row in _failure_summary(all_failures):
            lines.append(f"| {row['system']} | {row['failure_type']} | {row['count']} |")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir / "summary.md"


def run_p1_prompted(config_path: Path, limit: int = None) -> Path:
    return run_p0(config_path, limit=limit)
