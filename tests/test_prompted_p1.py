from pathlib import Path

from claimpatch.data.synthetic_generator import generate_benchmark_instance
from claimpatch.baselines.minimal_edit_prompt import _normalize_unstructured_impact_labels, unstructured_selective_edit_patch
from claimpatch.models.structured_generation import extract_json_object, load_prompt
from claimpatch.pipelines.prompted_modules import (
    PromptedPatchPipeline,
    _ablate_update_for_prompt,
    _repair_dependency_edge_payloads,
    _repair_impact_label_payloads,
    _repair_impact_operation_consistency,
    _repair_semantic_patch_payload,
)
from claimpatch.pipelines.run_experiment import run_p1_prompted
from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.updates import ImpactLabel


def test_extract_json_object_from_wrapped_text():
    parsed = extract_json_object("prefix\n{\"a\": 1}\nsuffix")
    assert parsed["a"] == 1


def test_load_prompt_v1():
    text = load_prompt("impact_classification", "v1")
    assert "MUST_CHANGE" in text


def test_load_unstructured_selective_edit_prompt_v1():
    text = load_prompt("unstructured_selective_edit", "v1")
    assert "dependency graph" in text


def test_prompted_pipeline_builds_schema_valid_patch():
    instance = generate_benchmark_instance(0)
    labels, patch, records = PromptedPatchPipeline().build_patch(
        instance.answer_v0,
        instance.updates[0],
        instance.fresh_answers[0],
    )
    assert labels
    assert patch.operations
    assert patch.metadata["generator"] == "p1_prompted"
    assert len(records) == 3


def test_prompted_pipeline_can_disable_schema_repair():
    instance = generate_benchmark_instance(0)
    labels, patch, records = PromptedPatchPipeline(enable_schema_repair=False).build_patch(
        instance.answer_v0,
        instance.updates[0],
        instance.fresh_answers[0],
    )
    assert labels
    assert patch.metadata["schema_repair_applied"] is False


def test_metadata_ablation_is_input_only_and_soft_removes_category_labels():
    instance = generate_benchmark_instance(0)
    update = next(
        update
        for update in instance.updates
        if update.metadata["hard_case"] == "metric_revision_threshold_cross"
    )
    original = update.model_dump()
    ablated = _ablate_update_for_prompt(update, "soft")
    assert update.model_dump() == original
    assert "old_category" not in ablated["metadata"]
    assert "new_category" not in ablated["metadata"]
    assert "hard_case" not in ablated["metadata"]
    assert ablated["metadata"]["attribute"] in {"a_accuracy", "b_accuracy"}
    assert "old_diff" in ablated["metadata"]
    assert ablated["modified_evidence"][0]["metadata"]["attribute"] in {"a_accuracy", "b_accuracy"}


def test_metadata_ablation_hard_removes_structured_adapter_hints():
    instance = generate_benchmark_instance(0)
    update = next(
        update
        for update in instance.updates
        if update.metadata["hard_case"] == "metric_revision_threshold_cross"
    )
    original = update.model_dump()
    ablated = _ablate_update_for_prompt(update, "hard")
    assert update.model_dump() == original
    assert ablated["metadata"] == {}
    assert ablated["modified_evidence"]
    evidence = ablated["modified_evidence"][0]
    assert evidence["metadata"] == {}
    assert evidence["source_uri"] is None
    assert evidence["evidence_id"] == "modified_evidence_1"
    assert "accuracy is" in evidence["text"]


def test_run_p1_writes_summary():
    summary = run_p1_prompted(Path("configs/experiments/p1_prompted.yaml"))
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "P1 prompted-mock" in text
    bootstrap = (summary.parent / "bootstrap_ci.csv").read_text(encoding="utf-8")
    assert "collateral_edit_rate" in bootstrap
    assert "p1_prompted_mock-unstructured_selective_edit" in bootstrap


def test_unstructured_selective_edit_mock_is_graph_free():
    instance = generate_benchmark_instance(0)
    step = next(i for i, update in enumerate(instance.updates) if update.metadata["attribute"] == "a_accuracy")
    labels, patch = unstructured_selective_edit_patch(
        instance.answer_v0,
        instance.updates[step],
        instance.fresh_answers[step],
        instance.gold_impact_labels[step],
    )
    by_id = {label.claim_id: label for label in labels}
    touched = {claim_id for op in patch.operations for claim_id in op.claim_ids}
    assert by_id["c_a_acc"].state == "MUST_CHANGE"
    assert by_id["c_diff"].state == "STILL_VALID"
    assert "c_diff" not in touched


def test_unstructured_selective_edit_normalizes_common_loose_output():
    instance = generate_benchmark_instance(0)
    raw = [
        {
            "claim_id": "c_a_acc",
            "state": "MUST_CHANGE",
            "direct": True,
            "reason": "changed",
            "gold_operation": "REPLACE",
        },
        {
            "claim_id": "c_citation",
            "state": "REBIND",
            "direct": False,
            "reason": "citation moved",
            "gold_operation": "REBIND",
        },
    ]
    labels = _normalize_unstructured_impact_labels(instance.answer_v0, raw)
    by_id = {label.claim_id: label for label in labels}
    assert len(labels) == len([claim for claim in instance.answer_v0.claims if claim.status == "active"])
    assert by_id["c_citation"].state == "MUST_CHANGE"
    assert by_id["c_citation"].gold_operation == "REBIND"
    assert by_id["c_b_acc"].state == "STILL_VALID"


def test_patch_repair_fills_missing_reason_and_new_claim():
    instance = generate_benchmark_instance(0)
    raw_patch = {
        "patch_id": "p",
        "answer_id": instance.answer_v0.answer_id,
        "from_version": 0,
        "to_version": 1,
        "preconditions": [{"claim_id": "c_a_acc", "expected_version": 0}],
        "operations": [
            {
                "op": "REPLACE",
                "claim_ids": ["c_a_acc"],
                "new_claims": [],
                "dependency_updates": [],
            }
        ],
        "preserve_claim_ids": [],
    }
    repaired, log = _repair_semantic_patch_payload(
        raw_patch,
        instance.answer_v0,
        instance.fresh_answers[0],
        instance.gold_impact_labels[0],
    )
    patch = SemanticPatch.model_validate(repaired)
    assert patch.operations[0].reason
    assert patch.operations[0].new_claims[0].claim_id == "c_a_acc"
    assert "operation_reason_filled" in log
    assert "replace_new_claim_filled_from_fresh_target" in log


def test_patch_repair_drops_incomplete_dependency_updates():
    instance = generate_benchmark_instance(0)
    raw_patch = {
        "patch_id": "p",
        "answer_id": instance.answer_v0.answer_id,
        "from_version": 0,
        "to_version": 1,
        "preconditions": [],
        "operations": [
            {
                "op": "REBIND",
                "claim_ids": ["c_citation"],
                "new_evidence_ids": ["e_inst_0000_a_accuracy_v1"],
                "reason": "citation refresh",
                "dependency_updates": [
                    {"source_claim_ids": ["c_a_acc"], "target_claim_id": "c_citation"}
                ],
            }
        ],
        "preserve_claim_ids": [],
    }
    repaired, log = _repair_semantic_patch_payload(
        raw_patch,
        instance.answer_v0,
        instance.fresh_answers[0],
        instance.gold_impact_labels[0],
    )
    patch = SemanticPatch.model_validate(repaired)
    assert patch.operations[0].dependency_updates == []
    assert "dependency_update_incomplete_dropped" in log


def test_dependency_repair_maps_unknown_type_to_other():
    raw = [
        {
            "source_claim_ids": ["c_split"],
            "target_claim_id": "c_comparison",
            "dependency_type": "factual",
            "confidence": 0.8,
            "rule": "loose local model type",
            "metadata": {},
        }
    ]
    repaired, log = _repair_dependency_edge_payloads(raw)
    assert repaired[0]["dependency_type"] == "other"
    assert repaired[0]["metadata"]["schema_repair_original_dependency_type"] == "factual"
    assert log == ["dependency_type_to_other:factual"]


def test_dependency_repair_expands_plural_target_claim_ids():
    raw = [
        {
            "source_claim_ids": ["c_split"],
            "target_claim_ids": ["c_a_acc", "c_b_acc"],
            "dependency_type": "scope",
            "confidence": 1.0,
            "rule": "plural target emitted by local model",
            "metadata": {},
        }
    ]
    repaired, log = _repair_dependency_edge_payloads(raw)
    assert [edge["target_claim_id"] for edge in repaired] == ["c_a_acc", "c_b_acc"]
    assert "target_claim_ids" not in repaired[0]
    assert "dependency_target_claim_ids_expanded" in log


def test_impact_repair_fills_missing_active_claim_as_still_valid():
    instance = generate_benchmark_instance(0)
    raw = [
        {
            "claim_id": "c_a_acc",
            "state": "MUST_CHANGE",
            "direct": True,
            "reason": "changed",
            "gold_operation": "REPLACE",
        }
    ]
    repaired, log = _repair_impact_label_payloads(instance.answer_v0, raw)
    by_id = {item["claim_id"]: item for item in repaired}
    assert by_id["c_split"]["state"] == "STILL_VALID"
    assert "impact_label_missing_filled_still_valid:c_split" in log


def test_patch_repair_aligns_preconditions_to_current_versions():
    instance = generate_benchmark_instance(0)
    raw_patch = instance.gold_patches[0].model_dump()
    raw_patch["from_version"] = 99
    raw_patch["preconditions"][0]["expected_version"] = 99
    repaired, log = _repair_semantic_patch_payload(
        raw_patch,
        instance.answer_v0,
        instance.fresh_answers[0],
        instance.gold_impact_labels[0],
    )
    assert repaired["from_version"] == instance.answer_v0.version
    assert repaired["preconditions"][0]["expected_version"] == instance.answer_v0.claims[0].version
    assert "patch_from_version_aligned_to_current" in log


def test_patch_repair_fills_missing_operations_from_impact_labels():
    instance = generate_benchmark_instance(0)
    raw_patch = {
        "patch_id": "p",
        "answer_id": instance.answer_v0.answer_id,
        "from_version": 0,
        "to_version": 1,
        "preconditions": [],
        "operations": [],
        "preserve_claim_ids": [],
    }
    repaired, log = _repair_semantic_patch_payload(
        raw_patch,
        instance.answer_v0,
        instance.fresh_answers[0],
        instance.gold_impact_labels[0],
    )
    patch = SemanticPatch.model_validate(repaired)
    touched = {claim_id for op in patch.operations for claim_id in op.claim_ids}
    must_change = {label.claim_id for label in instance.gold_impact_labels[0] if label.state == "MUST_CHANGE"}
    assert must_change.issubset(touched)
    assert any(item.startswith("missing_operation_filled_from_impact:") for item in log)


def test_impact_operation_repair_fixes_rebind_on_numeric_claim():
    instance = generate_benchmark_instance(5)
    labels = [
        ImpactLabel(
            claim_id="c_b_acc",
            state="MUST_CHANGE",
            direct=True,
            reason="model chose wrong op",
            gold_operation="REBIND",
        ),
        ImpactLabel(
            claim_id="c_citation",
            state="MUST_CHANGE",
            direct=False,
            reason="citation refresh",
            gold_operation="REPLACE",
        ),
    ]
    repaired = _repair_impact_operation_consistency(instance.answer_v0, labels)
    by_id = {label.claim_id: label for label in repaired}
    assert by_id["c_b_acc"].gold_operation == "REPLACE"
    assert by_id["c_citation"].gold_operation == "REBIND"
