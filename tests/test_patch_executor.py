import pytest

from decap.data.synthetic_generator import generate_benchmark_instance
from decap.patch.executor import PatchExecutionError, apply_patch_transaction
from decap.schemas.claims import ClaimNode
from decap.schemas.patches import PatchOperation, PatchPrecondition, SemanticPatch


def _instance():
    return generate_benchmark_instance(10)


def test_oracle_replace_patch_transaction_success():
    inst = _instance()
    updated, log = apply_patch_transaction(inst.answer_v0, inst.gold_patches[0])
    assert updated.version == 1
    assert log


def test_patch_idempotency_fails_second_application():
    inst = _instance()
    updated, _ = apply_patch_transaction(inst.answer_v0, inst.gold_patches[0])
    with pytest.raises(PatchExecutionError):
        apply_patch_transaction(updated, inst.gold_patches[0])


def test_rebind_updates_evidence_ids():
    inst = _instance()
    updated, _ = apply_patch_transaction(inst.answer_v0, inst.gold_patches[0])
    citation = [c for c in updated.claims if c.claim_id == "c_citation" and c.status == "active"][0]
    assert any(eid.endswith("_v1") for eid in citation.evidence_ids)


def test_precondition_failure_rolls_back_by_exception():
    inst = _instance()
    patch = inst.gold_patches[0].model_copy(deep=True)
    patch.preconditions[0].expected_version = 99
    with pytest.raises(PatchExecutionError):
        apply_patch_transaction(inst.answer_v0, patch)
    assert inst.answer_v0.version == 0


def test_delete_operation_marks_deleted():
    inst = _instance()
    patch = SemanticPatch(
        patch_id="delete",
        answer_id=inst.answer_v0.answer_id,
        from_version=0,
        to_version=1,
        preconditions=[PatchPrecondition(claim_id="c_split", expected_version=0)],
        operations=[PatchOperation(op="DELETE", claim_ids=["c_split"], reason="test delete")],
        preserve_claim_ids=[],
    )
    updated, _ = apply_patch_transaction(inst.answer_v0, patch)
    assert [c for c in updated.claims if c.claim_id == "c_split"][0].status == "deleted"


def test_insert_operation_adds_claim():
    inst = _instance()
    new_claim = ClaimNode(
        claim_id="c_new",
        answer_id=inst.answer_v0.answer_id,
        version=1,
        text="A new supported claim.",
        claim_type="factual",
        evidence_ids=[inst.answer_v0.evidence[0].evidence_id],
        valid_from=1,
    )
    patch = SemanticPatch(
        patch_id="insert",
        answer_id=inst.answer_v0.answer_id,
        from_version=0,
        to_version=1,
        preconditions=[],
        operations=[PatchOperation(op="INSERT", claim_ids=[], new_claims=[new_claim], reason="test insert")],
        preserve_claim_ids=[],
    )
    updated, _ = apply_patch_transaction(inst.answer_v0, patch)
    assert any(c.claim_id == "c_new" for c in updated.claims)


def test_split_operation_deletes_old_and_adds_new():
    inst = _instance()
    new_claim = inst.answer_v0.claims[4].model_copy(deep=True)
    new_claim.claim_id = "c_split_part"
    patch = SemanticPatch(
        patch_id="split",
        answer_id=inst.answer_v0.answer_id,
        from_version=0,
        to_version=1,
        preconditions=[PatchPrecondition(claim_id="c_split", expected_version=0)],
        operations=[PatchOperation(op="SPLIT", claim_ids=["c_split"], new_claims=[new_claim], reason="test split")],
        preserve_claim_ids=[],
    )
    updated, _ = apply_patch_transaction(inst.answer_v0, patch)
    assert any(c.claim_id == "c_split_part" for c in updated.claims)


def test_invalidate_operation_marks_invalidated():
    inst = _instance()
    patch = SemanticPatch(
        patch_id="invalidate",
        answer_id=inst.answer_v0.answer_id,
        from_version=0,
        to_version=1,
        preconditions=[PatchPrecondition(claim_id="c_split", expected_version=0)],
        operations=[PatchOperation(op="INVALIDATE", claim_ids=["c_split"], reason="test invalidate")],
        preserve_claim_ids=[],
    )
    updated, _ = apply_patch_transaction(inst.answer_v0, patch)
    assert [c for c in updated.claims if c.claim_id == "c_split"][0].status == "invalidated"

