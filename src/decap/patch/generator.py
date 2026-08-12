from typing import Dict, List

from decap.schemas.claims import ClaimNode
from decap.schemas.patches import PatchOperation, PatchPrecondition, SemanticPatch
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


def _claims_by_id(answer: AnswerVersion) -> Dict[str, ClaimNode]:
    return {claim.claim_id: claim for claim in answer.claims}


def build_oracle_patch(
    current: AnswerVersion,
    update: EvidenceUpdate,
    fresh: AnswerVersion,
    labels: List[ImpactLabel],
    patch_id: str = None,
) -> SemanticPatch:
    """Build a deterministic gold patch from labels and fresh target claims."""

    patch_id = patch_id or f"patch_{current.answer_id}_v{update.from_version}_to_v{update.to_version}"
    current_claims = _claims_by_id(current)
    fresh_claims = _claims_by_id(fresh)
    operations: List[PatchOperation] = []
    preserve: List[str] = []
    preconditions: List[PatchPrecondition] = []
    for label in labels:
        claim = current_claims[label.claim_id]
        preconditions.append(PatchPrecondition(claim_id=claim.claim_id, expected_version=claim.version))
        if label.state != "MUST_CHANGE":
            preserve.append(claim.claim_id)
            continue
        op = label.gold_operation or "REPLACE"
        if op == "REBIND":
            fresh_claim = fresh_claims[label.claim_id]
            operations.append(
                PatchOperation(
                    op="REBIND",
                    claim_ids=[label.claim_id],
                    old_evidence_ids=list(claim.evidence_ids),
                    new_evidence_ids=list(fresh_claim.evidence_ids),
                    reason=label.reason,
                )
            )
        else:
            operations.append(
                PatchOperation(
                    op="REPLACE",
                    claim_ids=[label.claim_id],
                    new_claims=[fresh_claims[label.claim_id]],
                    reason=label.reason,
                )
            )
    current_evidence_ids = {e.evidence_id for e in current.evidence}
    new_evidence_records = [e.model_dump() for e in fresh.evidence if e.evidence_id not in current_evidence_ids]
    return SemanticPatch(
        patch_id=patch_id,
        answer_id=current.answer_id,
        from_version=update.from_version,
        to_version=update.to_version,
        preconditions=preconditions,
        operations=operations,
        preserve_claim_ids=preserve,
        postconditions=["schema_valid", "graph_valid", "no_stale_must_change_claims"],
        metadata={"generator": "oracle_rule_based", "new_evidence_records": new_evidence_records},
    )


def build_patch_from_impact(current: AnswerVersion, fresh: AnswerVersion, labels: List[ImpactLabel]) -> SemanticPatch:
    update_stub = EvidenceUpdate(
        update_id=f"predicted_update_{current.version}_to_{fresh.version}",
        entity_id=current.answer_id,
        from_version=current.version,
        to_version=fresh.version,
        change_type="predicted",
    )
    return build_oracle_patch(current, update_stub, fresh, labels, patch_id=f"decap_{current.answer_id}_{fresh.version}")
