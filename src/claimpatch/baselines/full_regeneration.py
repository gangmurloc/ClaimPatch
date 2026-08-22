from copy import deepcopy
from typing import List

from claimpatch.schemas.patches import PatchOperation, PatchPrecondition, SemanticPatch
from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import ImpactLabel


def full_regeneration_patch(current: AnswerVersion, fresh: AnswerVersion, labels: List[ImpactLabel]) -> SemanticPatch:
    """Mock full regeneration as replacing every claim with its fresh counterpart."""

    fresh_by_id = {fresh_claim.claim_id: fresh_claim for fresh_claim in fresh.claims}
    current_evidence_ids = {e.evidence_id for e in current.evidence}
    active_claims = [claim for claim in current.claims if claim.status == "active"]
    operations = [
        PatchOperation(
            op="REPLACE",
            claim_ids=[claim.claim_id],
            new_claims=[deepcopy(fresh_by_id[claim.claim_id])],
            reason="Full regeneration rewrites all claims.",
        )
        for claim in active_claims
    ]
    return SemanticPatch(
        patch_id=f"full_regen_{current.answer_id}_{fresh.version}",
        answer_id=current.answer_id,
        from_version=current.version,
        to_version=fresh.version,
        preconditions=[PatchPrecondition(claim_id=c.claim_id, expected_version=c.version) for c in active_claims],
        operations=operations,
        preserve_claim_ids=[],
        metadata={
            "baseline": "full_regeneration",
            "new_evidence_records": [
                e.model_dump() for e in fresh.evidence if e.evidence_id not in current_evidence_ids
            ],
        },
    )
