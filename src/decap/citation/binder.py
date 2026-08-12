from typing import List

from decap.schemas.claims import ClaimNode


def bind_evidence(claim: ClaimNode, evidence_ids: List[str]) -> ClaimNode:
    updated = claim.model_copy(deep=True)
    updated.evidence_ids = list(evidence_ids)
    return updated

