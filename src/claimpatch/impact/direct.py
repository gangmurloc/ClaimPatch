from typing import List, Set

from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import EvidenceUpdate


def directly_impacted_claims(answer: AnswerVersion, update: EvidenceUpdate) -> Set[str]:
    """Find claims bound to modified/removed evidence or changed attributes."""

    changed_evidence_ids = {e.evidence_id for e in update.modified_evidence}
    changed_evidence_ids.update(update.removed_evidence_ids)
    changed_attributes = {
        e.metadata.get("attribute")
        for e in update.modified_evidence
        if e.metadata.get("attribute") is not None
    }
    direct: Set[str] = set()
    for claim in answer.claims:
        if claim.status != "active":
            continue
        if changed_evidence_ids.intersection(claim.evidence_ids):
            direct.add(claim.claim_id)
        elif claim.metadata.get("slot") in changed_attributes:
            direct.add(claim.claim_id)
    return direct
