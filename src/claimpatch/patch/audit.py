from typing import Dict, List

from claimpatch.schemas.patches import PreservationCertificate, SemanticPatch
from claimpatch.schemas.results import AnswerVersion


def build_preservation_certificates(answer: AnswerVersion, patch: SemanticPatch) -> List[PreservationCertificate]:
    claims = {claim.claim_id: claim for claim in answer.claims}
    certs: List[PreservationCertificate] = []
    for claim_id in patch.preserve_claim_ids:
        claim = claims[claim_id]
        certs.append(
            PreservationCertificate(
                claim_id=claim_id,
                decision="PRESERVE",
                reason_code="UNAFFECTED_OR_STILL_VALID",
                supporting_evidence_ids=list(claim.evidence_ids),
                validation_status="NOT_CHECKED",
            )
        )
    return certs

