from typing import List

from claimpatch.schemas.results import AnswerVersion


def citation_errors(answer: AnswerVersion) -> List[str]:
    evidence_ids = {e.evidence_id for e in answer.evidence}
    errors = []
    for claim in answer.claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(f"{claim.claim_id} references missing evidence {evidence_id}")
    return errors

