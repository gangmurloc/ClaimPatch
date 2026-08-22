from typing import Dict

from claimpatch.schemas.results import AnswerVersion


def citation_metrics(answer: AnswerVersion) -> Dict[str, float]:
    evidence_ids = {e.evidence_id for e in answer.evidence}
    citation_claims = [c for c in answer.claims if c.claim_type == "citation_only" and c.status == "active"]
    orphan = sum(1 for claim in citation_claims for eid in claim.evidence_ids if eid not in evidence_ids)
    total = sum(len(claim.evidence_ids) for claim in citation_claims)
    return {
        "citation_orphan_rate": orphan / total if total else 0.0,
        "citation_claim_count": float(len(citation_claims)),
    }

