from typing import Dict, List, Set

from claimpatch.graph.traversal import closure
from claimpatch.impact.direct import directly_impacted_claims
from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import EvidenceUpdate, ImpactLabel


def _category(diff: int) -> str:
    if diff >= 8:
        return "substantially outperforms"
    if diff >= 3:
        return "outperforms"
    if diff >= 1:
        return "slightly outperforms"
    if diff == 0:
        return "matches"
    return "underperforms"


def _old_values(answer: AnswerVersion) -> Dict[str, object]:
    return {claim.metadata.get("slot"): claim.metadata.get("value") for claim in answer.claims if claim.status == "active"}


def _new_value(update: EvidenceUpdate, attribute: str, fallback: object) -> object:
    for evidence in update.modified_evidence + update.added_evidence:
        if evidence.metadata.get("attribute") == attribute:
            return evidence.metadata.get("value")
    return fallback


def rule_based_impact(answer: AnswerVersion, update: EvidenceUpdate) -> List[ImpactLabel]:
    """P0 deterministic impact closure and semantic revalidation."""

    direct = directly_impacted_claims(answer, update)
    candidates = closure(answer, direct)
    values = _old_values(answer)
    old_a = int(values.get("a_accuracy", 0))
    old_b = int(values.get("b_accuracy", 0))
    new_a = int(_new_value(update, "a_accuracy", old_a))
    new_b = int(_new_value(update, "b_accuracy", old_b))
    old_diff = old_a - old_b
    new_diff = new_a - new_b
    old_category = _category(old_diff)
    new_category = _category(new_diff)

    labels: List[ImpactLabel] = []
    for claim in answer.claims:
        if claim.status != "active":
            continue
        claim_id = claim.claim_id
        is_direct = claim_id in direct
        if claim_id not in candidates and claim.claim_type != "citation_only":
            labels.append(
                ImpactLabel(
                    claim_id=claim_id,
                    state="STILL_VALID",
                    direct=False,
                    reason="Outside dependency closure.",
                )
            )
            continue

        if claim.metadata.get("slot") == "a_accuracy":
            state = "MUST_CHANGE" if new_a != old_a else "STILL_VALID"
            op = "REPLACE" if state == "MUST_CHANGE" else None
            reason = "Directly bound metric evidence changed."
        elif claim.metadata.get("slot") == "b_accuracy":
            state = "MUST_CHANGE" if new_b != old_b else "STILL_VALID"
            op = "REPLACE" if state == "MUST_CHANGE" else None
            reason = "Model B metric evidence changed." if state == "MUST_CHANGE" else "Model B metric unchanged."
        elif claim.metadata.get("slot") == "difference":
            state = "MUST_CHANGE" if new_diff != old_diff else "STILL_VALID"
            op = "REPLACE" if state == "MUST_CHANGE" else None
            reason = "Numeric downstream difference re-evaluated."
        elif claim.metadata.get("slot") == "comparison":
            state = "MUST_CHANGE" if new_category != old_category else "STILL_VALID"
            op = "REPLACE" if state == "MUST_CHANGE" else None
            reason = "Comparative threshold re-evaluated."
        elif claim.claim_type == "citation_only":
            state = "MUST_CHANGE"
            op = "REBIND"
            reason = "Citation must bind to current metric evidence version."
        else:
            state = "STILL_VALID"
            op = None
            reason = "No changed evidence or semantic dependency requires editing."
        labels.append(
            ImpactLabel(claim_id=claim_id, state=state, direct=is_direct, reason=reason, gold_operation=op)
        )
    return labels
