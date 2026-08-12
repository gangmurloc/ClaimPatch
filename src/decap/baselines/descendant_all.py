from typing import List, Set

from decap.graph.traversal import closure
from decap.impact.direct import directly_impacted_claims
from decap.patch.generator import build_patch_from_impact
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


def descendant_all_labels(answer: AnswerVersion, update: EvidenceUpdate, gold_labels: List[ImpactLabel]) -> List[ImpactLabel]:
    direct = directly_impacted_claims(answer, update)
    must = closure(answer, direct)
    out: List[ImpactLabel] = []
    for label in gold_labels:
        if label.claim_id in must:
            op = "REBIND" if label.claim_id == "c_citation" else "REPLACE"
            out.append(
                ImpactLabel(
                    claim_id=label.claim_id,
                    state="MUST_CHANGE",
                    direct=label.claim_id in direct,
                    reason="Descendant-all baseline edits direct impacts and every descendant.",
                    gold_operation=op,
                )
            )
        else:
            out.append(
                ImpactLabel(
                    claim_id=label.claim_id,
                    state="STILL_VALID",
                    direct=False,
                    reason="Outside descendant closure.",
                )
            )
    return out


def descendant_all_patch(
    current: AnswerVersion,
    update: EvidenceUpdate,
    fresh: AnswerVersion,
    gold_labels: List[ImpactLabel],
):
    return build_patch_from_impact(current, fresh, descendant_all_labels(current, update, gold_labels))

