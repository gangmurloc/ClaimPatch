from typing import List

from decap.patch.generator import build_patch_from_impact
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import ImpactLabel


def direct_only_labels(labels: List[ImpactLabel]) -> List[ImpactLabel]:
    out: List[ImpactLabel] = []
    for label in labels:
        if label.direct and label.state == "MUST_CHANGE":
            out.append(label)
        else:
            out.append(
                ImpactLabel(
                    claim_id=label.claim_id,
                    state="STILL_VALID",
                    direct=label.direct,
                    reason="Direct-only baseline does not propagate implicit impact.",
                )
            )
    return out


def direct_only_patch(current: AnswerVersion, fresh: AnswerVersion, labels: List[ImpactLabel]):
    return build_patch_from_impact(current, fresh, direct_only_labels(labels))

