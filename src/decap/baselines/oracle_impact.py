from typing import List

from decap.patch.generator import build_patch_from_impact
from decap.schemas.patches import SemanticPatch
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import ImpactLabel


def oracle_impact_patch(current: AnswerVersion, fresh: AnswerVersion, labels: List[ImpactLabel]) -> SemanticPatch:
    return build_patch_from_impact(current, fresh, labels)

