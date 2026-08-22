from typing import List

from claimpatch.patch.generator import build_patch_from_impact
from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import ImpactLabel


def oracle_impact_patch(current: AnswerVersion, fresh: AnswerVersion, labels: List[ImpactLabel]) -> SemanticPatch:
    return build_patch_from_impact(current, fresh, labels)

