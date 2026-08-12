from typing import List

from decap.impact.propagation import rule_based_impact
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


class MockImpactClassifier:
    """P0-compatible stand-in for the later learned impact classifier."""

    name = "rule_based_mock"

    def predict(self, answer: AnswerVersion, update: EvidenceUpdate) -> List[ImpactLabel]:
        return rule_based_impact(answer, update)

