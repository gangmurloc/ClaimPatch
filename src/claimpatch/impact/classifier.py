from typing import List

from claimpatch.impact.propagation import rule_based_impact
from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import EvidenceUpdate, ImpactLabel


class MockImpactClassifier:
    """P0-compatible stand-in for the later learned impact classifier."""

    name = "rule_based_mock"

    def predict(self, answer: AnswerVersion, update: EvidenceUpdate) -> List[ImpactLabel]:
        return rule_based_impact(answer, update)

