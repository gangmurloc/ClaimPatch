from typing import Dict, List

from claimpatch.schemas.updates import ImpactLabel


def labels_by_claim(labels: List[ImpactLabel]) -> Dict[str, ImpactLabel]:
    return {label.claim_id: label for label in labels}


def impacted_claim_ids(labels: List[ImpactLabel]) -> List[str]:
    return [label.claim_id for label in labels if label.state == "MUST_CHANGE"]

