from typing import Dict, List

from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.updates import ImpactLabel


def preservation_metrics(patch: SemanticPatch, gold_labels: List[ImpactLabel]) -> Dict[str, float]:
    gold_preserve = {label.claim_id for label in gold_labels if label.state == "STILL_VALID"}
    predicted_preserve = set(patch.preserve_claim_ids)
    tp = len(gold_preserve & predicted_preserve)
    precision = tp / len(predicted_preserve) if predicted_preserve else 0.0
    recall = tp / len(gold_preserve) if gold_preserve else 1.0
    unsupported = len(predicted_preserve - gold_preserve) / len(predicted_preserve) if predicted_preserve else 0.0
    return {
        "preserve_precision": precision,
        "preserve_recall": recall,
        "unsupported_preservation_rate": unsupported,
    }

