from typing import Dict, Iterable, List

from claimpatch.schemas.updates import ImpactLabel


LABELS = ["MUST_CHANGE", "STILL_VALID", "UNCERTAIN"]


def _by_claim(labels: Iterable[ImpactLabel]) -> Dict[str, str]:
    return {label.claim_id: label.state for label in labels}


def impact_classification_metrics(predicted: List[ImpactLabel], gold: List[ImpactLabel]) -> Dict[str, float]:
    pred = _by_claim(predicted)
    ref = _by_claim(gold)
    metrics: Dict[str, float] = {}
    f1s = []
    for label in LABELS:
        tp = sum(1 for cid, state in ref.items() if state == label and pred.get(cid) == label)
        fp = sum(1 for cid, state in pred.items() if state == label and ref.get(cid) != label)
        fn = sum(1 for cid, state in ref.items() if state == label and pred.get(cid) != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"{label.lower()}_precision"] = precision
        metrics[f"{label.lower()}_recall"] = recall
        metrics[f"{label.lower()}_f1"] = f1
        f1s.append(f1)
    metrics["impact_macro_f1"] = sum(f1s) / len(f1s)
    return metrics


def dependency_complete_success(predicted: List[ImpactLabel], gold: List[ImpactLabel]) -> float:
    pred_must = {label.claim_id for label in predicted if label.state == "MUST_CHANGE"}
    gold_must = {label.claim_id for label in gold if label.state == "MUST_CHANGE"}
    return 1.0 if gold_must.issubset(pred_must) else 0.0

