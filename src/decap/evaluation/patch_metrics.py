from typing import Dict, List, Set

from decap.schemas.patches import SemanticPatch
from decap.schemas.updates import ImpactLabel


def touched_claim_ids(patch: SemanticPatch) -> Set[str]:
    touched: Set[str] = set()
    for op in patch.operations:
        touched.update(op.claim_ids)
    return touched


def gold_impacted_claim_ids(labels: List[ImpactLabel]) -> Set[str]:
    return {label.claim_id for label in labels if label.state == "MUST_CHANGE"}


def patch_metrics(patch: SemanticPatch, gold_labels: List[ImpactLabel], all_claim_ids: Set[str]) -> Dict[str, float]:
    touched = touched_claim_ids(patch)
    impacted = gold_impacted_claim_ids(gold_labels)
    unaffected = all_claim_ids - impacted
    tp = len(touched & impacted)
    precision = tp / len(touched) if touched else (1.0 if not impacted else 0.0)
    recall = tp / len(impacted) if impacted else 1.0
    collateral = len(touched - impacted) / len(unaffected) if unaffected else 0.0
    residual = len(impacted - touched) / len(impacted) if impacted else 0.0
    broken_correct = len((touched - impacted) & unaffected) / len(unaffected) if unaffected else 0.0
    return {
        "patch_precision": precision,
        "patch_recall": recall,
        "dependency_complete_success": 1.0 if impacted.issubset(touched) else 0.0,
        "collateral_edit_rate": collateral,
        "residual_stale_rate": residual,
        "broken_correct_rate": broken_correct,
        "patch_footprint_claims": float(len(touched)),
        "operation_count": float(len(patch.operations)),
    }

