from typing import List

from decap.patch.generator import build_patch_from_impact
from decap.schemas.results import AnswerVersion
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


def attribute_no_graph_labels(gold_labels: List[ImpactLabel], update: EvidenceUpdate) -> List[ImpactLabel]:
    """Strong graph-free baseline.

    It edits claims whose own attribute changed and handles citation/source
    refreshes, but it does not propagate through dependency edges.
    """

    changed_attributes = {
        evidence.metadata.get("attribute")
        for evidence in update.modified_evidence + update.added_evidence
        if evidence.metadata.get("attribute") is not None
    }
    out: List[ImpactLabel] = []
    for label in gold_labels:
        claim_attr = {
            "c_a_acc": "a_accuracy",
            "c_b_acc": "b_accuracy",
            "c_citation": "citation",
        }.get(label.claim_id)
        should_edit = False
        if label.claim_id == "c_citation" and update.change_type in {
            "metric_revision",
            "citation_refresh",
            "source_refresh",
        }:
            should_edit = True
        elif claim_attr in changed_attributes and label.gold_operation:
            should_edit = True
        if should_edit:
            out.append(
                ImpactLabel(
                    claim_id=label.claim_id,
                    state="MUST_CHANGE",
                    direct=label.direct,
                    reason="Attribute-no-graph baseline edits direct attribute/source claims only.",
                    gold_operation=label.gold_operation or ("REBIND" if label.claim_id == "c_citation" else "REPLACE"),
                )
            )
        else:
            out.append(
                ImpactLabel(
                    claim_id=label.claim_id,
                    state="STILL_VALID",
                    direct=label.direct,
                    reason="Attribute-no-graph baseline does not model downstream dependencies.",
                )
            )
    return out


def attribute_no_graph_patch(
    current: AnswerVersion,
    update: EvidenceUpdate,
    fresh: AnswerVersion,
    gold_labels: List[ImpactLabel],
):
    return build_patch_from_impact(current, fresh, attribute_no_graph_labels(gold_labels, update))

