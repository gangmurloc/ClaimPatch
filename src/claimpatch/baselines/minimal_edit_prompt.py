from typing import Any, Dict, List, Optional, Tuple

from claimpatch.baselines.attribute_no_graph import attribute_no_graph_labels
from claimpatch.models.llm_client import MockLLMClient
from claimpatch.models.structured_generation import (
    extract_json_object,
    load_prompt,
    render_prompt,
)
from claimpatch.patch.generator import build_patch_from_impact
from claimpatch.pipelines.prompted_modules import (
    PromptedGenerationError,
    _repair_impact_operation_consistency,
    _validate_complete_impact_labels,
)
from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.results import AnswerVersion
from claimpatch.schemas.updates import EvidenceUpdate, ImpactLabel


_OPERATION_AS_STATE = {"REPLACE", "DELETE", "INSERT", "SPLIT", "MERGE", "REBIND", "INVALIDATE"}


def unstructured_selective_edit_labels(
    current: AnswerVersion,
    update: EvidenceUpdate,
    gold_labels: List[ImpactLabel],
    client: Optional[object] = None,
) -> List[ImpactLabel]:
    """Graph-free minimal-edit baseline.

    With a real LLM client, this is a one-shot selective-edit prompt that sees
    the current answer and evidence update but not ClaimPatch's dependency graph.
    With the deterministic mock client, it returns a graph-free direct/source
    heuristic so tests remain reproducible.
    """

    heuristic = attribute_no_graph_labels(gold_labels, update)
    if client is None or isinstance(client, MockLLMClient):
        return [
            ImpactLabel(
                claim_id=label.claim_id,
                state=label.state,
                direct=label.direct,
                reason="Unstructured selective edit mock: graph-free direct/source edit.",
                gold_operation=label.gold_operation,
            )
            for label in heuristic
        ]

    template = load_prompt("unstructured_selective_edit", "v1")
    prompt = render_prompt(
        template,
        current_answer=current.model_dump(),
        update=update.model_dump(),
    )
    payload = {"impact_labels": [label.model_dump() for label in heuristic]}
    raw = client.generate_text(prompt, task="unstructured_selective_edit", payload=payload)
    try:
        parsed = extract_json_object(raw)
        predicted = _normalize_unstructured_impact_labels(current, parsed["impact_labels"])
    except Exception as exc:
        raise PromptedGenerationError("unstructured_selective_edit", str(exc), raw_text=raw, prompt=prompt) from exc
    return [
        ImpactLabel(
            claim_id=label.claim_id,
            state=label.state,
            direct=label.direct,
            reason=label.reason,
            gold_operation=label.gold_operation,
        )
        for label in predicted
    ]


def unstructured_selective_edit_patch(
    current: AnswerVersion,
    update: EvidenceUpdate,
    fresh: AnswerVersion,
    gold_labels: List[ImpactLabel],
    client: Optional[object] = None,
) -> Tuple[List[ImpactLabel], SemanticPatch]:
    labels = unstructured_selective_edit_labels(current, update, gold_labels, client=client)
    patch = build_patch_from_impact(current, fresh, labels)
    patch.metadata["baseline"] = "unstructured_selective_edit"
    patch.metadata["baseline_backend"] = getattr(client, "model_name", "graph_free_mock")
    patch.metadata["uses_dependency_graph"] = False
    return labels, patch


def _normalize_unstructured_impact_labels(answer: AnswerVersion, raw_labels: Any) -> List[ImpactLabel]:
    """Map loose unstructured baseline output into the executable label schema.

    This is intentionally baseline-specific. Unstructured edit prompts often
    output only changed claims and sometimes put an operation name in `state`.
    For a fair selective-edit baseline, omissions are interpreted as preservation
    decisions rather than structured-generation failure.
    """

    if not isinstance(raw_labels, list):
        raise ValueError("impact_labels must be a list")
    active_ids = [claim.claim_id for claim in answer.claims if claim.status == "active"]
    active_set = set(active_ids)
    by_id: Dict[str, ImpactLabel] = {}
    for raw in raw_labels:
        if not isinstance(raw, dict):
            raise ValueError("impact label entries must be objects")
        item = dict(raw)
        claim_id = item.get("claim_id")
        if claim_id not in active_set:
            raise ValueError(f"impact label references non-active claim: {claim_id}")
        state = item.get("state")
        if state in _OPERATION_AS_STATE:
            item["state"] = "MUST_CHANGE"
            item["gold_operation"] = state
        item.setdefault("direct", False)
        item.setdefault("reason", "Unstructured selective edit output.")
        if item.get("state") == "MUST_CHANGE" and item.get("gold_operation") is None:
            item["gold_operation"] = "REPLACE"
        by_id[str(claim_id)] = ImpactLabel.model_validate(item)
    for claim_id in active_ids:
        if claim_id not in by_id:
            by_id[claim_id] = ImpactLabel(
                claim_id=claim_id,
                state="STILL_VALID",
                direct=False,
                reason="Unstructured selective edit omitted this claim; interpreted as preserve.",
                gold_operation=None,
            )
    predicted = [by_id[claim_id] for claim_id in active_ids]
    predicted = _repair_impact_operation_consistency(answer, predicted)
    _validate_complete_impact_labels(answer, predicted)
    return predicted


__all__ = ["unstructured_selective_edit_labels", "unstructured_selective_edit_patch"]
