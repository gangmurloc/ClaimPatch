from typing import Dict

from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.results import AnswerVersion


def approximate_cost(answer: AnswerVersion, patch: SemanticPatch) -> Dict[str, float]:
    active_claims = [claim for claim in answer.claims if claim.status == "active"]
    answer_tokens = sum(len(claim.text.split()) for claim in active_claims)
    patch_tokens = sum(len(claim.text.split()) for op in patch.operations for claim in op.new_claims)
    patch_tokens += sum(max(1, len(op.new_evidence_ids)) for op in patch.operations if op.op == "REBIND")
    return {
        "approx_answer_tokens": float(answer_tokens),
        "approx_patch_tokens": float(patch_tokens),
        "token_reduction_vs_full": 1.0 - (patch_tokens / answer_tokens) if answer_tokens else 0.0,
    }
