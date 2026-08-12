from typing import List

from decap.graph.builder import validate_answer_graph
from decap.schemas.patches import SemanticPatch
from decap.schemas.results import AnswerVersion


def validate_patch_against_answer(answer: AnswerVersion, patch: SemanticPatch) -> List[str]:
    errors: List[str] = []
    claim_ids = {claim.claim_id for claim in answer.claims}
    for precondition in patch.preconditions:
        if precondition.claim_id not in claim_ids:
            errors.append(f"missing precondition claim: {precondition.claim_id}")
    for op in patch.operations:
        if op.op != "INSERT":
            for claim_id in op.claim_ids:
                if claim_id not in claim_ids:
                    errors.append(f"missing operation claim: {claim_id}")
    errors.extend(validate_answer_graph(answer))
    return errors

