from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from claimpatch.graph.builder import validate_answer_graph
from claimpatch.schemas.claims import ClaimNode
from claimpatch.schemas.evidence import EvidenceRecord
from claimpatch.schemas.patches import PatchOperation, SemanticPatch
from claimpatch.schemas.results import AnswerVersion


class PatchExecutionError(RuntimeError):
    pass


def _claims_by_id(answer: AnswerVersion) -> Dict[str, ClaimNode]:
    return {claim.claim_id: claim for claim in answer.claims}


def _replace_claim(answer: AnswerVersion, op: PatchOperation, to_version: int) -> None:
    claims = _claims_by_id(answer)
    if len(op.claim_ids) != 1 or len(op.new_claims) != 1:
        raise PatchExecutionError("REPLACE requires exactly one old and one new claim")
    old = claims[op.claim_ids[0]]
    old.status = "replaced"
    old.valid_until = to_version
    new_claim = deepcopy(op.new_claims[0])
    new_claim.version = to_version
    new_claim.valid_from = to_version
    new_claim.status = "active"
    answer.claims.append(new_claim)


def _delete_claim(answer: AnswerVersion, op: PatchOperation, to_version: int) -> None:
    claims = _claims_by_id(answer)
    for claim_id in op.claim_ids:
        claims[claim_id].status = "deleted"
        claims[claim_id].valid_until = to_version
    answer.dependencies = [
        edge
        for edge in answer.dependencies
        if edge.target_claim_id not in op.claim_ids and not set(edge.source_claim_ids).intersection(op.claim_ids)
    ]


def _insert_claim(answer: AnswerVersion, op: PatchOperation, to_version: int) -> None:
    for new_claim in op.new_claims:
        claim = deepcopy(new_claim)
        claim.version = to_version
        claim.valid_from = to_version
        claim.status = "active"
        answer.claims.append(claim)


def _rebind_claim(answer: AnswerVersion, op: PatchOperation) -> None:
    claims = _claims_by_id(answer)
    for claim_id in op.claim_ids:
        claims[claim_id].evidence_ids = list(op.new_evidence_ids)
        claims[claim_id].metadata["rebinding_reason"] = op.reason


def _invalidate_claim(answer: AnswerVersion, op: PatchOperation, to_version: int) -> None:
    claims = _claims_by_id(answer)
    for claim_id in op.claim_ids:
        claims[claim_id].status = "invalidated"
        claims[claim_id].valid_until = to_version


def _apply_operation(answer: AnswerVersion, op: PatchOperation, to_version: int) -> None:
    if op.op == "REPLACE":
        _replace_claim(answer, op, to_version)
    elif op.op == "DELETE":
        _delete_claim(answer, op, to_version)
    elif op.op == "INSERT":
        _insert_claim(answer, op, to_version)
    elif op.op == "REBIND":
        _rebind_claim(answer, op)
    elif op.op == "INVALIDATE":
        _invalidate_claim(answer, op, to_version)
    elif op.op == "SPLIT":
        _delete_claim(answer, op, to_version)
        _insert_claim(answer, op, to_version)
    elif op.op == "MERGE":
        _delete_claim(answer, op, to_version)
        _insert_claim(answer, op, to_version)
    else:
        raise PatchExecutionError(f"unsupported patch operation: {op.op}")
    answer.dependencies.extend(op.dependency_updates)


def apply_patch_transaction(
    answer: AnswerVersion,
    patch: SemanticPatch,
    available_evidence: Optional[List[EvidenceRecord]] = None,
) -> Tuple[AnswerVersion, List[str]]:
    """Apply patch to a copy and commit only if validation passes."""

    working = deepcopy(answer)
    log: List[str] = []
    claims = _claims_by_id(working)
    if working.version != patch.from_version:
        raise PatchExecutionError(
            f"patch version mismatch: answer v{working.version}, patch expects v{patch.from_version}"
        )
    for precondition in patch.preconditions:
        claim = claims.get(precondition.claim_id)
        if claim is None:
            raise PatchExecutionError(f"precondition references missing claim {precondition.claim_id}")
        if claim.version != precondition.expected_version:
            raise PatchExecutionError(
                f"claim {claim.claim_id} version {claim.version} != expected {precondition.expected_version}"
            )
        if claim.status != precondition.expected_status:
            raise PatchExecutionError(
                f"claim {claim.claim_id} status {claim.status} != expected {precondition.expected_status}"
            )
    for op in patch.operations:
        missing = [claim_id for claim_id in op.claim_ids if claim_id not in claims]
        if missing and op.op != "INSERT":
            raise PatchExecutionError(f"operation references missing claims: {missing}")
        _apply_operation(working, op, patch.to_version)
        claims = _claims_by_id(working)
        log.append(f"applied {op.op} to {','.join(op.claim_ids) or '[insert]'}")
    existing_evidence_ids = {e.evidence_id for e in working.evidence}
    for evidence in available_evidence or []:
        if evidence.evidence_id not in existing_evidence_ids:
            working.evidence.append(deepcopy(evidence))
            existing_evidence_ids.add(evidence.evidence_id)
            log.append(f"added available evidence {evidence.evidence_id}")
    for raw_evidence in patch.metadata.get("new_evidence_records", []):
        evidence = EvidenceRecord.model_validate(raw_evidence)
        if evidence.evidence_id not in existing_evidence_ids:
            working.evidence.append(evidence)
            existing_evidence_ids.add(evidence.evidence_id)
            log.append(f"added evidence {evidence.evidence_id}")
    working.version = patch.to_version
    working.parent_version = patch.from_version
    working.applied_patch_id = patch.patch_id
    active_latest: Dict[str, ClaimNode] = {}
    for claim in working.claims:
        if claim.status == "active":
            active_latest[claim.claim_id] = claim
    working.claims = [claim for claim in working.claims if claim.status != "active"] + list(active_latest.values())
    errors = validate_answer_graph(working)
    if errors:
        raise PatchExecutionError("graph validation failed after patch: " + "; ".join(errors))
    return working, log
