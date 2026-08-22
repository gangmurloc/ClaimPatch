from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

from claimpatch.schemas.claims import ClaimNode
from claimpatch.schemas.graph import DependencyEdge


PatchOp = Literal["REPLACE", "DELETE", "INSERT", "SPLIT", "MERGE", "REBIND", "INVALIDATE"]


class PatchPrecondition(BaseModel):
    claim_id: str
    expected_version: int
    expected_status: str = "active"


class PatchOperation(BaseModel):
    op: PatchOp
    claim_ids: List[str]
    new_claims: List[ClaimNode] = Field(default_factory=list)
    old_evidence_ids: List[str] = Field(default_factory=list)
    new_evidence_ids: List[str] = Field(default_factory=list)
    reason: str
    dependency_updates: List[DependencyEdge] = Field(default_factory=list)

    @field_validator("claim_ids")
    @classmethod
    def non_empty_for_non_insert(cls, value: List[str]) -> List[str]:
        return value


class SemanticPatch(BaseModel):
    patch_id: str
    answer_id: str
    from_version: int
    to_version: int
    preconditions: List[PatchPrecondition]
    operations: List[PatchOperation]
    preserve_claim_ids: List[str]
    postconditions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreservationCertificate(BaseModel):
    claim_id: str
    decision: Literal["PRESERVE"]
    reason_code: str
    supporting_evidence_ids: List[str]
    validation_status: Literal["ENTAILED", "UNCERTAIN", "NOT_CHECKED"]

