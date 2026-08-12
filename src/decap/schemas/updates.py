from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from decap.schemas.evidence import EvidenceRecord


class EvidenceUpdate(BaseModel):
    """Evidence delta between two answer/evidence versions."""

    update_id: str
    entity_id: str
    from_version: int
    to_version: int
    added_evidence: List[EvidenceRecord] = Field(default_factory=list)
    removed_evidence_ids: List[str] = Field(default_factory=list)
    modified_evidence: List[EvidenceRecord] = Field(default_factory=list)
    change_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImpactLabel(BaseModel):
    """Gold or predicted impact state for a claim."""

    claim_id: str
    state: Literal["MUST_CHANGE", "STILL_VALID", "UNCERTAIN"]
    direct: bool
    reason: str
    gold_operation: Optional[str] = None

