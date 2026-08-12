from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ClaimStatus = Literal["active", "invalidated", "replaced", "deleted", "uncertain"]
ClaimType = Literal[
    "factual",
    "numeric",
    "comparative",
    "interpretive",
    "temporal",
    "recommendation",
    "citation_only",
]


class ClaimNode(BaseModel):
    """Atomic claim in an answer version."""

    claim_id: str
    answer_id: str
    version: int
    text: str
    normalized_form: Optional[str] = None
    claim_type: ClaimType
    evidence_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus = "active"
    valid_from: int
    valid_until: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

