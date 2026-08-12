from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


DependencyType = Literal[
    "numeric",
    "logical",
    "comparative",
    "temporal",
    "causal",
    "citation",
    "scope",
    "other",
]


class DependencyEdge(BaseModel):
    """Hyperedge from one or more source claims to a target claim."""

    source_claim_ids: List[str]
    target_claim_id: str
    dependency_type: DependencyType
    confidence: float = 1.0
    rule: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_claim_ids")
    @classmethod
    def non_empty_sources(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("DependencyEdge.source_claim_ids must not be empty")
        return value

