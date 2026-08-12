from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """Versioned evidence item used to support claims."""

    evidence_id: str
    entity_id: str
    version: int
    text: str
    source_uri: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

