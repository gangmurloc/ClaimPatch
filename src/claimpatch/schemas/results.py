from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from claimpatch.schemas.claims import ClaimNode
from claimpatch.schemas.evidence import EvidenceRecord
from claimpatch.schemas.graph import DependencyEdge
from claimpatch.schemas.patches import SemanticPatch
from claimpatch.schemas.updates import EvidenceUpdate, ImpactLabel


class AnswerVersion(BaseModel):
    """Rendered answer plus structured graph state."""

    answer_id: str
    version: int
    question: str
    rendered_text: str
    claims: List[ClaimNode]
    dependencies: List[DependencyEdge]
    evidence: List[EvidenceRecord]
    parent_version: Optional[int] = None
    applied_patch_id: Optional[str] = None


class SyntheticInstance(BaseModel):
    instance_id: str
    question: str
    evidence_v0: List[EvidenceRecord]
    answer_v0: AnswerVersion
    updates: List[EvidenceUpdate]
    gold_impact_labels: List[List[ImpactLabel]]
    gold_patches: List[SemanticPatch]
    fresh_answers: List[AnswerVersion]
    metadata: Dict[str, Any] = Field(default_factory=dict)

