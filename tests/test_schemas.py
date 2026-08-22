import pytest

from claimpatch.schemas import EvidenceRecord
from claimpatch.schemas.graph import DependencyEdge
from claimpatch.schemas.patches import PatchPrecondition, SemanticPatch


def test_evidence_schema_valid():
    record = EvidenceRecord(evidence_id="e1", entity_id="x", version=0, text="A is 1.", metadata={})
    assert record.evidence_id == "e1"


def test_dependency_edge_rejects_empty_sources():
    with pytest.raises(ValueError):
        DependencyEdge(source_claim_ids=[], target_claim_id="c2", dependency_type="numeric")


def test_patch_precondition_default_status():
    pre = PatchPrecondition(claim_id="c1", expected_version=0)
    assert pre.expected_status == "active"


def test_semantic_patch_schema_minimal():
    patch = SemanticPatch(
        patch_id="p1",
        answer_id="a1",
        from_version=0,
        to_version=1,
        preconditions=[],
        operations=[],
        preserve_claim_ids=[],
    )
    assert patch.to_version == 1

