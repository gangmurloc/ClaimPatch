from copy import deepcopy

from claimpatch.data.synthetic_generator import generate_benchmark_instance
from claimpatch.graph.builder import validate_answer_graph
from claimpatch.graph.traversal import closure, descendants
from claimpatch.schemas.graph import DependencyEdge


def _answer():
    return generate_benchmark_instance(0).answer_v0


def test_valid_synthetic_graph():
    assert validate_answer_graph(_answer()) == []


def test_descendants_include_numeric_and_comparison():
    out = descendants(_answer(), ["c_a_acc"])
    assert {"c_diff", "c_comparison", "c_citation"}.issubset(out)


def test_closure_includes_direct_claim():
    out = closure(_answer(), ["c_a_acc"])
    assert "c_a_acc" in out


def test_cycle_detection():
    answer = deepcopy(_answer())
    answer.dependencies.append(
        DependencyEdge(source_claim_ids=["c_comparison"], target_claim_id="c_a_acc", dependency_type="logical")
    )
    assert any("cycle" in err for err in validate_answer_graph(answer))


def test_missing_evidence_detection():
    answer = deepcopy(_answer())
    answer.claims[0].evidence_ids = ["missing"]
    assert any("missing evidence" in err for err in validate_answer_graph(answer))

