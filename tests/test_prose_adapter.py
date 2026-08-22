import json

import numpy as np

from claimpatch.data.synthetic_generator import generate_benchmark_instance
from claimpatch.pipelines.prose_adapter import (
    ExtractedClaim,
    ExtractedClaimGraph,
    ExtractedDependency,
    ProseGenerator,
    ProseGraphExtractor,
    _aligned_downstream_probe,
    _graph_metrics,
    _match_nodes,
    _smoke_go_no_go,
)


class _FixedClient:
    model_name = "fixed-test-client"

    def generate_text(self, prompt, task="generic", payload=None):
        if task.startswith("prose_generation"):
            return json.dumps({"prose": "A natural answer paragraph."})
        return json.dumps(
            {
                "claims": [
                    {"local_id": "p1", "text": "Model-AA achieved 73% accuracy.", "claim_type": "numeric"},
                    {"local_id": "p2", "text": "Model-AB achieved 65% accuracy.", "claim_type": "numeric"},
                    {
                        "local_id": "p3",
                        "text": "Model-AA exceeded Model-AB by 8 percentage points.",
                        "claim_type": "numeric",
                    },
                ],
                "dependencies": [
                    {
                        "source_claim_ids": ["p1", "p2"],
                        "target_claim_id": "p3",
                        "dependency_type": "numeric",
                        "confidence": 1.0,
                        "rule": "difference",
                    }
                ],
            }
        )


def test_prose_generator_returns_json_prose():
    prose, raw, format_success = ProseGenerator(_FixedClient()).generate(
        "question",
        ["fact one", "fact two"],
        "explicit",
        "lead with result",
    )
    assert prose == "A natural answer paragraph."
    assert json.loads(raw)["prose"] == prose
    assert format_success is True


def test_prose_generator_keeps_plain_text_with_visible_format_failure():
    class _PlainClient:
        model_name = "plain"

        def generate_text(self, prompt, task="generic", payload=None):
            return "A direct paragraph without the requested JSON wrapper."

    prose, _, format_success = ProseGenerator(_PlainClient()).generate(
        "question",
        ["fact one"],
        "implicit",
        "compact",
    )
    assert prose.startswith("A direct paragraph")
    assert format_success is False


def test_prose_graph_extractor_validates_local_graph():
    graph, _ = ProseGraphExtractor(_FixedClient()).extract("question", "answer")
    assert len(graph.claims) == 3
    assert graph.dependencies[0].source_claim_ids == ["p1", "p2"]


def test_node_matching_is_one_to_one_and_thresholded():
    instance = generate_benchmark_instance(0, seed=13, sequential_steps=1)
    gold = instance.answer_v0.claims[:2]
    predicted = [
        ExtractedClaim(local_id="p1", text=gold[0].text, claim_type=gold[0].claim_type),
        ExtractedClaim(local_id="p2", text=gold[1].text, claim_type=gold[1].claim_type),
    ]
    embeddings = {
        gold[0].text: np.asarray([1.0, 0.0]),
        gold[1].text: np.asarray([0.0, 1.0]),
    }
    mapping, details = _match_nodes(predicted, gold, embeddings, threshold=0.70)
    assert mapping == {"p1": gold[0].claim_id, "p2": gold[1].claim_id}
    assert all(detail["accepted"] for detail in details)


def test_aligned_graph_probe_is_perfect_for_exact_graph():
    instance = generate_benchmark_instance(0, seed=13, sequential_steps=1)
    graph = ExtractedClaimGraph(
        claims=[
            ExtractedClaim(
                local_id=f"p{index}",
                text=claim.text,
                claim_type=claim.claim_type,
            )
            for index, claim in enumerate(instance.answer_v0.claims, 1)
        ],
        dependencies=[],
    )
    mapping = {
        claim.local_id: gold.claim_id
        for claim, gold in zip(graph.claims, instance.answer_v0.claims)
    }
    reverse = {gold_id: local_id for local_id, gold_id in mapping.items()}
    graph.dependencies = [
        ExtractedDependency(
            source_claim_ids=[reverse[source] for source in edge.source_claim_ids],
            target_claim_id=reverse[edge.target_claim_id],
            dependency_type=edge.dependency_type,
            confidence=edge.confidence,
            rule=edge.rule,
        )
        for edge in instance.answer_v0.dependencies
    ]
    edge_metrics, mapped_edges = _graph_metrics(graph, instance.answer_v0, mapping)
    downstream = _aligned_downstream_probe(instance, mapping, mapped_edges)
    assert edge_metrics["edge_typed_f1"] == 1.0
    assert downstream["dcs_loss"] == 0.0
    assert downstream["collateral_increase"] == 0.0
    assert downstream["residual_increase"] == 0.0


def test_smoke_gate_requires_every_off_diagonal_cell():
    base = {
        "n": 10,
        "parse_success": 1.0,
        "edge_typed_f1": 0.8,
        "multi_parent_recall": 0.8,
        "dcs_loss": 0.1,
        "off_diagonal": True,
    }
    cells = {
        "a->b|explicit": dict(base),
        "a->b|implicit": dict(base),
        "b->a|explicit": dict(base),
        "b->a|implicit": dict(base),
    }
    assert _smoke_go_no_go(cells)["pass"] is True
    cells["b->a|implicit"]["edge_typed_f1"] = 0.0
    assert _smoke_go_no_go(cells)["pass"] is False
