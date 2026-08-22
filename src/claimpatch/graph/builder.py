from typing import Dict, List, Set

import networkx as nx

from claimpatch.schemas.results import AnswerVersion


def claim_ids(answer: AnswerVersion) -> Set[str]:
    return {claim.claim_id for claim in answer.claims}


def build_dependency_digraph(answer: AnswerVersion) -> nx.DiGraph:
    graph = nx.DiGraph()
    for claim in answer.claims:
        graph.add_node(claim.claim_id)
    for edge in answer.dependencies:
        for source in edge.source_claim_ids:
            graph.add_edge(source, edge.target_claim_id, dependency_type=edge.dependency_type)
    return graph


def validate_answer_graph(answer: AnswerVersion) -> List[str]:
    errors: List[str] = []
    ids = claim_ids(answer)
    for edge in answer.dependencies:
        if edge.target_claim_id not in ids:
            errors.append(f"dependency target missing: {edge.target_claim_id}")
        for source in edge.source_claim_ids:
            if source not in ids:
                errors.append(f"dependency source missing: {source}")
            if source == edge.target_claim_id:
                errors.append(f"self-loop dependency: {source}")
    graph = build_dependency_digraph(answer)
    if not nx.is_directed_acyclic_graph(graph):
        errors.append("dependency graph contains a cycle")
    evidence_ids = {e.evidence_id for e in answer.evidence}
    for claim in answer.claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(f"claim {claim.claim_id} references missing evidence {evidence_id}")
    return errors


def claims_by_id(answer: AnswerVersion) -> Dict[str, object]:
    return {claim.claim_id: claim for claim in answer.claims}

