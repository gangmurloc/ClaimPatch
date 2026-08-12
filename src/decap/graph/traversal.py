from typing import Iterable, Set

import networkx as nx

from decap.graph.builder import build_dependency_digraph
from decap.schemas.results import AnswerVersion


def descendants(answer: AnswerVersion, claim_ids: Iterable[str]) -> Set[str]:
    graph = build_dependency_digraph(answer)
    impacted = set(claim_ids)
    out = set()
    for claim_id in impacted:
        if claim_id in graph:
            out.update(nx.descendants(graph, claim_id))
    return out


def closure(answer: AnswerVersion, direct_claim_ids: Iterable[str]) -> Set[str]:
    direct = set(direct_claim_ids)
    return direct | descendants(answer, direct)

