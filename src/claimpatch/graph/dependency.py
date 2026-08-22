from typing import List

from claimpatch.schemas.graph import DependencyEdge


def dependencies_targeting(edges: List[DependencyEdge], claim_id: str) -> List[DependencyEdge]:
    return [edge for edge in edges if edge.target_claim_id == claim_id]

