from pathlib import Path

from decap.schemas.results import AnswerVersion


def write_dot(answer: AnswerVersion, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph claim_graph {"]
    for claim in answer.claims:
        lines.append(f'  "{claim.claim_id}" [label="{claim.claim_id}: {claim.claim_type}"];')
    for edge in answer.dependencies:
        for source in edge.source_claim_ids:
            lines.append(f'  "{source}" -> "{edge.target_claim_id}" [label="{edge.dependency_type}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

