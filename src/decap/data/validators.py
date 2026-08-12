from typing import Iterable, List

from decap.graph.builder import validate_answer_graph
from decap.schemas.results import SyntheticInstance


def validate_instances(instances: Iterable[SyntheticInstance]) -> List[str]:
    errors: List[str] = []
    for instance in instances:
        graph_errors = validate_answer_graph(instance.answer_v0)
        errors.extend([f"{instance.instance_id}: {err}" for err in graph_errors])
        if len(instance.updates) != len(instance.gold_impact_labels):
            errors.append(f"{instance.instance_id}: update/impact length mismatch")
        if len(instance.updates) != len(instance.gold_patches):
            errors.append(f"{instance.instance_id}: update/patch length mismatch")
    return errors

