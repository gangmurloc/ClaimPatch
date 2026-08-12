import json
from pathlib import Path
from typing import Iterable, List

from decap.schemas.results import SyntheticInstance


def write_jsonl(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.rstrip("\n") + "\n")


def load_synthetic_jsonl(path: Path) -> List[SyntheticInstance]:
    instances: List[SyntheticInstance] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(SyntheticInstance.model_validate(json.loads(line)))
    return instances

