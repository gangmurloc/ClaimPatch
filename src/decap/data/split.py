from typing import Dict, List, Sequence, TypeVar

T = TypeVar("T")


def entity_level_split(items: Sequence[T], train: float = 0.7, validation: float = 0.15) -> Dict[str, List[T]]:
    """Deterministic split preserving chain/entity integrity by item order."""

    n = len(items)
    train_end = int(n * train)
    val_end = train_end + int(n * validation)
    return {
        "train": list(items[:train_end]),
        "validation": list(items[train_end:val_end]),
        "test": list(items[val_end:]),
    }

