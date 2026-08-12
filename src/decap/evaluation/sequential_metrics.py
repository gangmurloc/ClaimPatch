from typing import Dict, List


def average_metric(rows: List[Dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0

