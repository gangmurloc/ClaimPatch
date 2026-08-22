import random
from typing import Dict, List


def paired_bootstrap_delta(
    a_values: List[float],
    b_values: List[float],
    seed: int = 13,
    samples: int = 1000,
) -> Dict[str, float]:
    if len(a_values) != len(b_values):
        raise ValueError("paired bootstrap requires equal-length arrays")
    if not a_values:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    n = len(a_values)
    deltas = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(a_values[i] for i in idx) / n
        b = sum(b_values[i] for i in idx) / n
        deltas.append(a - b)
    deltas.sort()
    return {
        "mean_delta": (sum(a_values) / n) - (sum(b_values) / n),
        "ci_low": deltas[int(0.025 * samples)],
        "ci_high": deltas[int(0.975 * samples) - 1],
    }

