from pathlib import Path

from decap.data.synthetic_generator import generate_synthetic_dataset
from decap.pipelines.run_experiment import run_p0


def test_generator_creates_requested_size():
    instances = generate_synthetic_dataset(7, seed=3, sequential_steps=2)
    assert len(instances) == 7
    assert len(instances[0].updates) == 2


def test_run_p0_writes_summary():
    summary = run_p0(Path("configs/experiments/p0_rule_based.yaml"))
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "failures: 0" in text


def test_p0_predictions_written():
    path = Path("outputs/p0/predictions.jsonl")
    assert path.exists()
    assert sum(1 for _ in path.open("r", encoding="utf-8")) == 1500
