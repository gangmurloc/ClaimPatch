from pathlib import Path
from types import SimpleNamespace

from claimpatch.data.synthetic_generator import generate_synthetic_dataset
from claimpatch.pipelines.run_experiment import _environment, run_p0


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


def test_environment_records_model_and_source_provenance():
    client = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-model-commit")),
        tokenizer=SimpleNamespace(init_kwargs={}),
    )
    config = {
        "model": {
            "backend": "local_transformers",
            "model_name": "example/model",
            "revision": "requested-tag",
            "torch_dtype": "bf16",
            "device_map": "auto",
            "load_in_4bit": False,
            "do_sample": False,
            "max_new_tokens": 512,
        }
    }

    environment = _environment(config, shared_client=client)

    assert environment["git_commit"] != "unavailable"
    assert isinstance(environment["git_dirty"], bool)
    assert environment["model"]["identifier"] == "example/model"
    assert environment["model"]["requested_revision"] == "requested-tag"
    assert environment["model"]["resolved_revision"] == "resolved-model-commit"
    assert environment["model"]["torch_dtype"] == "bf16"
