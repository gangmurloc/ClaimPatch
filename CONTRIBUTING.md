# Contributing

Issues and focused pull requests are welcome. Please keep changes aligned with
the repository's research scope and avoid committing model weights, generated
answer corpora, local paths, credentials, or unpublished benchmark records.

Before opening a pull request, run:

```bash
pytest -q
decap run-p0 --config configs/experiments/p0_rule_based.yaml --limit 4
python scripts/preflight_public_release.py
```

Changes to metrics, prompts, repair rules, or benchmark generation should state
whether they invalidate comparison with the included aggregate results.

