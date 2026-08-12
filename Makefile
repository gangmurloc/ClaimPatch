.PHONY: test smoke p0 preflight

test:
	PYTHONPATH=src python3 -m pytest -q

smoke:
	PYTHONPATH=src python3 -m decap.cli run-p0 --config configs/experiments/p0_rule_based.yaml --limit 4

p0:
	PYTHONPATH=src python3 -m decap.cli run-p0 --config configs/experiments/p0_rule_based.yaml

preflight:
	python3 scripts/preflight_public_release.py
