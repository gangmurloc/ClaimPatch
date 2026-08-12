import argparse
from pathlib import Path

from decap.data.loaders import write_jsonl
from decap.data.synthetic_generator import generate_synthetic_dataset, iter_jsonl
from decap.pipelines.prose_adapter import (
    build_prose_audit,
    run_prose_evaluation,
    run_prose_extraction,
    run_prose_generation,
)
from decap.pipelines.run_experiment import run_p0, run_p1_prompted


def main() -> None:
    parser = argparse.ArgumentParser(prog="decap")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic")
    gen.add_argument("--config", default="configs/synthetic.yaml")
    gen.add_argument("--size", type=int, default=100)
    gen.add_argument("--seed", type=int, default=13)
    gen.add_argument("--output", default="data/synthetic/benchmark_results_100.jsonl")

    p0 = sub.add_parser("run-p0")
    p0.add_argument("--config", default="configs/experiments/p0_rule_based.yaml")
    p0.add_argument("--limit", type=int, default=None)

    p1 = sub.add_parser("run-p1")
    p1.add_argument("--config", default="configs/experiments/p1_prompted.yaml")
    p1.add_argument("--limit", type=int, default=None)

    prose_generate = sub.add_parser("prose-adapter-generate")
    prose_generate.add_argument("--config", required=True)
    prose_generate.add_argument("--model", required=True)
    prose_generate.add_argument("--force", action="store_true")

    prose_audit = sub.add_parser("prose-adapter-audit")
    prose_audit.add_argument("--config", required=True)

    prose_extract = sub.add_parser("prose-adapter-extract")
    prose_extract.add_argument("--config", required=True)
    prose_extract.add_argument("--model", required=True)
    prose_extract.add_argument("--force", action="store_true")

    prose_evaluate = sub.add_parser("prose-adapter-evaluate")
    prose_evaluate.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "generate-synthetic":
        instances = generate_synthetic_dataset(size=args.size, seed=args.seed)
        write_jsonl(Path(args.output), iter_jsonl(instances))
        print(f"Wrote {len(instances)} instances to {args.output}")
    elif args.command == "run-p0":
        summary = run_p0(Path(args.config), limit=args.limit)
        print(f"P0 complete. Summary: {summary}")
    elif args.command == "run-p1":
        summary = run_p1_prompted(Path(args.config), limit=args.limit)
        print(f"P1 prompted run complete. Summary: {summary}")
    elif args.command == "prose-adapter-generate":
        path = run_prose_generation(Path(args.config), args.model, force=args.force)
        print(f"Prose generation artifact: {path}")
    elif args.command == "prose-adapter-audit":
        path = build_prose_audit(Path(args.config))
        print(f"Prose audit artifact: {path}")
    elif args.command == "prose-adapter-extract":
        path = run_prose_extraction(Path(args.config), args.model, force=args.force)
        print(f"Prose graph extraction artifact: {path}")
    elif args.command == "prose-adapter-evaluate":
        path = run_prose_evaluation(Path(args.config))
        print(f"Prose adapter summary: {path}")
if __name__ == "__main__":
    main()
