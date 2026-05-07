#!/usr/bin/env python3
"""
Run GPT-4o evaluation on existing inference responses.

Usage:
    python scripts/run_eval_only.py --responses results/main/main_dreams_internvl/20240501_120000/responses/
    python scripts/run_eval_only.py --responses /path/to/responses --judge llama_guard
    python scripts/run_eval_only.py --responses /path/to/responses --resume
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.evaluation.gpt4o_evaluator import GPT4oEvaluator
from harness.evaluation.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True, help="Directory containing *.jsonl response files")
    p.add_argument("--output-dir", default=None, help="Output dir for eval results (default: responses/../eval_results)")
    p.add_argument("--judge", default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini", "llama_guard"])
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--benchmarks", nargs="*", default=None, help="Only evaluate specific benchmarks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    responses_dir = Path(args.responses)
    eval_dir = Path(args.output_dir) if args.output_dir else responses_dir.parent / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if args.benchmarks:
        jsonl_files = [f for f in jsonl_files if f.stem in args.benchmarks]

    if not jsonl_files:
        print(f"No JSONL files found in {responses_dir}")
        return

    if args.judge in ("gpt-4o", "gpt-4o-mini"):
        import os
        evaluator = GPT4oEvaluator(
            model=args.judge,
            api_key_env="OPENAI_API_KEY",
            max_concurrent=20,
            compute_per_category=True,
        )
        all_metrics = {}
        for jsonl_path in jsonl_files:
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}.jsonl"
            print(f"[eval] {benchmark} ({sum(1 for _ in open(jsonl_path))} samples)")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  {metrics.format_table_row()}")

    elif args.judge == "llama_guard":
        from harness.evaluation.llama_guard import LlamaGuardEvaluator
        evaluator = LlamaGuardEvaluator()
        all_metrics = {}
        for jsonl_path in jsonl_files:
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}_llamaguard.jsonl"
            print(f"[eval-lg] {benchmark}")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  {metrics.format_table_row()}")
        evaluator.unload()

    # Save combined metrics
    metrics_path = eval_dir.parent / "metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing_metrics = json.load(f)
    existing_metrics.setdefault("benchmarks", {}).update(
        {b: m.get("overall", m) for b, m in all_metrics.items()}
    )
    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)

    print(f"\n[done] Eval results: {eval_dir}")
    print(f"[done] metrics.json: {metrics_path}")


if __name__ == "__main__":
    main()
