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
    p.add_argument(
        "--evaluator-type",
        default="auto",
        choices=["auto", "gpt4o", "rule", "harmbench", "accuracy"],
        help=(
            "Evaluator backend. 'auto' reads benchmark loader's evaluator_type "
            "attribute and dispatches accordingly. Explicit values override per-run."
        ),
    )
    return p.parse_args()


def _resolve_evaluator_type(args, benchmark_name: str) -> str:
    """auto → look up benchmark loader's evaluator_type; else use args.evaluator_type."""
    if args.evaluator_type != "auto":
        return args.evaluator_type
    # Map benchmark name to loader class via engine registry, then read class attr.
    try:
        from harness.inference.engine import BENCHMARK_REGISTRY
        spec = BENCHMARK_REGISTRY.get(benchmark_name)
        if spec and not spec.startswith("_"):
            mod_path, cls_name = spec.rsplit(".", 1)
            import importlib
            cls = getattr(importlib.import_module(mod_path), cls_name)
            return getattr(cls, "evaluator_type", "gpt4o")
    except Exception:
        pass
    return "gpt4o"


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

    all_metrics: dict = {}

    if args.judge == "llama_guard":
        from harness.evaluation.llama_guard import LlamaGuardEvaluator
        evaluator = LlamaGuardEvaluator()
        for jsonl_path in jsonl_files:
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}_llamaguard.jsonl"
            print(f"[eval-lg] {benchmark}")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  {metrics.format_table_row()}")
        evaluator.unload()

    else:
        # GPT-4o, rule, accuracy, harmbench routing via BenchmarkEvaluator
        from harness.evaluation import get_evaluator
        for jsonl_path in jsonl_files:
            benchmark = jsonl_path.stem
            ev_type = _resolve_evaluator_type(args, benchmark)
            print(f"[eval] {benchmark} (evaluator={ev_type})")

            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f if line.strip()]

            if ev_type == "gpt4o":
                # Reuse async-incremental GPT4oEvaluator path for resume support.
                evaluator = GPT4oEvaluator(
                    model=args.judge if args.judge != "gpt-4o" else "gpt-4o",
                    api_key_env="OPENAI_API_KEY",
                    max_concurrent=20,
                    compute_per_category=True,
                )
                out_jsonl = eval_dir / f"{benchmark}.jsonl"
                _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            else:
                ev = get_evaluator(ev_type)
                annotated, metrics = ev.evaluate(records)
                out_jsonl = eval_dir / f"{benchmark}.jsonl"
                with open(out_jsonl, "w") as f:
                    for r in annotated:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            all_metrics[benchmark] = metrics.to_dict()
            print(f"  {metrics.format_table_row()}")

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
