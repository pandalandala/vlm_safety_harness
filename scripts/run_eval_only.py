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

from harness.evaluation import get_evaluator
from harness.evaluation.gpt4o_evaluator import GPT4oEvaluator
from harness.inference.engine import BENCHMARK_REGISTRY


def _merge_metrics(existing_metrics: dict, all_metrics: dict[str, dict]) -> dict:
    merged = existing_metrics or {}
    if len(all_metrics) == 1:
        only_metrics = next(iter(all_metrics.values()))
        merged.update({k: v for k, v in only_metrics.items() if k != "overall"})
        merged["overall"] = only_metrics.get("overall", {})
    merged.setdefault("benchmarks", {}).update(
        {b: m.get("overall", m) for b, m in all_metrics.items()}
    )
    return merged


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
    try:
        spec = BENCHMARK_REGISTRY.get(benchmark_name)
        if spec and not spec.startswith("_"):
            mod_path, cls_name = spec.rsplit(".", 1)
            import importlib
            cls = getattr(importlib.import_module(mod_path), cls_name)
            return getattr(cls, "evaluator_type", "gpt4o")
    except Exception:
        pass
    return "gpt4o"


def _build_evaluator(args, benchmark_name: str, output_jsonl: Path):
    ev_type = _resolve_evaluator_type(args, benchmark_name)
    kwargs = {}
    if ev_type == "gpt4o":
        kwargs = {
            "model": args.judge if args.judge != "gpt-4o" else "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "max_concurrent": 20,
            "compute_per_category": True,
            "output_jsonl": str(output_jsonl),
        }
    return ev_type, get_evaluator(ev_type, **kwargs)


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
        for jsonl_path in jsonl_files:
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}.jsonl"
            ev_type, evaluator = _build_evaluator(args, benchmark, out_jsonl)
            print(f"[eval] {benchmark} (evaluator={ev_type})")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  {metrics.format_table_row()}")

    # Save combined metrics
    metrics_path = eval_dir.parent / "metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing_metrics = json.load(f)
    merged_metrics = _merge_metrics(existing_metrics, all_metrics)
    with open(metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2)

    print(f"\n[done] Eval results: {eval_dir}")
    print(f"[done] metrics.json: {metrics_path}")


if __name__ == "__main__":
    main()
