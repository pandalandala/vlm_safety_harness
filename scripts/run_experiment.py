#!/usr/bin/env python3
"""
Main experiment runner: train → infer → evaluate → report.

Usage:
    python scripts/run_experiment.py configs/experiments/main/main_dreams_internvl.yaml
    python scripts/run_experiment.py prelim/A1_textual_shortcut.yaml --dry-run
    python scripts/run_experiment.py main/main_dreams_internvl.yaml --skip-train --model-path /path/to/ckpt
    python scripts/run_experiment.py main/main_dreams_internvl.yaml --override training.learning_rate=5e-6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.loader import ConfigLoader
from harness.config.registry import ExperimentRegistry
from harness.config.schema import ExperimentConfig
from harness.data.converters import save_llamafactory_dataset
from harness.data.dataset import HarnessDataset
from harness.evaluation.gpt4o_evaluator import GPT4oEvaluator
from harness.evaluation.metrics import MetricsDict
from harness.gpu.allocator import GPUAllocator
from harness.inference.engine import InferenceEngine
from harness.training.trainer import HarnessTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to experiment YAML config")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dot-notation overrides, e.g. training.learning_rate=5e-6")
    p.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint")
    p.add_argument("--skip-inference", action="store_true", help="Skip inference, use existing responses")
    p.add_argument("--skip-eval", action="store_true", help="Skip GPT-4o evaluation")
    p.add_argument("--model-path", default=None, help="Explicit checkpoint path (implies --skip-train)")
    p.add_argument("--limit", type=int, default=None, help="Limit inference samples per benchmark")
    p.add_argument("--benchmarks", nargs="*", default=None, help="Override which benchmarks to run")
    p.add_argument("--force", action="store_true", help="Re-run even if config hash already completed")
    p.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    p.add_argument("--resume", action="store_true", default=True, help="Resume inference/eval from partial output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg: ExperimentConfig = ConfigLoader.load(args.config, overrides=args.override)
    registry = ExperimentRegistry()

    # Check if already completed
    if not args.force:
        existing = registry.is_completed(cfg)
        if existing:
            print(f"[skip] Already completed: {existing}")
            print("Use --force to re-run.")
            return

    run_dir = registry.make_run_dir(cfg)
    print(f"[run] {cfg.name} → {run_dir}")

    # GPU allocation
    allocator = GPUAllocator()
    print(allocator.status_report())

    # ── Training ──────────────────────────────────────────────────────────
    skip_train = args.skip_train or args.model_path or not cfg.training.enabled
    model_path = args.model_path or cfg.model.hf_path

    if not skip_train:
        train_plan = allocator.plan_training(cfg.model.size_b)
        print(f"[train] GPUs: {train_plan.cuda_visible_devices()} | effective_batch={train_plan.effective_batch}")

        if args.dry_run:
            print(f"[dry-run] Would train with: {train_plan}")
        else:
            ds = HarnessDataset.from_config(cfg.dataset, mode="train")
            data_file = run_dir / "train_data.json"
            save_llamafactory_dataset(
                [ds[i] for i in range(len(ds))],
                data_file,
                use_cot=cfg.training.use_cot_labels,
                cot_format=cfg.training.cot_format,
            )
            trainer = HarnessTrainer()
            model_path = str(trainer.prepare_and_run(cfg, data_file, train_plan, run_dir / "checkpoint"))
            print(f"[train] Done → {model_path}")

    with open(run_dir / "gpu_plan.json", "w") as f:
        if not skip_train:
            train_plan_data = {"train_gpus": train_plan.num_gpus, "train_gpu_ids": train_plan.gpu_ids}
        else:
            train_plan_data = {"train_gpus": 0, "train_gpu_ids": []}
        json.dump(train_plan_data, f)

    # ── Inference ─────────────────────────────────────────────────────────
    responses_dir = run_dir / "responses"
    if not args.skip_inference:
        infer_plan = allocator.plan_inference(cfg.model.size_b)
        print(f"[infer] GPUs: {infer_plan.gpu_ids} | tp={infer_plan.tensor_parallel_size}")

        if args.dry_run:
            print(f"[dry-run] Would infer with: {infer_plan} on {args.benchmarks or cfg.inference.benchmarks}")
        else:
            engine = InferenceEngine(
                cfg=cfg,
                model_path=model_path,
                infer_plan=infer_plan,
                output_dir=run_dir,
                batch_size=cfg.inference.batch_size,
            )
            engine.run_all_benchmarks(
                benchmarks=args.benchmarks,
                resume=args.resume,
                max_samples=args.limit,
            )
            print(f"[infer] Done → {responses_dir}")

    # ── Evaluation ────────────────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}

    if not args.skip_eval and not args.dry_run:
        evaluator = GPT4oEvaluator(
            model=cfg.evaluation.model,
            api_key_env=cfg.evaluation.api_key_env,
            max_concurrent=cfg.evaluation.max_concurrent_requests,
            compute_per_category=cfg.evaluation.compute_per_category,
        )

        eval_dir = run_dir / "eval_results"
        for jsonl_path in sorted(responses_dir.glob("*.jsonl")):
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}.jsonl"
            print(f"[eval] {benchmark}")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  → {metrics.format_table_row()}")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_out = {"benchmarks": {b: m.get("overall", m) for b, m in all_metrics.items()}}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n[done] Run saved to: {run_dir}")
    print(f"[done] metrics.json written")


if __name__ == "__main__":
    main()
