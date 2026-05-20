#!/usr/bin/env python3
"""
Main experiment runner: train → infer → evaluate → report.

Usage:
    python scripts/run_experiment.py configs/experiments/main/main_dreams_internvl.yaml
    python scripts/run_experiment.py prelim/A1_textual_shortcut.yaml --dry-run
    python scripts/run_experiment.py main/main_dreams_internvl.yaml --skip-train --model-path /path/to/ckpt
    python scripts/run_experiment.py main/main_dreams_internvl.yaml --override training.learning_rate=5e-6
    python scripts/run_experiment.py main/main_dreams_internvl.yaml --resume-latest-train
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.loader import ConfigLoader
from harness.config.registry import ExperimentRegistry
from harness.config.schema import ExperimentConfig
from harness.data.converters import save_llamafactory_dataset
from harness.data.dataset import HarnessDataset
from harness.evaluation import get_evaluator
from harness.gpu.allocator import GPUAllocator
from harness.inference.engine import BENCHMARK_REGISTRY, InferenceEngine
from harness.training.trainer import HarnessTrainer
from harness.utils.logger import init_session


def _resolve_evaluator_type(benchmark_name: str) -> str:
    spec = BENCHMARK_REGISTRY.get(benchmark_name)
    if spec and not spec.startswith("_"):
        import importlib

        mod_path, cls_name = spec.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_path), cls_name)
        return getattr(cls, "evaluator_type", "gpt4o")
    return "gpt4o"


def _build_evaluator(cfg: ExperimentConfig, benchmark_name: str, output_jsonl: Path):
    ev_type = _resolve_evaluator_type(benchmark_name)
    kwargs = {}
    if ev_type == "gpt4o":
        kwargs = {
            "model": cfg.evaluation.model,
            "api_key_env": cfg.evaluation.api_key_env,
            "max_concurrent": cfg.evaluation.max_concurrent_requests,
            "compute_per_category": cfg.evaluation.compute_per_category,
            "output_jsonl": str(output_jsonl),
        }
    return ev_type, get_evaluator(ev_type, **kwargs)


def _merge_metrics(metrics_by_benchmark: dict[str, dict]) -> dict:
    merged: dict = {"benchmarks": {}}
    if len(metrics_by_benchmark) == 1:
        only_metrics = next(iter(metrics_by_benchmark.values()))
        merged.update({k: v for k, v in only_metrics.items() if k != "overall"})
        merged["overall"] = only_metrics.get("overall", {})
    for benchmark, metrics in metrics_by_benchmark.items():
        merged["benchmarks"][benchmark] = metrics.get("overall", metrics)
    return merged


def _has_model_weights(path: Path) -> bool:
    """Return True if a checkpoint directory appears to contain model weights."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_file():
            continue
        if child.name == "model.safetensors" or child.name == "pytorch_model.bin":
            return True
        if child.suffix in {".safetensors", ".bin"}:
            return True
    return False


def _find_latest_training_checkpoint(results_root: Path, group: str, name: str) -> Path | None:
    """Find the most recent checkpoint for a given experiment, if any exists."""
    experiment_root = results_root / group / name
    if not experiment_root.exists():
        return None

    for run_dir in sorted((p for p in experiment_root.iterdir() if p.is_dir()), reverse=True):
        ckpt_roots = []
        direct_ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
        nested_ckpts = []
        checkpoint_dir = run_dir / "checkpoint"
        if checkpoint_dir.is_dir():
            nested_ckpts = [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()]
        ckpt_roots.extend(direct_ckpts)
        ckpt_roots.extend(nested_ckpts)

        ckpts = sorted(ckpt_roots, key=lambda p: p.stat().st_mtime, reverse=True)
        for ckpt in ckpts:
            if _has_model_weights(ckpt):
                return ckpt
        final_ckpt = run_dir / "checkpoint"
        if _has_model_weights(final_ckpt):
            return final_ckpt

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to experiment YAML config")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dot-notation overrides, e.g. training.learning_rate=5e-6")
    p.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint")
    p.add_argument("--skip-inference", action="store_true", help="Skip inference, use existing responses")
    p.add_argument("--skip-eval", action="store_true", help="Skip GPT-4o evaluation")
    p.add_argument("--model-path", default=None, help="Explicit checkpoint path (implies --skip-train)")
    p.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES for this run, e.g. 0 or 0,1,2,3",
    )
    p.add_argument(
        "--resume-latest-train",
        action="store_true",
        help="Resume training from the latest checkpoint for this experiment if one exists",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit inference samples per benchmark")
    p.add_argument("--benchmarks", nargs="*", default=None, help="Override which benchmarks to run")
    p.add_argument("--experiment-id", default="", metavar="EID",
                   help="Experiment label inserted into the result path, e.g. E1, E2, A1, E4")
    p.add_argument("--force", action="store_true", help="Re-run even if config hash already completed")
    p.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    p.add_argument("--resume", action="store_true", default=True, help="Resume inference/eval from partial output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_session("run_experiment", tag=args.config, category="main")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()
        print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    cfg: ExperimentConfig = ConfigLoader.load(args.config, overrides=args.override)
    registry = ExperimentRegistry()
    exp_id = args.experiment_id  # e.g. "E1", "E2", "A1", ""

    # Check if already completed
    if not args.force:
        existing = registry.is_completed(cfg, experiment_id=exp_id)
        if existing:
            print(f"[skip] Already completed: {existing}")
            print("Use --force to re-run.")
            return

    run_dir = registry.make_run_dir(cfg, experiment_id=exp_id)
    print(f"[run] {cfg.name} → {run_dir}")

    # GPU allocation
    allocator = GPUAllocator()
    print(allocator.status_report())

    # ── Training ──────────────────────────────────────────────────────────
    skip_train = args.skip_train or args.model_path or not cfg.training.enabled
    model_path = args.model_path or cfg.model.hf_path

    if not skip_train:
        if args.resume_latest_train and cfg.training.resume_from_checkpoint is None:
            latest_ckpt = _find_latest_training_checkpoint(registry.results_root, cfg.group, cfg.name)
            if latest_ckpt is not None:
                cfg.training.resume_from_checkpoint = latest_ckpt
                print(f"[train] Resuming from latest checkpoint: {latest_ckpt}")
            else:
                print("[train] No previous checkpoint found; starting from scratch.")

        train_plan = allocator.plan_training(cfg.model.size_b, cfg.model.architecture)
        print(f"[train] GPUs: {train_plan.cuda_visible_devices()} | effective_batch={train_plan.effective_batch}")

        if args.dry_run:
            print(f"[dry-run] Would train with: {train_plan}")
            if cfg.training.general_data is not None:
                print(f"[dry-run] general_data={cfg.training.general_data.model_dump()}")
        else:
            ds = HarnessDataset.from_config(cfg.dataset, mode="train")
            data_file = run_dir / "train_data.json"
            save_llamafactory_dataset(
                ds.to_train_records(),
                data_file,
                image_root=cfg.dataset.image_root,
                use_cot=cfg.training.use_cot_labels,
                cot_format=cfg.training.cot_format,
                general_data_cfg=cfg.training.general_data,
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
        eval_dir = run_dir / "eval_results"
        for jsonl_path in sorted(responses_dir.glob("*.jsonl")):
            benchmark = jsonl_path.stem
            out_jsonl = eval_dir / f"{benchmark}.jsonl"
            ev_type, evaluator = _build_evaluator(cfg, benchmark, out_jsonl)
            print(f"[eval] {benchmark} (evaluator={ev_type})")
            _, metrics = evaluator.evaluate_file(jsonl_path, out_jsonl, resume=args.resume)
            all_metrics[benchmark] = metrics.to_dict()
            print(f"  → {metrics.format_table_row()}")

    # ── Save metrics ──────────────────────────────────────────────────────
    # Only write metrics.json when eval actually ran.  Leaving it absent when
    # --skip-eval is used prevents the registry from marking this run as
    # "completed", so a subsequent --skip-train run on another machine can
    # still proceed to inference + evaluation normally.
    if all_metrics:
        metrics_out = _merge_metrics(all_metrics)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"\n[done] Run saved to: {run_dir}")
        print(f"[done] metrics.json written")
    else:
        print(f"\n[done] Run saved to: {run_dir}")
        print("[done] Eval skipped — metrics.json NOT written; registry will not mark this run as completed.")


if __name__ == "__main__":
    main()
