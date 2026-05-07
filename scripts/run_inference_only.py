#!/usr/bin/env python3
"""
Run inference only (no training, no evaluation).

Usage:
    python scripts/run_inference_only.py --config prelim/A3_synthetic_real_gap.yaml
    python scripts/run_inference_only.py --config main/main_dreams_internvl.yaml --model-path /ckpt
    python scripts/run_inference_only.py --config main/main_dreams_internvl.yaml --limit 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.loader import ConfigLoader
from harness.config.registry import ExperimentRegistry
from harness.gpu.allocator import GPUAllocator
from harness.inference.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.add_argument("--model-path", default=None, help="Checkpoint path (default: config hf_path)")
    p.add_argument("--output-dir", default=None, help="Output dir (default: auto from registry)")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader.load(args.config, overrides=args.override)

    allocator = GPUAllocator()
    print(allocator.status_report())
    infer_plan = allocator.plan_inference(cfg.model.size_b)
    print(f"[infer] plan: {infer_plan.tensor_parallel_size}x TP, GPUs={infer_plan.gpu_ids}")

    model_path = args.model_path or cfg.model.hf_path

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        registry = ExperimentRegistry()
        out_dir = registry.make_run_dir(cfg)

    engine = InferenceEngine(
        cfg=cfg,
        model_path=model_path,
        infer_plan=infer_plan,
        output_dir=out_dir,
        batch_size=cfg.inference.batch_size,
    )
    result_paths = engine.run_all_benchmarks(
        benchmarks=args.benchmarks,
        resume=args.resume,
        max_samples=args.limit,
    )

    print(f"\n[done] Inference outputs:")
    for bname, path in result_paths.items():
        print(f"  {bname}: {path}")


if __name__ == "__main__":
    main()
