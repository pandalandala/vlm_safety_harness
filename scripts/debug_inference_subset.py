#!/usr/bin/env python3
"""Run patched inference recovery logic on a small subset of eval sample ids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.loader import ConfigLoader
from harness.data.dataset import HarnessDataset
from harness.gpu.allocator import GPUAllocator
from harness.inference.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Experiment config path")
    p.add_argument("--model-path", required=True, help="Checkpoint or model path")
    p.add_argument("--ids", nargs="+", type=int, required=True, help="Eval sample ids to run")
    p.add_argument("--output", required=True, help="Where to write the subset JSONL")
    p.add_argument(
        "--infer-backend",
        choices=["auto", "vllm", "huggingface"],
        default="auto",
        help="Override the LF chat backend for debugging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader.load(args.config)
    if args.infer_backend != "auto":
        cfg.inference.backend = "hf" if args.infer_backend == "huggingface" else "vllm"

    records = HarnessDataset.from_config(cfg.dataset, mode="eval").to_eval_records()
    wanted = set(args.ids)
    subset = [record for record in records if record.get("id") in wanted]
    found_ids = {record.get("id") for record in subset}
    missing = sorted(wanted - found_ids)
    if missing:
        raise ValueError(f"Missing eval ids in dataset: {missing}")

    allocator = GPUAllocator()
    infer_plan = allocator.plan_inference(cfg.model.size_b)
    out_path = Path(args.output)

    engine = InferenceEngine(
        cfg=cfg,
        model_path=args.model_path,
        infer_plan=infer_plan,
        output_dir=out_path.parent,
        batch_size=len(subset),
    )
    engine._init_backend()
    try:
        outputs = engine._generate_with_recovery(subset)
    finally:
        if engine._backend:
            engine._backend.unload()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(outputs)} rows to {out_path}")
    if engine._backend is not None:
        print(f"[backend] {engine._backend.infer_backend}")
    for row in outputs:
        status = "ERR" if "[INFERENCE_ERROR]" in row.get("response", "") else "OK"
        preview = row.get("response", "").replace("\n", " ")[:160]
        print(f"{row.get('id')}: {status} {preview}")


if __name__ == "__main__":
    main()
