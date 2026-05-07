#!/usr/bin/env python3
"""
Generate structured CoT safety labels for DREAMS training data.
Uses available large VLM (Qwen2-VL-72B or InternVL2.5-78B) via vLLM.

Usage:
    python scripts/generate_cot_labels.py --input data_links/our_dataset/scored.json
    python scripts/generate_cot_labels.py --input scored.json --backend openai --model gpt-4o
    python scripts/generate_cot_labels.py --input scored.json --resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.gpu.allocator import GPUAllocator
from harness.training.cot_generator import CoTGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSON (scored.json or train.json)")
    p.add_argument("--output", default=None, help="Output JSON (default: input_cot.json)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai"])
    p.add_argument("--model", default=None,
                   help="Model path/name. vLLM default: Qwen/Qwen2-VL-72B-Instruct")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None, help="Only process first N records")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / (input_path.stem + "_cot.json")

    # Pick model
    if args.model:
        model_path = args.model
        tp = 1
        gpu_ids = None
    elif args.backend == "vllm":
        # Use largest available GPU config
        allocator = GPUAllocator()
        print(allocator.status_report())
        available = allocator.get_available()
        n_gpus = len(available)
        gpu_ids = [g.index for g in available]

        # Prefer 72B if enough GPUs
        if n_gpus >= 4:
            model_path = "Qwen/Qwen2-VL-72B-Instruct"
            tp = min(n_gpus, 4)
        elif n_gpus >= 2:
            model_path = "OpenGVLab/InternVL2_5-26B"
            tp = 2
        else:
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            tp = 1

        print(f"[cot] Using model: {model_path} (tp={tp}, gpus={gpu_ids[:tp]})")
    else:
        model_path = "gpt-4o"
        tp = 1
        gpu_ids = None

    generator = CoTGenerator(
        model_path=model_path,
        backend=args.backend,
        gpu_ids=gpu_ids[:tp] if gpu_ids else None,
        tensor_parallel_size=tp,
        max_tokens=512,
    )

    print(f"[cot] Input: {input_path}")
    print(f"[cot] Output: {output_path}")

    output_path = generator.generate_file(
        input_json=input_path,
        output_json=output_path,
        resume=args.resume,
    )

    print(f"[done] CoT labels written to: {output_path}")


if __name__ == "__main__":
    main()
