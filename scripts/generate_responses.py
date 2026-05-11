#!/usr/bin/env python3
"""
Generate CoT + safety responses for DREAMS training data using Qwen3.5-122B-A10B.

Reads /mnt/hdd/xuran/mis_dataset_builder/dataset/train.json (sharegpt format with
empty `gpt` turns), runs the VLM annotator with the RESPONSE_GENERATION_SYSTEM
prompt, fills `conversations[1]["value"]` with the generated response, and
writes the result to train_annotated.json with per-batch checkpointing.

Usage:
    python scripts/generate_responses.py \\
        --dataset /mnt/hdd/xuran/mis_dataset_builder/dataset \\
        --model Qwen/Qwen3.5-122B-A10B \\
        --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json \\
        --batch-size 4

Resume is on by default; pass --no-resume to overwrite from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.gpu.allocator import GPUAllocator
from harness.training.cot_generator import CoTGenerator


DATASET_ROOT = Path("/mnt/hdd/xuran/mis_dataset_builder/dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DATASET_ROOT),
                   help="Dataset root directory (contains train.json + images_train/)")
    p.add_argument("--input", default="train.json",
                   help="Input filename under --dataset (sharegpt format)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: <dataset>/train_annotated.json)")
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B",
                   help="HF repo id or local model path")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N records (debugging)")
    p.add_argument("--gpu-ids", default=None,
                   help="Comma-separated GPU ids; default uses GPUAllocator")
    return p.parse_args()


def load_train_records(input_path: Path, dataset_root: Path) -> list[dict]:
    """
    Load sharegpt-format train.json and flatten to flat records the
    CoTGenerator can consume. Each record carries a `_original` reference back
    to its original sharegpt entry for output reconstruction.
    """
    with open(input_path) as f:
        raw = json.load(f)

    records: list[dict] = []
    for r in raw:
        human_val = r["conversations"][0]["value"]
        question = human_val.replace("<image>\n", "").strip()

        img_paths = r.get("image", [])
        record = {
            "id": r["id"],
            "category": r.get("category", ""),
            "question": question,
            "image_path1": str(dataset_root / img_paths[0]) if len(img_paths) > 0 else "",
            "image_path2": str(dataset_root / img_paths[1]) if len(img_paths) > 1 else "",
            "_original": r,
        }
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    input_path = dataset_root / args.input
    output_path = Path(args.output) if args.output else dataset_root / "train_annotated.json"

    if not input_path.exists():
        sys.exit(f"[gen] Input not found: {input_path}")

    # GPU plan
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        allocator = GPUAllocator()
        print(allocator.status_report())
        available = allocator.get_available()
        gpu_ids = [g.index for g in available]
    if not gpu_ids:
        sys.exit("[gen] No available GPUs detected.")
    tp = len(gpu_ids)
    print(f"[gen] GPUs: {gpu_ids} (tp={tp})")
    print(f"[gen] Model: {args.model}")
    print(f"[gen] Input:  {input_path}")
    print(f"[gen] Output: {output_path}")

    all_records = load_train_records(input_path, dataset_root)
    if args.limit:
        all_records = all_records[: args.limit]
    print(f"[gen] Loaded {len(all_records)} records")

    # Resume: build done_map keyed by id from existing output
    done_map: dict[int, dict] = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        done_map = {r["id"]: r for r in existing}
        done_ids = {
            r["id"]
            for r in existing
            if len(r.get("conversations", [])) > 1
            and r["conversations"][1].get("value", "").strip()
        }
        print(f"[gen] Resume: {len(done_ids)} already done, {len(all_records) - len(done_ids)} remaining")
    else:
        done_ids = set()

    pending = [r for r in all_records if r["id"] not in done_ids]
    if not pending:
        print("[gen] All records already annotated.")
        return

    generator = CoTGenerator(
        model_path=args.model,
        backend="vllm",
        gpu_ids=gpu_ids,
        tensor_parallel_size=tp,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        mode="full_response",
    )

    bs = args.batch_size
    total = len(pending)
    for i in range(0, total, bs):
        batch = pending[i : i + bs]
        # Strip _original before passing to generator (keeps memory tidy + avoids
        # surprising the model if it ever leaks into prompt construction).
        batch_for_gen = [{k: v for k, v in r.items() if k != "_original"} for r in batch]
        results = generator.generate_batch(batch_for_gen)

        # Map back into sharegpt format
        result_by_id = {r["id"]: r for r in results}
        for r in batch:
            gen = result_by_id.get(r["id"], {})
            response_text = gen.get("gpt_response", "")
            orig = json.loads(json.dumps(r["_original"]))  # deep copy
            if len(orig.get("conversations", [])) >= 2:
                orig["conversations"][1]["value"] = response_text
            done_map[orig["id"]] = orig

        # Checkpoint after every batch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(list(done_map.values()), f, ensure_ascii=False, indent=2)

        n_done = i + len(batch)
        print(f"[gen] {n_done}/{total} processed → {output_path}")

    print(f"[done] {len(done_map)} records written to {output_path}")


if __name__ == "__main__":
    main()
