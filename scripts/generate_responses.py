#!/usr/bin/env python3
"""
Generate CoT + safety responses for DREAMS training data using Qwen3.5-122B-A10B.

Reads /mnt/hdd/xuran/mis_dataset_builder/dataset/train.json (sharegpt format with
empty `gpt` turns), runs the VLM annotator with the RESPONSE_GENERATION_SYSTEM
prompt, fills `conversations[1]["value"]` with the generated response, and
writes the result to train_annotated.json with per-batch checkpointing.

Usage:
    python scripts/generate_responses.py \
        --dataset /mnt/hdd/xuran/mis_dataset_builder/dataset \
        --model Qwen/Qwen3.5-122B-A10B \
        --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json \
        --batch-size 4

Resume is on by default; pass --no-resume to overwrite from scratch.

Qwen3.5-122B-A10B memory on 8×RTX A6000 (48 GB each = 384 GB total):
  Weights 122B×2B ≈ 244 GB | Remaining ≈ 140 GB for KV cache + activations.
  Only 12/48 layers use full attention (rest: Gated DeltaNet, no KV cache) →
  KV cache footprint is small. gpu_memory_utilization=0.88 is safe.

Prompt strategy (RESPONSE_GENERATION_SYSTEM):
  Three-Source harm detection framework (A=single-image, B=text-level, C=combined)
  identical to the rubric used in the MIS paper's GPT-4o evaluation. Output is
  two-part: <safety_analysis> CoT block + 3-5 sentence natural safety response.
  Thinking mode is disabled (enable_thinking=False in apply_chat_template) so the
  model skips <think>...</think> tokens and outputs the annotation directly.

Logs: logs/data_prep/YYYYMMDD_HHMMSS_generate_responses.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.gpu.allocator import GPUAllocator
from harness.training.cot_generator import CoTGenerator
from harness.utils.logger import get_logger


DATASET_ROOT = Path("/mnt/hdd/xuran/mis_dataset_builder/dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DATASET_ROOT),
                   help="Dataset root directory (contains train.json + images_train/)")
    p.add_argument("--input", default="train.json",
                   help="Input filename under --dataset (sharegpt format)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: <dataset>/train_annotated.json)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai"],
                   help="Inference backend to use")
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B",
                   help="HF repo id or local model path")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1500,
                   help="Max output tokens per sample. Full CoT + 3-5 sentence response "
                        "typically needs 800-1200 tokens; default 1500 adds safety margin.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max_model_len (input+output context window)")
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
    log = get_logger("generate_responses", category="data_prep")

    dataset_root = Path(args.dataset)
    input_path = dataset_root / args.input
    output_path = Path(args.output) if args.output else dataset_root / "train_annotated.json"

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(f"[gen] Input not found: {input_path}")

    # GPU plan
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        allocator = GPUAllocator()
        report = allocator.status_report()
        log.info("GPU status:\n%s", report)
        print(report)
        available = allocator.get_available()
        gpu_ids = [g.index for g in available]
    if not gpu_ids:
        log.error("No available GPUs detected.")
        sys.exit("[gen] No available GPUs detected.")
    tp = len(gpu_ids)

    log.info("GPUs: %s  (tensor_parallel_size=%d)", gpu_ids, tp)
    log.info("Model:  %s", args.model)
    log.info("Input:  %s", input_path)
    log.info("Output: %s", output_path)
    log.info("max_tokens=%d  max_model_len=%d  batch_size=%d",
             args.max_tokens, args.max_model_len, args.batch_size)

    all_records = load_train_records(input_path, dataset_root)
    if args.limit:
        all_records = all_records[: args.limit]
    log.info("Loaded %d records", len(all_records))

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
        log.info("Resume: %d already done, %d remaining",
                 len(done_ids), len(all_records) - len(done_ids))
    else:
        done_ids = set()

    pending = [r for r in all_records if r["id"] not in done_ids]
    if not pending:
        log.info("All records already annotated. Nothing to do.")
        return

    generator = CoTGenerator(
        model_path=args.model,
        backend=args.backend,
        gpu_ids=gpu_ids,
        tensor_parallel_size=tp,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        mode="full_response",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
    )

    bs = args.batch_size
    total = len(pending)
    for i in range(0, total, bs):
        batch = pending[i : i + bs]
        batch_for_gen = [{k: v for k, v in r.items() if k != "_original"} for r in batch]
        results = generator.generate_batch(batch_for_gen)

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
        log.info("%d/%d processed → %s", n_done, total, output_path)

    log.info("Done. %d records written to %s", len(done_map), output_path)


if __name__ == "__main__":
    main()
