#!/usr/bin/env python3
"""
Build counterfactual (CF) pair index for E5 evaluation.

Usage (offline, run once):
    python scripts/build_cf_pairs.py \
        --test-json /mnt/hdd/xuran/mis_dataset_builder/dataset/test.json \
        --benign-pool /path/to/openimages_subset \
        --output /mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json \
        --cf-images-dir /mnt/hdd/xuran/mis_dataset_builder/dataset/cf_images \
        --swap-idx 2 --seed 0

The benign pool may come from any of:
    - OpenImages (recommended) — `pip` package already installed; download a
      subset using its `download_images` helper or any other method
    - ImageNet (already in HF cache for many users)
    - COCO val2017 (https://cocodataset.org)
    - Any directory of safe images

Optional GPT-4o-mini quality-judge step (`--quality-judge`) verifies the
synthesized (orig_question, kept_image, benign_image) triple is genuinely safe.
Rejected triples retry up to `--max-retries` times.

Outputs:
    test_cf.json — list of CF records, see harness/data/cf_synthesizer.py docstring
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.data.cf_synthesizer import CFSynthesizer, collect_benign_pool


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build CF pair index for E5.")
    p.add_argument("--test-json", required=True, type=Path,
                   help="DREAMS test.json (or filtered subset)")
    p.add_argument("--benign-pool", required=True, type=Path,
                   help="Directory containing benign images (any tree depth).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output path for test_cf.json")
    p.add_argument("--cf-images-dir", type=Path, default=None,
                   help="If set, copy chosen benign images here, renamed to "
                        "<cf_id>.<ext>. If unset, CF records reference benign "
                        "images at their original pool paths.")
    p.add_argument("--swap-idx", type=int, choices=[1, 2], default=2,
                   help="Which image (1 or 2) to replace with benign.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit input records (for smoke testing).")
    p.add_argument("--filter-harm-type", choices=["explicit", "implicit"], default=None,
                   help="Only process records with this harm_type.")
    p.add_argument("--quality-judge", choices=["gpt-4o-mini", "gpt-4o"], default=None,
                   help="If set, verify pair safety with LLM judge (not yet wired).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries when quality-judge rejects a pair.")
    p.add_argument("--skip-judge", action="store_true",
                   help="Explicitly skip quality judge (alias for omitting --quality-judge).")
    return p


def load_test_records(test_json: Path, filter_harm_type: str | None,
                      limit: int | None) -> list[dict]:
    with open(test_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    if filter_harm_type:
        data = [r for r in data if r.get("harm_type") == filter_harm_type]
    if limit:
        data = data[:limit]
    return data


def maybe_run_quality_judge(*_args, **_kwargs):
    """Placeholder for GPT-4o-mini judge integration.

    Hook signature kept so build_cf_pairs.py can accept --quality-judge today
    even though the eval call is not yet implemented. Wire up via
    harness.evaluation.gpt4o_evaluator when CF synth becomes the bottleneck.
    """
    return True  # accept all pairs for now


def main() -> int:
    args = build_argparser().parse_args()

    if args.quality_judge and args.skip_judge:
        print("ERROR: --quality-judge and --skip-judge are mutually exclusive", file=sys.stderr)
        return 2

    print(f"[load] test records from {args.test_json}")
    records = load_test_records(args.test_json, args.filter_harm_type, args.limit)
    print(f"  loaded: {len(records)} records")

    print(f"[load] benign pool from {args.benign_pool}")
    pool = collect_benign_pool(args.benign_pool)
    print(f"  pool size: {len(pool)} images")

    synth = CFSynthesizer(
        benign_pool=pool,
        swap_image_idx=args.swap_idx,
        seed=args.seed,
    )

    print(f"[synth] generating CF pairs (swap_idx={args.swap_idx})")
    cf_records = synth.synthesize(records, cf_image_dir=args.cf_images_dir)

    if args.quality_judge:
        print(f"[judge] {args.quality_judge} pair-safety verification "
              "(stub — accepts all; wire in real judge when needed)")
        accepted = []
        for cr in cf_records:
            if maybe_run_quality_judge(cr, judge_model=args.quality_judge):
                accepted.append(cr)
        cf_records = accepted
        print(f"  accepted: {len(cf_records)}")

    out_path = synth.write_jsonl(cf_records, args.output)
    print(f"[write] {out_path}  ({len(cf_records)} CF records)")

    pair_index = synth.to_pair_index(cf_records)
    pair_index_path = args.output.with_suffix(".pair_index.json")
    with open(pair_index_path, "w") as f:
        json.dump(pair_index, f, indent=2)
    print(f"[write] {pair_index_path}  ({len(pair_index)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
