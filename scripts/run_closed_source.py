#!/usr/bin/env python3
"""
Tier C closed-source orchestrator. Calls OpenAI / Google / Anthropic SDKs
directly (no LF dependency), writes responses to standard JSONL format, then
hands off to harness GPT-4o evaluator.

User must supply --models with explicit model ids; the script raises rather
than picking defaults (per memory feedback_user_reminders).

Usage:
    python scripts/run_closed_source.py --models gpt-5.5 --benchmarks our_test
    python scripts/run_closed_source.py --models claude-opus-4.7 gemini-3.1-pro \\
        --benchmarks our_test mis_easy --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.data.benchmarks import (
    AdvBenchBenchmark, FigStepBenchmark, MISBenchmark, MMSafetyBenchmark,
    MSSBenchmark, MMStarBenchmark,
)
from harness.data.dataset import HarnessDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="Closed-source model ids (e.g., gpt-5.5, gemini-3.1-pro, claude-opus-4.7).")
    p.add_argument("--benchmarks", nargs="+", default=["our_test"],
                   help="Benchmark names to evaluate.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-root", default="results/main/closed_source")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--our-test-path",
                   default="/mnt/hdd/xuran/mis_dataset_builder/dataset/test.json")
    p.add_argument("--our-test-image-root",
                   default="/mnt/hdd/xuran/mis_dataset_builder/dataset")
    return p.parse_args()


def load_benchmark_records(name: str, args) -> list[dict]:
    """Map benchmark name to records via existing harness loaders."""
    if name == "our_test":
        ds = HarnessDataset(
            data_path=Path(args.our_test_path),
            image_root=Path(args.our_test_image_root),
            mode="eval",
            max_samples=args.limit,
        )
        return ds.to_eval_records()
    if name in ("mis_easy", "mis_hard", "mis_real"):
        return MISBenchmark(subset=name, max_samples=args.limit).load()
    raise ValueError(f"Closed-source orchestrator does not yet wire benchmark={name}; "
                     "add a loader call here when you need it.")


# ── SDK adapters (lazy-imported) ──────────────────────────────────────────


def call_openai(model_id: str, records: Iterable[dict]) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for r in records:
        content = [{"type": "text", "text": r["question"]}]
        for k in ("image_path1", "image_path2"):
            p = r.get(k)
            if p and Path(p).exists():
                with open(p, "rb") as fp:
                    import base64 as _b64
                    b64 = _b64.b64encode(fp.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=0.0,
        )
        out.append({**r, "response": resp.choices[0].message.content or ""})
    return out


def call_anthropic(model_id: str, records: Iterable[dict]) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = []
    for r in records:
        content = []
        for k in ("image_path1", "image_path2"):
            p = r.get(k)
            if p and Path(p).exists():
                import base64 as _b64
                with open(p, "rb") as fp:
                    b64 = _b64.b64encode(fp.read()).decode()
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
        content.append({"type": "text", "text": r["question"]})
        resp = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        out.append({**r, "response": text})
    return out


def call_gemini(model_id: str, records: Iterable[dict]) -> list[dict]:
    from google import genai  # google-genai SDK
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out = []
    for r in records:
        parts: list = [r["question"]]
        for k in ("image_path1", "image_path2"):
            p = r.get(k)
            if p and Path(p).exists():
                from PIL import Image
                parts.append(Image.open(p))
        resp = client.models.generate_content(model=model_id, contents=parts)
        out.append({**r, "response": getattr(resp, "text", "") or ""})
    return out


def dispatch(model_id: str, records: list[dict]) -> list[dict]:
    """Pick the right SDK by prefix."""
    mid = model_id.lower()
    if mid.startswith("gpt") or mid.startswith("o"):
        return call_openai(model_id, records)
    if mid.startswith("claude"):
        return call_anthropic(model_id, records)
    if mid.startswith("gemini"):
        return call_gemini(model_id, records)
    raise ValueError(f"Cannot map closed-source model id to SDK: {model_id}")


def main() -> int:
    args = parse_args()
    if not args.models:
        print("ERROR: --models is required (no defaults).", file=sys.stderr)
        return 2

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        for bench in args.benchmarks:
            print(f"[run] {model_id} on {bench}")
            records = load_benchmark_records(bench, args)
            print(f"  loaded {len(records)} records")
            if args.dry_run:
                print(f"  [dry-run] would call SDK for {model_id}")
                continue
            outputs = dispatch(model_id, records)
            out_path = output_root / model_id / f"{bench}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for o in outputs:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            print(f"  → {out_path}")
    print("[done] Run scripts/run_eval_only.py on output dirs to score with GPT-4o.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
