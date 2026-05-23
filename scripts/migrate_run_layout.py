#!/usr/bin/env python3
"""
One-time migration: old timestamp-leaf run dirs -> new model_tag-leaf layout.

Old: results/{group}/{eid?}/{cfg}/{YYYYMMDD_HHMMSS}/
New: results/{group}/{eid?}/{cfg}/{model_tag}/

model_tag is recovered from logs/ (the "loading weights file" / tokenizer
"name_or_path=" lines record the real model each run actually loaded). Runs that
share a (parent, model_tag) keep only the newest timestamp. A run_meta.json is
written into each migrated dir.

Usage:
    python scripts/migrate_run_layout.py              # dry-run (default)
    python scripts/migrate_run_layout.py --apply      # actually move
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.registry import model_tag_from

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

TS_RE = re.compile(r"^\d{8}_\d{6}$")
RUNPATH_RE = re.compile(r"(results/[^\s]+?/\d{8}_\d{6})")
WEIGHTS_RE = re.compile(r"loading weights file (\S+)/[^/\s]+\.(?:safetensors|bin)")
NAMEPATH_RE = re.compile(r"name_or_path='([^']+)'")
_MODEL_HINT = ("models/", "/mnt", "OpenGVLab", "Qwen", "lmms-lab", "openbmb",
               "google/", "zai-org", "moonshotai")


def build_log_index() -> dict[str, str]:
    """Map run_dir (relative results/... path) -> real model path, parsed from logs."""
    index: dict[str, str] = {}
    if not LOGS.exists():
        return index
    for log in LOGS.rglob("*.log"):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue
        run_m = RUNPATH_RE.search(text)
        if not run_m:
            continue
        run_path = run_m.group(1)
        model = None
        wm = WEIGHTS_RE.search(text)
        if wm:
            model = wm.group(1)
        else:
            for nm in NAMEPATH_RE.findall(text):
                if any(h in nm for h in _MODEL_HINT) and "/root/codespace" not in nm:
                    model = nm
                    break
        if model:
            # last writer wins is fine; same run_path logs carry same model
            index[run_path] = model
    return index


def find_old_runs() -> list[Path]:
    """All timestamp-leaf run dirs that contain responses/."""
    runs = []
    for resp in RESULTS.rglob("responses"):
        run_dir = resp.parent
        if TS_RE.match(run_dir.name) and resp.is_dir():
            runs.append(run_dir)
    return sorted(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually move dirs (default: dry-run)")
    args = ap.parse_args()

    log_index = build_log_index()
    old_runs = find_old_runs()

    # Group by destination (parent + model_tag); keep newest timestamp per dest.
    plan: dict[Path, list[tuple[str, Path, str]]] = {}
    unresolved: list[Path] = []
    for run in old_runs:
        rel = str(run.relative_to(ROOT)).replace("results/", "results/", 1)
        # normalize key to "results/..." form used in logs
        rel_results = str(run.relative_to(ROOT))
        model_path = log_index.get(rel_results) or log_index.get(rel_results.replace("\\", "/"))
        if not model_path:
            unresolved.append(run)
            continue
        tag = model_tag_from(model_path)
        dest = run.parent / tag
        plan.setdefault(dest, []).append((run.name, run, model_path))

    print(f"=== {len(old_runs)} old runs, {len(plan)} destinations, "
          f"{len(unresolved)} unresolved ===\n")

    moves: list[tuple[Path, Path, str]] = []
    for dest, entries in sorted(plan.items()):
        entries.sort(key=lambda e: e[0], reverse=True)  # newest timestamp first
        newest_ts, newest_run, model_path = entries[0]
        dropped = [e[0] for e in entries[1:]]
        print(f"DEST {dest.relative_to(ROOT)}")
        print(f"  keep  {newest_run.relative_to(ROOT)}  (model={model_path})")
        for d in dropped:
            print(f"  drop  {d}  (older, same model_tag)")
        moves.append((newest_run, dest, model_path))

    if unresolved:
        print("\n=== UNRESOLVED (no log → model unknown, left untouched) ===")
        for r in unresolved:
            print(f"  {r.relative_to(ROOT)}")

    if not args.apply:
        print("\n[dry-run] nothing moved. Re-run with --apply to execute.")
        return

    print("\n[apply] executing moves...")
    for src, dest, model_path in moves:
        if src == dest:
            print(f"  [skip] already at dest: {dest.relative_to(ROOT)}")
            continue
        if dest.exists():
            shutil.rmtree(dest)  # newest overwrites
        shutil.move(str(src), str(dest))
        meta = {
            "model_path": model_path,
            "model_tag": dest.name,
            "config_name": dest.parent.name,
            "experiment_id": dest.parent.parent.name if dest.parent.parent.name.startswith(("E", "A")) else "",
            "timestamp": src.name,
            "migrated_from": str(src.relative_to(ROOT)),
        }
        (dest / "run_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  moved {src.name} -> {dest.relative_to(ROOT)}")
    print("\n[done] migration complete.")


if __name__ == "__main__":
    main()
