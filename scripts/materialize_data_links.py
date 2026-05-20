#!/usr/bin/env python
"""Materialize data_links/<name> symlinks into real-file snapshots.

Default mode is --dry-run. Use --apply to actually rsync and swap.

Per-target steps:
  1. resolve data_links/<name> symlink -> src
  2. rsync -aL src/ data_links/.<name>.materialize_tmp/   (-L dereferences inner symlinks)
  3. verify tmp has zero broken-and-zero remaining symlinks
  4. atomically swap: unlink original symlink, mv tmp -> data_links/<name>

Idempotent: refuses to act on a target that is already a real dir unless --force.
Resumable: rsync is restartable; tmp dir is reused if present.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/mnt/hdd/xuran/vlm_safety_harness")
DATA_LINKS = PROJECT_ROOT / "data_links"
LOG_DIR = PROJECT_ROOT / "logs" / "materialize"
DEFAULT_TARGETS = ["our_dataset", "mis_test", "mis_train"]


def count_files(path: Path, follow: bool) -> int:
    args = ["find", str(path)]
    if follow:
        args.append("-follow")
    args += ["-type", "f"]
    out = subprocess.run(args, capture_output=True, text=True, check=False)
    return sum(1 for line in out.stdout.splitlines() if line)


def count_broken_symlinks(path: Path) -> int:
    out = subprocess.run(
        ["find", str(path), "-type", "l", "-xtype", "l"],
        capture_output=True, text=True, check=False,
    )
    return sum(1 for line in out.stdout.splitlines() if line)


def count_remaining_symlinks(path: Path) -> int:
    out = subprocess.run(
        ["find", str(path), "-type", "l"],
        capture_output=True, text=True, check=False,
    )
    return sum(1 for line in out.stdout.splitlines() if line)


def dir_bytes(path: Path, follow: bool) -> int:
    flag = "-sbL" if follow else "-sb"
    out = subprocess.run(["du", flag, str(path)], capture_output=True, text=True, check=False)
    if not out.stdout.strip():
        return -1
    return int(out.stdout.split()[0])


def free_bytes(path: Path) -> int:
    s = shutil.disk_usage(path)
    return s.free


def materialize_one(name: str, *, apply: bool, force: bool) -> dict:
    entry = DATA_LINKS / name
    tmp = DATA_LINKS / f".{name}.materialize_tmp"
    report: dict = {"name": name, "entry": str(entry)}
    t0 = time.time()

    if not entry.exists() and not entry.is_symlink():
        report["status"] = "missing"
        return report

    if entry.is_symlink():
        src = entry.resolve(strict=False)
        report["src"] = str(src)
        if not src.exists():
            report["status"] = "src_missing"
            return report
    else:
        # already a real dir
        if not force:
            report["status"] = "already_materialized"
            return report
        # force mode: treat current location as both src and dst -> abort with note
        report["status"] = "refused_force_on_real_dir"
        return report

    # broken-link precondition
    broken = count_broken_symlinks(src)
    report["src_broken_symlinks"] = broken
    if broken > 0:
        report["status"] = "src_has_broken_symlinks"
        return report

    src_file_count = count_files(src, follow=True)
    src_bytes = dir_bytes(src, follow=True)
    report["src_file_count"] = src_file_count
    report["src_bytes"] = src_bytes

    free = free_bytes(DATA_LINKS)
    report["free_bytes_before"] = free
    if src_bytes > 0 and src_bytes > free - (5 * 1024**3):  # keep 5G headroom
        report["status"] = "insufficient_disk"
        return report

    if not apply:
        report["status"] = "dry_run_ok"
        report["elapsed_sec"] = round(time.time() - t0, 2)
        return report

    # rsync into tmp (resumable)
    tmp.mkdir(parents=True, exist_ok=True)
    rsync_cmd = [
        "rsync", "-aL", "--info=progress2",
        f"{src}/", f"{tmp}/",
    ]
    report["rsync_cmd"] = " ".join(rsync_cmd)
    rc = subprocess.call(rsync_cmd)
    if rc != 0:
        report["status"] = f"rsync_failed_rc{rc}"
        return report

    tmp_remaining = count_remaining_symlinks(tmp)
    tmp_file_count = count_files(tmp, follow=False)
    tmp_bytes = dir_bytes(tmp, follow=False)
    report["tmp_remaining_symlinks"] = tmp_remaining
    report["tmp_file_count"] = tmp_file_count
    report["tmp_bytes"] = tmp_bytes

    if tmp_remaining != 0:
        report["status"] = "tmp_still_has_symlinks"
        return report
    if tmp_file_count < src_file_count:
        report["status"] = f"file_count_mismatch_{tmp_file_count}_vs_{src_file_count}"
        return report

    # atomic-ish swap
    entry.unlink()
    tmp.rename(entry)
    report["status"] = "ok"
    report["elapsed_sec"] = round(time.time() - t0, 2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="comma-separated subset of data_links entries")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", action="store_true", default=True)
    grp.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if entry is already a real dir (currently no-op safety)")
    args = ap.parse_args()

    apply = bool(args.apply)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{stamp}.json"

    results = []
    for name in targets:
        print(f"\n=== {name} ({'APPLY' if apply else 'DRY-RUN'}) ===", flush=True)
        r = materialize_one(name, apply=apply, force=args.force)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)

    summary = {
        "timestamp_utc": stamp,
        "apply": apply,
        "targets": targets,
        "results": results,
    }
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written: {log_path}")

    bad = [r for r in results if r.get("status") not in ("ok", "dry_run_ok", "already_materialized")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
