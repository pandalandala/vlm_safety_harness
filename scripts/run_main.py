#!/usr/bin/env python3
"""
Orchestrator for E1/E2/E3/E5 main experiments. Drives a cohort of configs over
the relevant pipeline slice (train + infer + eval + slicing).

Cohort definition lives at configs/experiments/main/_cohort.yaml.

Examples:
    python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams
    python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams,tier_b
    python scripts/run_main.py --experiment-id E5 --cohort tier_a_dreams,tier_a_mirage_data
    python scripts/run_main.py --experiment-id E1 --dry-run --limit 5

For E5: requires `dataset.cf_pairs_path` to point to test_cf.json (built once
via scripts/build_cf_pairs.py).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COHORT_FILE = ROOT / "configs/experiments/main/_cohort.yaml"

# Map experiment id → benchmark list + slicing flags + special pipeline behaviour.
EXPERIMENT_SPEC = {
    "E1": {
        "benchmarks": ["our_test"],
        "slice_fields": ["harm_type"],
        "needs_cf": False,
    },
    "E2": {
        "benchmarks": ["our_test"],
        "slice_fields": ["img_source_type"],
        "needs_cf": False,
    },
    "E3": {
        "benchmarks": ["advbench", "safebench", "mm_safety", "jailbreakv",
                       "siuo", "mss"],
        "slice_fields": [],
        "needs_cf": False,
    },
    "E5": {
        "benchmarks": ["our_test"],   # plus CF inference run
        "slice_fields": [],
        "needs_cf": True,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True,
                   choices=["E1", "E2", "E3", "E5", "all"])
    p.add_argument("--cohort", default="tier_a_dreams,tier_a_mirage_data,tier_a_base,tier_b",
                   help="Comma-separated cohort group names from _cohort.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="Pass through to run_experiment.py")
    return p.parse_args()


def load_cohort(group_names: list[str]) -> list[str]:
    """Load cohort YAML, return flat list of config filenames."""
    with open(COHORT_FILE) as f:
        cohort = yaml.safe_load(f) or {}
    out: list[str] = []
    for g in group_names:
        if g not in cohort:
            raise ValueError(f"Unknown cohort group: {g}. Available: {list(cohort.keys())}")
        for cfg in cohort[g]:
            if cfg not in out:
                out.append(cfg)
    return out


def run_one(
    config_name: str,
    benchmarks: list[str],
    args: argparse.Namespace,
) -> int:
    """Invoke scripts/run_experiment.py for a single config."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_experiment.py"),
        f"main/{config_name}",
        "--benchmarks", *benchmarks,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.force:
        cmd += ["--force"]
    if args.skip_train:
        cmd += ["--skip-train"]

    print(f"[run] {' '.join(cmd)}")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def precheck_e5(cohort_configs: list[str]) -> None:
    """E5 requires test_cf.json. Halt with clear error if missing."""
    cf_path = Path("/mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json")
    if not cf_path.exists():
        print(
            f"\n[precheck] E5 requires CF pair file at:\n  {cf_path}\n"
            "Run scripts/build_cf_pairs.py first. Example:\n"
            "  python scripts/build_cf_pairs.py \\\n"
            "    --test-json /mnt/hdd/xuran/mis_dataset_builder/dataset/test.json \\\n"
            "    --benign-pool <path-to-benign-image-dir> \\\n"
            "    --output /mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json \\\n"
            "    --cf-images-dir /mnt/hdd/xuran/mis_dataset_builder/dataset/cf_images\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    args = parse_args()

    exp_ids = ["E1", "E2", "E3", "E5"] if args.experiment_id == "all" else [args.experiment_id]
    cohort_groups = [g.strip() for g in args.cohort.split(",") if g.strip()]
    cohort_configs = load_cohort(cohort_groups)

    print(f"[plan] experiments={exp_ids} cohort={cohort_groups} ({len(cohort_configs)} configs)")

    for exp_id in exp_ids:
        spec = EXPERIMENT_SPEC[exp_id]
        if spec["needs_cf"]:
            precheck_e5(cohort_configs)

        print(f"\n=== Running {exp_id} (benchmarks={spec['benchmarks']}) ===")
        for cfg_name in cohort_configs:
            rc = run_one(cfg_name, spec["benchmarks"], args)
            if rc != 0:
                print(f"[fail] {cfg_name} exit={rc}", file=sys.stderr)
                if not args.dry_run:
                    return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
