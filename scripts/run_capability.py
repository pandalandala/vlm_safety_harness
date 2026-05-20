#!/usr/bin/env python3
"""
E4 capability orchestrator. Drives V0 / V1 / V3 (and V2 / V4 once unblocked)
against the capability benchmark suite (MMStar / MMMU / MuirBench / BLINK / MMT).

Hard guard: V2 / V4 launch blocked unless the config's training block contains
a populated `general_data` section (per docs/main_experiments/initial_framework_plan_v3 §5).

Usage:
    python scripts/run_capability.py --variants V0 V1 V3 --baseline internvl3_5
    python scripts/run_capability.py --variants V2 --baseline internvl3_5    # raises
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

CAPABILITY_BENCHMARKS = ["mmstar", "mmmu", "muirbench", "blink", "mmt"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["V0", "V1", "V3"],
                   help="Variants to run. V2/V4 require general-data mix.")
    p.add_argument("--baseline", default="internvl3_5",
                   help="Baseline arch suffix (e.g., internvl3_5).")
    p.add_argument("--benchmarks", nargs="+", default=CAPABILITY_BENCHMARKS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def variant_config_path(variant: str, baseline: str) -> Path:
    """Build the LF orchestration config path for this variant."""
    if variant == "V0":
        return ROOT / "configs/experiments/main" / f"E4_V0_{baseline}.yaml"
    if variant == "V1":
        return ROOT / "configs/experiments/main" / f"E4_V1_{baseline}_mirage.yaml"
    if variant == "V2":
        return ROOT / "configs/experiments/main" / f"E4_V2_{baseline}_mirage_general.yaml"
    if variant == "V3":
        return ROOT / "configs/experiments/main" / f"E4_V3_{baseline}_dreams.yaml"
    if variant == "V4":
        return ROOT / "configs/experiments/main" / f"E4_V4_{baseline}_dreams_general.yaml"
    raise ValueError(f"Unknown variant: {variant}")


def precheck_general_data(variant: str, config_path: Path) -> None:
    """Hard guard: V2/V4 require general_data populated. Else halt with reminder."""
    if variant not in ("V2", "V4"):
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    training = cfg.get("training") or {}
    general_data = training.get("general_data")
    if not general_data:
        raise SystemExit(
            f"\n[BLOCKED] {variant} ({config_path}) requires a general-data mix "
            "before launch.\n\n"
            "User asked to be reminded at this point — please specify the mix:\n"
            "  - source(s):    e.g., ShareGPT4V, LLaVA-OneVision-Data\n"
            "  - ratio:        DREAMS-vs-general sample counts\n"
            "  - total samples\n\n"
            "Update training.general_data block in:\n"
            f"  {config_path}\n"
            "and flip training.enabled to true. Then re-run.\n"
        )


def main() -> int:
    args = parse_args()

    for variant in args.variants:
        cfg_path = variant_config_path(variant, args.baseline)
        if not cfg_path.exists():
            print(f"[skip] {variant}: config not found at {cfg_path}", file=sys.stderr)
            continue

        precheck_general_data(variant, cfg_path)

        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_experiment.py"),
            str(cfg_path.relative_to(ROOT / "configs")),
            "--experiment-id", "E4",
            "--benchmarks", *args.benchmarks,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.dry_run:
            cmd += ["--dry-run"]
        # V0 has no SFT; V1/V3 have SFT enabled; V2/V4 require general_data (gated above)
        print(f"[run E4] variant={variant} ({cfg_path.name})")
        print(f"  cmd: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0 and not args.dry_run:
            print(f"[fail] {variant} exit={rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
