#!/usr/bin/env python3
"""
Generate Tier B baseline inference YAMLs from _baseline_template.yaml + _tier_b_models.csv.

Usage:
    python scripts/generate_baseline_configs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "configs/experiments/main/_baseline_template.yaml"
CSV = ROOT / "configs/experiments/main/_tier_b_models.csv"
OUT_DIR = ROOT / "configs/experiments/main"


def main() -> int:
    template = TEMPLATE.read_text()
    written = 0
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = (
                template
                .replace("__NAME__", f"main_baseline_{row['name']}")
                .replace("__DESC__", row["desc"])
                .replace("__MODEL_NAME__", row["name"])
                .replace("__HF_PATH__", row["hf_path"])
                .replace("__ARCH__", row["arch"])
                .replace("__SIZE_B__", row["size_b"])
                .replace("__TRUST_REMOTE_CODE__", row["trust_remote_code"])
            )
            out = OUT_DIR / f"main_baseline_{row['name']}.yaml"
            out.write_text(content)
            print(f"  wrote {out}")
            written += 1
    print(f"Generated {written} Tier B configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
