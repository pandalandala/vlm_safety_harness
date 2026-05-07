#!/usr/bin/env python3
"""
Aggregate all experiment results and generate paper tables.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --group main --format latex --output paper_tables/main.tex
    python scripts/generate_report.py --group prelim
    python scripts/generate_report.py --experiments main_dreams_internvl main_baseline_mis
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config.registry import ExperimentRegistry
from harness.reporting.table_generator import TableGenerator


HARNESS_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="all", choices=["prelim", "main", "ablation", "all"])
    p.add_argument("--experiments", nargs="*", default=None, help="Specific experiment names to include")
    p.add_argument("--format", default="both", choices=["latex", "markdown", "both"])
    p.add_argument("--output", default=None, help="Output file path (default: print to stdout)")
    p.add_argument("--benchmark", default=None, help="Single benchmark for ablation table")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    registry = ExperimentRegistry()
    gen = TableGenerator()

    if args.experiments:
        exp_names = args.experiments
    else:
        runs = registry.find_runs(group=args.group if args.group != "all" else None)
        exp_names = list(dict.fromkeys(r["name"] for r in runs))

    if not exp_names:
        print(f"No experiments found for group={args.group}")
        return

    print(f"[report] Generating table for {len(exp_names)} experiments: {exp_names}")

    formats = ["latex", "markdown"] if args.format == "both" else [args.format]

    for fmt in formats:
        if args.group == "ablation":
            table = gen.ablation_table(
                ablation_group="ablation",
                benchmark=args.benchmark or "mis_hard",
                fmt=fmt,
            )
        elif args.group == "prelim":
            table = gen.prelim_summary_table(fmt=fmt)
        else:
            table = gen.main_results_table(exp_names, fmt=fmt)

        if args.output:
            ext = ".tex" if fmt == "latex" else ".md"
            out_path = Path(args.output).with_suffix(ext)
            gen.save(table, out_path)
            print(f"[report] Saved {fmt} table → {out_path}")
        else:
            print(f"\n{'='*60}")
            print(f"[{fmt.upper()}]")
            print(table)


if __name__ == "__main__":
    main()
