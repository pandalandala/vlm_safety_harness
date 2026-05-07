"""
TableGenerator: generate LaTeX / Markdown tables from experiment results.

Supports:
  - Main results table (models × benchmarks)
  - Ablation table
  - A experiment summary table (Section 2 of the paper)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from harness.config.registry import ExperimentRegistry
from harness.reporting.aggregator import ResultAggregator


METRICS = ["ASR", "RSR", "RR", "HR"]
BENCHMARKS = ["mis_easy", "mis_hard", "mis_real"]


class TableGenerator:
    def __init__(self, results_root: Optional[Path] = None):
        self.registry = ExperimentRegistry(results_root) if results_root else ExperimentRegistry()
        self.aggregator = ResultAggregator(results_root)

    # ── Main results table ─────────────────────────────────────────────────

    def main_results_table(
        self,
        experiment_names: list[str],
        benchmarks: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
        fmt: str = "latex",
    ) -> str:
        bmarks = benchmarks or BENCHMARKS
        mets = metrics or ["ASR"]

        rows: dict[str, dict] = {}
        for name in experiment_names:
            run = self.registry.get_best_run(name)
            if not run:
                continue
            m = run.get("metrics", {}).get("benchmarks", {})
            rows[name] = {
                (b, met): m.get(b, {}).get(met, None)
                for b in bmarks for met in mets
            }

        if fmt == "latex":
            return self._latex_table(rows, bmarks, mets)
        return self._markdown_table(rows, bmarks, mets)

    def ablation_table(
        self,
        ablation_group: str = "ablation",
        benchmark: str = "mis_hard",
        metrics: Optional[list[str]] = None,
        fmt: str = "latex",
    ) -> str:
        mets = metrics or METRICS
        runs = self.registry.find_runs(group=ablation_group)
        rows: dict[str, dict] = {}
        for run in runs:
            name = run.get("name", "unknown")
            m = run.get("metrics", {}).get("benchmarks", {}).get(benchmark, {})
            rows[name] = {(benchmark, met): m.get(met) for met in mets}

        if fmt == "latex":
            return self._latex_table(rows, [benchmark], mets)
        return self._markdown_table(rows, [benchmark], mets)

    def prelim_summary_table(self, fmt: str = "markdown") -> str:
        """A experiment summary in Section 2 style."""
        rows = {
            "A1 (text-only, base)":   {"Benchmark": "MIS-easy/hard", "Metric": "ΔASR(text-only)"},
            "A1 (text-only, MIRage)": {"Benchmark": "MIS-easy/hard", "Metric": "ΔASR(text-only)"},
            "A2 (relation, base)":    {"Benchmark": "Relation Probe", "Metric": "ASR per type"},
            "A2 (relation, MIRage)":  {"Benchmark": "Relation Probe", "Metric": "ASR per type"},
            "A3 (synth vs real)":     {"Benchmark": "MIS easy/hard/real", "Metric": "ASR gap"},
            "A4 (counterfactual)":    {"Benchmark": "MSSBench", "Metric": "Consistency / FPR"},
        }
        # Try to load actual numbers from results
        for name in ("A1_textual_shortcut", "A2_pattern_coverage", "A3_synthetic_real_gap", "A4_counterfactual"):
            run = self.registry.get_best_run(name)
            if run:
                m = run.get("metrics", {})
                # Fill rows with actual numbers when available

        if fmt == "markdown":
            lines = ["| Experiment | Benchmark | Metric | Finding |",
                     "|-----------|-----------|--------|---------|"]
            for exp, info in rows.items():
                lines.append(f"| {exp} | {info['Benchmark']} | {info['Metric']} | (see results/) |")
            return "\n".join(lines)
        return ""  # LaTeX version TBD

    # ── Formatting helpers ─────────────────────────────────────────────────

    def _latex_table(
        self,
        rows: dict[str, dict],
        bmarks: list[str],
        mets: list[str],
    ) -> str:
        col_keys = [(b, m) for b in bmarks for m in mets]
        n_cols = len(col_keys)

        # Find best (lowest) values per column for bolding
        best: dict[tuple, float] = {}
        for col in col_keys:
            vals = [v for v in (rows[r].get(col) for r in rows) if v is not None]
            if vals:
                best[col] = min(vals)  # Lower ASR = better

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l" + "r" * n_cols + "}",
            r"\toprule",
        ]

        # Header
        header_bmarks = " & ".join(
            f"\\multicolumn{{{len(mets)}}}{{c}}{{{b}}}" for b in bmarks
        )
        lines.append(f"Method & {header_bmarks} \\\\")
        lines.append("\\cmidrule(lr)" + " ".join(
            f"{{{2 + i * len(mets)}-{1 + (i+1) * len(mets)}}}" for i in range(len(bmarks))
        ))
        sub_header = " & ".join(mets * len(bmarks))
        lines.append(f" & {sub_header} \\\\")
        lines.append(r"\midrule")

        # Rows
        for exp_name, row in rows.items():
            cells = []
            for col in col_keys:
                v = row.get(col)
                if v is None:
                    cells.append("-")
                else:
                    s = f"{v * 100:.1f}"
                    if col in best and abs(v - best[col]) < 1e-6:
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
            lines.append(f"{exp_name} & " + " & ".join(cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def _markdown_table(
        self,
        rows: dict[str, dict],
        bmarks: list[str],
        mets: list[str],
    ) -> str:
        col_keys = [(b, m) for b in bmarks for m in mets]
        headers = ["Method"] + [f"{b}/{m}" for b, m in col_keys]

        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for exp_name, row in rows.items():
            cells = [exp_name]
            for col in col_keys:
                v = row.get(col)
                cells.append(f"{v*100:.1f}%" if v is not None else "-")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def save(self, content: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return output_path
