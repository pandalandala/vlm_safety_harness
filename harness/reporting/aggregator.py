"""
ResultAggregator: collect metrics across multiple runs, compute mean ± std.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from harness.config.registry import ExperimentRegistry


class ResultAggregator:
    def __init__(self, results_root: Optional[Path] = None):
        self.registry = ExperimentRegistry(results_root) if results_root else ExperimentRegistry()

    def aggregate(
        self,
        experiment_name: str,
        group: Optional[str] = None,
        benchmarks: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
    ) -> dict:
        """
        Aggregate metrics across all runs of an experiment.
        Returns: {benchmark: {metric: {"mean": float, "std": float, "n": int}}}
        """
        runs = self.registry.find_runs(experiment_name=experiment_name, group=group)
        if not runs:
            return {}

        bench_metrics: dict[str, dict[str, list[float]]] = {}
        for run in runs:
            m = run.get("metrics", {})
            bmarks = m.get("benchmarks", {})
            for bname, bm in bmarks.items():
                if benchmarks and bname not in benchmarks:
                    continue
                bench_metrics.setdefault(bname, {})
                for metric_name, val in bm.items():
                    if metrics and metric_name not in metrics:
                        continue
                    bench_metrics[bname].setdefault(metric_name, []).append(val)

        result = {}
        for bname, bm in bench_metrics.items():
            result[bname] = {}
            for metric_name, vals in bm.items():
                n = len(vals)
                mean = sum(vals) / n
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
                result[bname][metric_name] = {"mean": mean, "std": std, "n": n}

        return result

    def load_single_run(self, run_dir: Path) -> dict:
        """Load metrics from a single run directory."""
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            return {}
        with open(metrics_path) as f:
            return json.load(f)

    def compare_experiments(
        self,
        experiment_names: list[str],
        benchmark: str,
        metric: str = "ASR",
    ) -> dict[str, dict]:
        """Compare multiple experiments on a single benchmark/metric."""
        result = {}
        for name in experiment_names:
            agg = self.aggregate(experiment_name=name, benchmarks=[benchmark], metrics=[metric])
            val = agg.get(benchmark, {}).get(metric, {})
            result[name] = val
        return result
