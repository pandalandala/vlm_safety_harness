"""
Experiment registry: tracks completed runs, supports deduplication and history queries.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .schema import ExperimentConfig

RESULTS_ROOT = Path(__file__).parent.parent.parent / "results"


def _config_hash(cfg: ExperimentConfig) -> str:
    """SHA256 of the serialized config (excluding tracking/output_dir fields)."""
    d = cfg.model_dump(exclude={"tracking": True, "training": {"output_dir"}})
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class ExperimentRegistry:
    def __init__(self, results_root: Path = RESULTS_ROOT):
        self.results_root = results_root

    def make_run_dir(self, cfg: ExperimentConfig) -> Path:
        """Create and return a timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_root / cfg.group / cfg.name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        snapshot_path = run_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w") as f:
            yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False, allow_unicode=True)

        # Save config hash
        (run_dir / "config_hash.txt").write_text(_config_hash(cfg))

        return run_dir

    def is_completed(self, cfg: ExperimentConfig) -> Optional[Path]:
        """
        Check if an identical config has already been run successfully.
        Returns the most recent matching run_dir, or None.
        """
        cfg_hash = _config_hash(cfg)
        group_dir = self.results_root / cfg.group / cfg.name
        if not group_dir.exists():
            return None

        for run_dir in sorted(group_dir.iterdir(), reverse=True):
            hash_file = run_dir / "config_hash.txt"
            metrics_file = run_dir / "metrics.json"
            if hash_file.exists() and metrics_file.exists():
                if hash_file.read_text().strip() == cfg_hash:
                    return run_dir
        return None

    def find_runs(
        self,
        experiment_name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """List completed runs with their metrics."""
        runs = []
        search_root = self.results_root
        if group:
            search_root = search_root / group
        if experiment_name:
            search_root = search_root / experiment_name

        for metrics_path in search_root.rglob("metrics.json"):
            run_dir = metrics_path.parent
            try:
                metrics = json.loads(metrics_path.read_text())
                cfg_path = run_dir / "config_snapshot.yaml"
                cfg_data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
                run_tags = cfg_data.get("tags", [])

                if tags and not all(t in run_tags for t in tags):
                    continue

                runs.append({
                    "run_dir": str(run_dir),
                    "name": cfg_data.get("name", run_dir.parent.name),
                    "group": cfg_data.get("group", run_dir.parent.parent.name),
                    "tags": run_tags,
                    "metrics": metrics,
                    "timestamp": run_dir.name,
                })
            except Exception:
                continue

        return sorted(runs, key=lambda r: r["timestamp"], reverse=True)

    def get_best_run(self, experiment_name: str, metric: str = "ASR_mis_hard") -> Optional[dict]:
        """Return the run with the best (lowest ASR) for the given experiment."""
        runs = self.find_runs(experiment_name=experiment_name)
        if not runs:
            return None

        def _get_metric(run: dict) -> float:
            m = run.get("metrics", {})
            # Try nested: benchmarks.mis_hard.ASR
            parts = metric.split("_", 1)
            if len(parts) == 2:
                bench_metrics = m.get("benchmarks", {}).get(parts[1], {})
                return bench_metrics.get(parts[0], float("inf"))
            return m.get("overall", {}).get(metric, float("inf"))

        return min(runs, key=_get_metric)
