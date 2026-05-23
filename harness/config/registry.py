"""
Experiment registry: tracks completed runs, supports deduplication and history queries.
"""
from __future__ import annotations

import hashlib
import json
import re
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


def model_tag_from(source: str, fallback: str = "model") -> str:
    """Derive a stable, filesystem-safe tag identifying the model used in a run.

    Uses the basename of the model path / HF id so that base vs SFT models
    (which may share the same config) land in distinct, traceable directories.

      /mnt/.../models/dreams_internvl3_5      -> dreams_internvl3_5
      OpenGVLab/InternVL3_5-8B-HF             -> InternVL3_5-8B-HF
      Qwen/Qwen3.5-9B                         -> Qwen3.5-9B
    """
    p = (source or "").rstrip("/")
    name = Path(p).name if p else ""
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return tag or fallback


class ExperimentRegistry:
    def __init__(self, results_root: Path = RESULTS_ROOT):
        self.results_root = results_root

    def run_dir_for(
        self, cfg: ExperimentConfig, experiment_id: str = "", model_tag: str = "model"
    ) -> Path:
        """Return the canonical run directory for a (config, experiment, model).

        Path layout (model_tag leaf — stable, traceable, overwritten on re-run):
            results/{group}/{experiment_id}/{cfg.name}/{model_tag}/   (when experiment_id set)
            results/{group}/{cfg.name}/{model_tag}/                   (no experiment_id)
        """
        tag = model_tag or "model"
        if experiment_id:
            return self.results_root / cfg.group / experiment_id / cfg.name / tag
        return self.results_root / cfg.group / cfg.name / tag

    def make_run_dir(
        self,
        cfg: ExperimentConfig,
        experiment_id: str = "",
        model_tag: str = "model",
        model_path: str = "",
    ) -> Path:
        """Create and return the canonical run directory for this model.

        Same (config, experiment_id, model_tag) → same directory; a new run
        overwrites the previous one in place. Writes run_meta.json recording the
        real model_path so every run is traceable to the model that produced it.
        """
        run_dir = self.run_dir_for(cfg, experiment_id, model_tag)
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = run_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w") as f:
            yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False, allow_unicode=True)

        (run_dir / "config_hash.txt").write_text(_config_hash(cfg))
        if experiment_id:
            (run_dir / "experiment_id.txt").write_text(experiment_id)

        run_meta = {
            "model_path": model_path,
            "model_tag": model_tag,
            "config_name": cfg.name,
            "group": cfg.group,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

        return run_dir

    def is_completed(
        self, cfg: ExperimentConfig, experiment_id: str = "", model_tag: str = "model"
    ) -> Optional[Path]:
        """
        Check if this exact (config, experiment, model) already completed successfully.
        Returns the run_dir if config hash matches and metrics.json exists, else None.
        """
        cfg_hash = _config_hash(cfg)
        run_dir = self.run_dir_for(cfg, experiment_id, model_tag)
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

                meta_path = run_dir / "run_meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                # timestamp now lives in run_meta (dir leaf is model_tag, not a timestamp)
                timestamp = meta.get("timestamp") or datetime.fromtimestamp(
                    metrics_path.stat().st_mtime
                ).strftime("%Y%m%d_%H%M%S")

                runs.append({
                    "run_dir": str(run_dir),
                    "name": cfg_data.get("name", run_dir.parent.name),
                    "group": cfg_data.get("group"),
                    "tags": run_tags,
                    "metrics": metrics,
                    "timestamp": timestamp,
                    "model_tag": meta.get("model_tag", run_dir.name),
                    "model_path": meta.get("model_path", ""),
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
