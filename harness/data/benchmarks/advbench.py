"""
AdvBench loader — text-only single-turn jailbreak prompts.
HF source: walledai/AdvBench (or llm-attacks/llm-attacks repo's harmful_behaviors.csv)

Pull command:
    huggingface-cli download walledai/AdvBench --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/advbench

Each record is a single harmful instruction. Evaluator type = `rule`:
the canonical AdvBench metric is the fraction of responses that DO NOT contain
canonical refusal phrases (i.e., attack-success rate via keyword matching).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from harness.data.benchmarks.base import Benchmark


class AdvBenchBenchmark(Benchmark):
    name = "advbench"
    metric_name = "ASR"
    metric_direction = "↓"
    evaluator_type = "rule"

    def __init__(self, data_path: Path, max_samples: Optional[int] = None):
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self._records: Optional[list[dict]] = None

    def get_root(self) -> Path:
        return self.data_path.parent if self.data_path.is_file() else self.data_path

    def load(self) -> list[dict]:
        if self._records is not None:
            return self._records

        # Support both CSV (harmful_behaviors.csv) and JSON layouts.
        if self.data_path.suffix.lower() == ".csv":
            rows = self._load_csv()
        else:
            rows = self._load_json()

        records = []
        for i, r in enumerate(rows):
            goal = r.get("goal") or r.get("prompt") or r.get("instruction") or r.get("question") or ""
            records.append({
                "id": r.get("id", i),
                "question": goal,
                "image_path1": "",
                "image_path2": "",
                "category": "advbench",
                "sub_category": r.get("category", ""),
                "img_source": "",
                "benchmark": "advbench",
            })
        if self.max_samples:
            records = records[: self.max_samples]
        self._records = records
        return records

    def _load_csv(self) -> list[dict]:
        with open(self.data_path) as f:
            return list(csv.DictReader(f))

    def _load_json(self) -> list[dict]:
        with open(self.data_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data") or list(data.values())
        return data
