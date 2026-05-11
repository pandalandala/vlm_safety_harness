"""
MMStar loader — capability anchor for E4.
HF source: Lin-Chen/MMStar

Pull command:
    huggingface-cli download Lin-Chen/MMStar --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mmstar

MMStar is multiple-choice (A/B/C/D). Evaluator type = `accuracy` —
exact-match against `gold_answer` field.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from harness.data.benchmarks.base import Benchmark


class MMStarBenchmark(Benchmark):
    name = "mmstar"
    metric_name = "Accuracy"
    metric_direction = "↑"
    evaluator_type = "accuracy"

    def __init__(self, data_path: Path, image_root: Optional[Path] = None,
                 max_samples: Optional[int] = None):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else self.data_path.parent
        self.max_samples = max_samples
        self._records: Optional[list[dict]] = None

    def get_root(self) -> Path:
        return self.image_root

    def load(self) -> list[dict]:
        if self._records is not None:
            return self._records

        # MMStar ships parquet shards; user must convert to JSONL or pass JSON.
        # Defer parquet handling to a follow-up — accept JSON/JSONL here.
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(self.data_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())

        records = []
        for i, r in enumerate(data):
            img = r.get("image") or r.get("image_path") or ""
            records.append({
                "id": r.get("id", i),
                "question": r.get("question", ""),
                "image_path1": str(self._resolve(img)) if img else "",
                "image_path2": "",
                "category": r.get("category", "mmstar"),
                "sub_category": r.get("l2_category") or r.get("sub_category", ""),
                "img_source": "",
                "gold_answer": r.get("answer") or r.get("gold_answer", ""),
                "options": r.get("options") or r.get("choices", []),
                "benchmark": "mmstar",
            })
        if self.max_samples:
            records = records[: self.max_samples]
        self._records = records
        return records

    def _resolve(self, p: str) -> Path:
        candidate = Path(p)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        return self.image_root / p
