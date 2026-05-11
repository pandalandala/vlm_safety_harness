"""
MM-SafetyBench loader.
HF source: isXinLiu/MM-SafetyBench

Pull command:
    huggingface-cli download isXinLiu/MM-SafetyBench --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mm_safety

Multimodal jailbreak benchmark — typographic image + text prompt.
Evaluator type = `gpt4o` per dataset README.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from harness.data.benchmarks.base import Benchmark


class MMSafetyBenchmark(Benchmark):
    name = "mm_safety"
    metric_name = "ASR"
    metric_direction = "↓"
    evaluator_type = "gpt4o"

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

        with open(self.data_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())

        records = []
        for i, r in enumerate(data):
            img1 = r.get("image_path") or r.get("image") or r.get("image_path1") or ""
            img2 = r.get("image_path2", "")
            records.append({
                "id": r.get("id", i),
                "question": r.get("question") or r.get("prompt") or r.get("instruction", ""),
                "image_path1": str(self._resolve(img1)) if img1 else "",
                "image_path2": str(self._resolve(img2)) if img2 else "",
                "category": r.get("category", "mm_safety"),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", ""),
                "benchmark": "mm_safety",
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
