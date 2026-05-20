"""MIS easy / hard / real benchmark loaders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from .base import Benchmark

MIS_TEST_ROOT = Path("/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test")

_SUBSET_META = {
    # image_path1/2 in MIS JSON are relative to MIS_TEST_ROOT (e.g. "easy_image/1/object1.png")
    "mis_easy": {"json_name": "mis_easy.json", "img_source": "AI-generated"},
    "mis_hard": {"json_name": "mis_hard.json", "img_source": "AI-generated"},
    "mis_real": {"json_name": "mis_real.json", "img_source": "Web-retrieved"},
}


class MISBenchmark(Benchmark):
    def __init__(
        self,
        subset: Literal["mis_easy", "mis_hard", "mis_real"],
        test_root: Optional[Path] = None,
        max_samples: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ):
        assert subset in _SUBSET_META, f"Unknown subset: {subset}"
        self.name = subset
        self.subset = subset
        self.max_samples = max_samples
        self.categories = [c.upper() for c in categories] if categories else None

        meta = _SUBSET_META[subset]
        self._root = Path(test_root) if test_root else MIS_TEST_ROOT
        self._json_path = self._root / meta["json_name"]
        self._default_img_source = meta["img_source"]

        self._records: Optional[list[dict]] = None

    def get_root(self) -> Path:
        return self._root

    def load(self) -> list[dict]:
        if self._records is not None:
            return self._records

        with open(self._json_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())

        if self.categories:
            raw = [r for r in raw if r.get("category", "").upper() in self.categories]

        if self.max_samples:
            raw = raw[:self.max_samples]

        records = []
        for i, r in enumerate(raw):
            # JSON paths are relative to MIS_TEST_ROOT
            img1 = str(self._root / r["image_path1"]) if r.get("image_path1") else ""
            img2 = str(self._root / r["image_path2"]) if r.get("image_path2") else ""
            records.append({
                "id": r.get("id", i),
                "question": r.get("question", ""),
                "image_path1": img1,
                "image_path2": img2,
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", self._default_img_source),
                "benchmark": self.name,
            })

        self._records = records
        return records

    @classmethod
    def all_subsets(
        cls,
        test_root: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> list["MISBenchmark"]:
        return [
            cls(subset=s, test_root=test_root, max_samples=max_samples)
            for s in ("mis_easy", "mis_hard", "mis_real")
        ]
