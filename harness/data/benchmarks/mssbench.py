"""MSSBench benchmark loader (safe/unsafe paired samples)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from .base import Benchmark


class MSSBenchmark(Benchmark):
    """
    MSSBench: paired safe/unsafe samples sharing the same text instruction.
    Used in A4 (Counterfactual) experiment to test model's fine-grained safety boundary.

    Expected structure (flexible — adapts to common MSSBench layouts):
    {
      "id": int,
      "instruction" | "question": str,
      "image_path" | "image": str,
      "is_safe" | "safe": bool,
      "category": str,
      "pair_id": int   (safe+unsafe share same pair_id)
    }
    """

    name = "mssbench"

    def __init__(
        self,
        data_path: Path,
        image_root: Optional[Path] = None,
        split: Optional[Literal["safe", "unsafe", "all"]] = "all",
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else self.data_path.parent
        self.split = split
        self.max_samples = max_samples
        self._records: Optional[list[dict]] = None

    def get_root(self) -> Path:
        return self.data_path.parent

    def load(self) -> list[dict]:
        if self._records is not None:
            return self._records

        raw = self._load_raw()
        records = []
        for i, r in enumerate(raw):
            img = r.get("image_path", r.get("image", ""))
            if img and not Path(img).is_absolute():
                img = str(self.image_root / img)

            is_safe = r.get("is_safe", r.get("safe", None))

            if self.split == "safe" and not is_safe:
                continue
            if self.split == "unsafe" and is_safe:
                continue

            records.append({
                "id": r.get("id", i),
                "question": r.get("question", r.get("instruction", "")),
                "image_path1": str(img),
                "image_path2": str(img),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": "MSSBench",
                "is_safe": is_safe,
                "pair_id": r.get("pair_id"),
                "benchmark": self.name,
            })

        if self.max_samples:
            records = records[:self.max_samples]

        self._records = records
        return records

    def _load_raw(self) -> list[dict]:
        if self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                raw = json.load(f)
            return raw if isinstance(raw, list) else list(raw.values())
        else:
            raw = []
            with open(self.data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
            return raw

    def get_unsafe(self) -> list[dict]:
        return [r for r in self.load() if r.get("is_safe") is False]

    def get_safe(self) -> list[dict]:
        return [r for r in self.load() if r.get("is_safe") is True]

    def get_pairs(self) -> list[tuple[dict, dict]]:
        """Return (unsafe, safe) pairs matched by pair_id."""
        by_pair: dict[int, dict[str, dict]] = {}
        for r in self.load():
            pid = r.get("pair_id")
            if pid is None:
                continue
            if pid not in by_pair:
                by_pair[pid] = {}
            key = "safe" if r.get("is_safe") else "unsafe"
            by_pair[pid][key] = r

        pairs = []
        for pid, group in by_pair.items():
            if "unsafe" in group and "safe" in group:
                pairs.append((group["unsafe"], group["safe"]))
        return pairs
