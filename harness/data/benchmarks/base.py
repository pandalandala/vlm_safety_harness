"""Abstract base class for all benchmarks."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class Benchmark(ABC):
    name: str

    # Phase 4 additions for E3 reporting + Phase 5 evaluator routing.
    metric_name: str = "ASR"
    metric_direction: Literal["↑", "↓"] = "↓"
    evaluator_type: Literal["gpt4o", "rule", "harmbench", "accuracy"] = "gpt4o"

    @abstractmethod
    def load(self) -> list[dict]:
        """
        Return list of unified records:
        {
          "id": int | str,
          "question": str,
          "image_path1": str,
          "image_path2": str,   # empty string if benchmark is single-image
          "category": str,
          "sub_category": str,
          "img_source": str,
          # benchmark-specific extra fields allowed (e.g., gold_answer for MMStar)
        }
        """
        ...

    @abstractmethod
    def get_root(self) -> Path:
        """Return path to benchmark data root."""
        ...

    def to_sharegpt(self, output_path: Path) -> Path:
        """
        Convert benchmark records to ShareGPT JSON format compatible with
        LlamaFactory dataset_info.json registration. Used when running batch
        inference via LFInferenceBackend.

        Subclasses can override for benchmark-specific structure (e.g.,
        single-image benchmarks should emit only one <image> token).
        """
        records = self.load()
        sharegpt = []
        for r in records:
            images = [p for p in (r.get("image_path1"), r.get("image_path2")) if p]
            placeholders = "".join("<image>\n" for _ in images)
            sharegpt.append({
                "conversations": [
                    {"from": "human",
                     "value": placeholders + r.get("question", "")},
                    {"from": "gpt", "value": ""},
                ],
                "images": images,
                "id": r.get("id"),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
            })
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt, f, ensure_ascii=False, indent=2)
        return output_path

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
