"""FigStep benchmark loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import Benchmark


class FigStepBenchmark(Benchmark):
    """
    FigStep: safety benchmark where harmful instructions are embedded in images as text.
    Single-image benchmark; image_path2 is left empty.

    Expected record format:
    {
      "id": int,
      "question" | "text": str,     (text visible to the model, typically generic)
      "image_path": str,             (image containing harmful text overlay)
      "category": str,
      "harmful_text": str            (the harmful text embedded in the image)
    }
    """

    name = "figstep"

    def __init__(
        self,
        data_path: Path,
        image_root: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else self.data_path.parent
        self.max_samples = max_samples
        self._records: Optional[list[dict]] = None

    def get_root(self) -> Path:
        return self.data_path.parent

    def load(self) -> list[dict]:
        if self._records is not None:
            return self._records

        if self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = list(raw.values())
        else:
            raw = []
            with open(self.data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))

        if self.max_samples:
            raw = raw[:self.max_samples]

        records = []
        for i, r in enumerate(raw):
            img = r.get("image_path", r.get("image", ""))
            if img and not Path(img).is_absolute():
                img = str(self.image_root / img)

            records.append({
                "id": r.get("id", i),
                "question": r.get("question", r.get("text", "Please follow the instructions shown in the image.")),
                "image_path1": str(img),
                "image_path2": "",
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": "FigStep",
                "harmful_text": r.get("harmful_text", ""),
                "benchmark": self.name,
            })

        self._records = records
        return records
