"""
Unified dataset loader for DREAMS and MIS data formats.
Supports training (with CoT responses) and evaluation (response-free) modes.

Two on-disk record schemas are accepted:

A) DREAMS train.json / scored.json — sharegpt with CoT response:
   {id, category, sub_category, conversations[2], image[2], img_source, vlm_score}

B) DREAMS test.json — flat eval format:
   {id, category, sub_category, question, image_path1, image_path2,
    path_name, img1_source, img2_source, harm_type}

`__getitem__` normalizes both shapes into a single dict.
`derive_img_source_type` maps the (img1_source, img2_source) pair into
{synth, real, mix, unknown} for E2 slicing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal, Optional

from PIL import Image


_SYNTH_TOKENS = {"AI-generated", "SD-3.5", "Stable Diffusion", "synth", "synthetic"}
_REAL_TOKENS = {"Web-crawled", "Web-retrieved", "local", "web", "real"}


def derive_img_source_type(record: dict) -> Literal["synth", "real", "mix", "unknown"]:
    """
    Rule-based mapping of (img1_source, img2_source) to a single coarse label.

    Both synth → synth; both real → real; one of each → mix; otherwise unknown.
    Falls back to single `img_source` field for DREAMS train.json schema.
    """
    a = record.get("img1_source") or ""
    b = record.get("img2_source") or ""
    if not (a or b):
        single = record.get("img_source") or ""
        a = b = single

    is_synth_a = a in _SYNTH_TOKENS
    is_synth_b = b in _SYNTH_TOKENS
    is_real_a = a in _REAL_TOKENS
    is_real_b = b in _REAL_TOKENS

    if is_synth_a and is_synth_b:
        return "synth"
    if is_real_a and is_real_b:
        return "real"
    if (is_synth_a and is_real_b) or (is_real_a and is_synth_b):
        return "mix"
    return "unknown"


class HarnessDataset:
    """
    Loads multi-image safety datasets in DREAMS or MIS format.

    Expected JSON structure (DREAMS/scored.json format):
    {
      "id": int,
      "category": str,
      "sub_category": str | null,
      "conversations": [
        {"from": "human", "value": "<image>\\n<image>\\n{question}"},
        {"from": "gpt",   "value": "{cot_response or empty}"}
      ],
      "image": ["images/{id}/object1.png", "images/{id}/object2.png"],
      "img_source": str,
      "vlm_score": int
    }
    """

    def __init__(
        self,
        data_path: Path,
        image_root: Path,
        mode: Literal["train", "eval"] = "eval",
        max_samples: Optional[int] = None,
        categories: Optional[list[str]] = None,
        filter_img_source: Optional[str] = None,
        filter_harm_type: Optional[Literal["explicit", "implicit"]] = None,
        filter_img_source_type: Optional[Literal["synth", "real", "mix", "unknown"]] = None,
        text_only_mode: bool = False,
        blank_image_path: Optional[Path] = None,
    ):
        self.image_root = Path(image_root)
        self.mode = mode
        self.text_only_mode = text_only_mode
        self.blank_image_path = blank_image_path

        with open(data_path) as f:
            raw = json.load(f)

        # Normalize list or dict
        if isinstance(raw, dict):
            raw = list(raw.values())

        # Apply filters
        if categories:
            categories_upper = [c.upper() for c in categories]
            raw = [r for r in raw if r.get("category", "").upper() in categories_upper]

        if filter_img_source:
            raw = [
                r for r in raw
                if filter_img_source.lower() in r.get("img_source", "").lower()
                or filter_img_source.lower() in r.get("img1_source", "").lower()
                or filter_img_source.lower() in r.get("img2_source", "").lower()
            ]

        if filter_harm_type:
            raw = [r for r in raw if r.get("harm_type") == filter_harm_type]

        if filter_img_source_type:
            raw = [r for r in raw if derive_img_source_type(r) == filter_img_source_type]

        if max_samples:
            raw = raw[:max_samples]

        self._records = raw

    @classmethod
    def from_config(cls, cfg_dataset, mode: str = "eval") -> "HarnessDataset":
        from harness.config.schema import DatasetConfig
        assert isinstance(cfg_dataset, DatasetConfig)
        path = cfg_dataset.train_path if mode == "train" else cfg_dataset.test_path
        assert path is not None, f"Dataset path for mode={mode} not set in config"
        return cls(
            data_path=path,
            image_root=cfg_dataset.image_root or path.parent,
            mode=mode,
            max_samples=cfg_dataset.max_train_samples if mode == "train" else cfg_dataset.max_test_samples,
            categories=cfg_dataset.categories,
            filter_img_source=cfg_dataset.filter_img_source,
            filter_harm_type=cfg_dataset.filter_harm_type,
            filter_img_source_type=cfg_dataset.filter_img_source_type,
            text_only_mode=cfg_dataset.text_only_mode,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        question = self._extract_question(r)
        image_paths = self._resolve_image_paths(r)

        if self.text_only_mode:
            img1 = self._get_blank_image()
            img2 = self._get_blank_image()
        else:
            img1 = Image.open(image_paths[0]).convert("RGB") if image_paths else None
            img2 = Image.open(image_paths[1]).convert("RGB") if len(image_paths) > 1 else None

        cot_response = ""
        if self.mode == "train" and len(r.get("conversations", [])) > 1:
            cot_response = r["conversations"][1].get("value", "")

        return {
            "id": r.get("id", idx),
            "question": question,
            "image1": img1,
            "image2": img2,
            "image_path1": str(image_paths[0]) if image_paths else "",
            "image_path2": str(image_paths[1]) if len(image_paths) > 1 else "",
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
            "img_source": r.get("img_source", ""),
            "img1_source": r.get("img1_source", ""),
            "img2_source": r.get("img2_source", ""),
            "img_source_type": derive_img_source_type(r),
            "harm_type": r.get("harm_type", ""),
            "path_name": r.get("path_name", ""),
            "vlm_score": r.get("vlm_score", -1),
            "cot_response": cot_response,
        }

    def iter_batches(self, batch_size: int) -> Iterator[list[dict]]:
        batch = []
        for i in range(len(self)):
            batch.append(self[i])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def get_category_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self._records:
            cat = r.get("category", "UNKNOWN")
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    def get_source_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self._records:
            src = r.get("img_source", "unknown")
            dist[src] = dist.get(src, 0) + 1
        return dist

    def to_eval_records(self) -> list[dict]:
        """Return list of dicts suitable for batch inference input.
        Includes harm_type and img_source_type for E1/E2 slicing."""
        records = []
        for i in range(len(self)):
            item = self[i]
            records.append({
                "id": item["id"],
                "question": item["question"],
                "image_path1": item["image_path1"],
                "image_path2": item["image_path2"],
                "category": item["category"],
                "sub_category": item.get("sub_category", ""),
                "img_source": item.get("img_source", ""),
                "img_source_type": item["img_source_type"],
                "harm_type": item["harm_type"],
                "path_name": item.get("path_name", ""),
            })
        return records

    def to_train_records(self) -> list[dict]:
        """Return lightweight records for LLaMA-Factory training export.

        This avoids eagerly opening every image file. Training export only
        needs text, image paths, and metadata.
        """
        records = []
        for r in self._records:
            question = self._extract_question(r)
            image_paths = self._resolve_image_paths(r)
            cot_response = ""
            if self.mode == "train" and len(r.get("conversations", [])) > 1:
                cot_response = r["conversations"][1].get("value", "")

            records.append({
                "id": r.get("id"),
                "question": question,
                "image_path1": str(image_paths[0]) if image_paths else "",
                "image_path2": str(image_paths[1]) if len(image_paths) > 1 else "",
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", ""),
                "img1_source": r.get("img1_source", ""),
                "img2_source": r.get("img2_source", ""),
                "harm_type": r.get("harm_type", ""),
                "path_name": r.get("path_name", ""),
                "cot_response": cot_response,
            })
        return records

    def _extract_question(self, r: dict) -> str:
        convs = r.get("conversations", [])
        if convs:
            val = convs[0].get("value", "")
            # Strip leading <image>\n<image>\n tokens
            val = val.replace("<image>\n<image>\n", "").replace("<image>\n", "").strip()
            return val
        return r.get("text_prompt", r.get("question", ""))

    def _resolve_image_paths(self, r: dict) -> list[Path]:
        # Schema A (DREAMS train.json): "image" → list of paths
        # Schema B (DREAMS test.json): "image_path1" / "image_path2" → strings
        images = r.get("image")
        if not images:
            images = [p for p in (r.get("image_path1"), r.get("image_path2")) if p]

        resolved = []
        for img in images:
            p = Path(img)
            if p.is_absolute() and p.exists():
                resolved.append(p)
            else:
                candidate = self.image_root / img
                resolved.append(candidate)  # Keep path even if not found
        return resolved

    def _get_blank_image(self) -> Image.Image:
        if self.blank_image_path and Path(self.blank_image_path).exists():
            return Image.open(self.blank_image_path).convert("RGB")
        # Generate a 224x224 black image
        return Image.new("RGB", (224, 224), color=(0, 0, 0))
