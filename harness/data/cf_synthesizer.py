"""
CFSynthesizer — build counterfactual (CF) pairs for E5 (CF Consistency).

DREAMS test.json has no native CF pair metadata. This module synthesizes pairs
offline by image-swap: for each unsafe test record, replace one of its two
images with a benign image sampled from a public pool (OpenImages / ImageNet /
COCO / any dir of images). The result is a `(orig_unsafe, cf_safe)` pair.

Design choices:
  - cf_id = orig_id + 1_000_000 (collision-safe offset; orig_ids fit in 6 digits)
  - default swap_idx = 2 (replace the second image)
  - benign pool = directory of image files (any `.jpg/.jpeg/.png/.webp`)
  - optional quality judge: post-hoc safety verification via GPT-4o-mini judge
    can be wired in build_cf_pairs.py (not enforced here)

Output schema (per record):
  {
    "orig_id": 15320,
    "cf_id":   1015320,
    "swap_idx": 2,
    "image_path1": "<retained-from-orig>",  # original (unchanged)
    "image_path2": "<benign-image-path>",   # swapped
    "benign_image_path": "<absolute-path-to-original-benign>",
    "question":   "<copied from orig>",
    "category":   "<copied from orig>",
    "sub_category": "<copied from orig>",
    "harm_type":  "<copied from orig>",
    "img_source_type": "cf_swapped",       # post-swap label
    "cf_label":   "safe"
  }
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Optional

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CF_ID_OFFSET = 1_000_000


def collect_benign_pool(pool_dir: Path) -> list[Path]:
    """Recursively collect image files from `pool_dir`."""
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        raise FileNotFoundError(f"benign pool dir does not exist: {pool_dir}")
    images = [p for p in pool_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS]
    if not images:
        raise ValueError(f"no images found under {pool_dir}")
    return images


class CFSynthesizer:
    def __init__(
        self,
        benign_pool: Iterable[Path] | Path,
        swap_image_idx: int = 2,
        seed: int = 0,
    ):
        if isinstance(benign_pool, (str, Path)):
            benign_pool = collect_benign_pool(Path(benign_pool))
        self.benign_pool: list[Path] = [Path(p) for p in benign_pool]
        if swap_image_idx not in (1, 2):
            raise ValueError("swap_image_idx must be 1 or 2")
        self.swap_image_idx = swap_image_idx
        self._rng = random.Random(seed)

    # ── Public API ────────────────────────────────────────────────────────

    def synthesize(
        self,
        records: list[dict],
        cf_image_dir: Optional[Path] = None,
    ) -> list[dict]:
        """
        For each input record, sample one benign image and emit a CF record.

        Args:
            records: list of unsafe DREAMS test records (must include
                     id, question, image_path1, image_path2, category, harm_type).
            cf_image_dir: optional dir to copy chosen benign image into,
                          renamed to `<cf_id>.<ext>`. If None, references the
                          benign image at its original pool path.

        Returns:
            list of CF records (see module docstring for schema).
        """
        if cf_image_dir:
            cf_image_dir = Path(cf_image_dir)
            cf_image_dir.mkdir(parents=True, exist_ok=True)

        out: list[dict] = []
        for r in records:
            orig_id = r["id"]
            cf_id = orig_id + _CF_ID_OFFSET
            benign_src = self._sample_benign()

            if cf_image_dir is not None:
                dest = cf_image_dir / f"{cf_id}{benign_src.suffix.lower()}"
                shutil.copyfile(benign_src, dest)
                benign_path = dest
            else:
                benign_path = benign_src

            keep_path = r["image_path1"] if self.swap_image_idx == 2 else r["image_path2"]
            new_image_path1 = keep_path if self.swap_image_idx == 2 else str(benign_path)
            new_image_path2 = str(benign_path) if self.swap_image_idx == 2 else keep_path

            out.append({
                "orig_id": orig_id,
                "cf_id": cf_id,
                "id": cf_id,                  # alias so downstream loaders work
                "swap_idx": self.swap_image_idx,
                "image_path1": new_image_path1,
                "image_path2": new_image_path2,
                "benign_image_path": str(benign_src),
                "question": r.get("question", ""),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "harm_type": r.get("harm_type", ""),
                "img_source_type": "cf_swapped",
                "cf_label": "safe",
            })
        return out

    @staticmethod
    def to_pair_index(cf_records: list[dict]) -> dict[int, int]:
        """Return {orig_id: cf_id} mapping."""
        return {r["orig_id"]: r["cf_id"] for r in cf_records}

    @staticmethod
    def write_jsonl(cf_records: list[dict], output_path: Path) -> Path:
        """Write CF records as JSON array to `output_path`."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cf_records, f, ensure_ascii=False, indent=2)
        return output_path

    # ── Internals ─────────────────────────────────────────────────────────

    def _sample_benign(self) -> Path:
        return self._rng.choice(self.benign_pool)
