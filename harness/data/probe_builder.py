"""
ProbeBuilder: construct specialized test sets for A experiments.

ALL data must come from MIS test sets or public benchmarks (COCO/OpenImages).
DREAMS data is strictly forbidden here.

A1: text-only probe   — replace images with black frames, keep text
A2: relation probe    — MIS-hard + manually constructed non-standard relation samples
A3: synthetic-vs-real — direct use of MIS easy/hard/real (no building needed)
A4: counterfactual    — load MSSBench (safe/unsafe paired samples)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image


class ProbeBuilder:

    # ── A1: Text-only probe ───────────────────────────────────────────────

    @staticmethod
    def build_text_only_probe(
        mis_test_json: Path,
        output_path: Path,
        max_samples: Optional[int] = None,
    ) -> Path:
        """
        A1: Replace both images with 224×224 black frames.
        Image paths point to generated black images saved alongside output_path.

        Output JSON records keep original text fields intact.
        """
        with open(mis_test_json) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())
        if max_samples:
            raw = raw[:max_samples]

        black_img_path = Path(output_path).parent / "black_frame_224.png"
        if not black_img_path.exists():
            Image.new("RGB", (224, 224), color=(0, 0, 0)).save(black_img_path)
            black_img_path = black_img_path.resolve()

        records = []
        for r in raw:
            records.append({
                "id": r.get("id"),
                "question": r.get("question", ""),
                "image_path1": str(black_img_path),
                "image_path2": str(black_img_path),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", ""),
                "probe_type": "text_only",
                "original_image_path1": r.get("image_path1", ""),
                "original_image_path2": r.get("image_path2", ""),
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return output_path

    # ── A2: Relation-type probe ───────────────────────────────────────────

    @staticmethod
    def annotate_relation_types(
        mis_hard_json: Path,
        output_path: Path,
    ) -> Path:
        """
        A2 pre-step: tag each MIS-hard sample with its relation type using GPT-4o.

        Adds field: "relation_type": one of
          "tool_target" | "before_after" | "identity_linking" | "context_shift"

        This method writes a TODO list; actual GPT-4o calls happen in gpt4o_evaluator.py.
        Returns path to output (may be empty until evaluated).
        """
        with open(mis_hard_json) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())

        # Add placeholder field; filled by GPT-4o evaluator
        records = []
        for r in raw:
            records.append({**r, "relation_type": None})

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def build_relation_type_probe(
        mis_hard_json: Path,
        extra_probe_jsonl: Optional[Path],
        output_path: Path,
        mis_sample_n: int = 50,
    ) -> Path:
        """
        A2: Merge MIS-hard samples (tool→target) with non-standard relation samples.

        extra_probe_jsonl: JSONL of hand-crafted probes with "relation_type" field.
          Each record: {id, question, image_path1, image_path2, category, relation_type}
          Images sourced from COCO/OpenImages, text hand-written.
        """
        with open(mis_hard_json) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())

        records = []
        for r in raw[:mis_sample_n]:
            records.append({
                "id": r.get("id"),
                "question": r.get("question", ""),
                "image_path1": r.get("image_path1", ""),
                "image_path2": r.get("image_path2", ""),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "relation_type": "tool_target",
                "probe_type": "relation_type",
            })

        if extra_probe_jsonl and Path(extra_probe_jsonl).exists():
            with open(extra_probe_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        r["probe_type"] = "relation_type"
                        records.append(r)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return output_path

    # ── A3: Synthetic-real gap (no construction needed) ───────────────────

    @staticmethod
    def load_mis_subset(
        mis_test_json: Path,
        image_dir: Path,
        subset_label: str,
        max_samples: Optional[int] = None,
    ) -> list[dict]:
        """
        A3: Load MIS easy/hard/real subset as unified records.
        subset_label: "mis_easy" | "mis_hard" | "mis_real" (used in probe_type field)
        """
        with open(mis_test_json) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())
        if max_samples:
            raw = raw[:max_samples]

        records = []
        for i, r in enumerate(raw):
            img1 = str(image_dir / r["image_path1"]) if r.get("image_path1") else ""
            img2 = str(image_dir / r["image_path2"]) if r.get("image_path2") else ""
            records.append({
                "id": r.get("id", i),
                "question": r.get("question", ""),
                "image_path1": img1,
                "image_path2": img2,
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", ""),
                "probe_type": subset_label,
            })
        return records

    # ── A4: MSSBench counterfactual probe ─────────────────────────────────

    @staticmethod
    def load_mssbench(
        mssbench_path: Path,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        A4: Load MSSBench into unified record format.

        MSSBench has paired safe/unsafe samples with the same text instruction.
        Expected source format (may vary by MSSBench version):
        {
          "id": int,
          "question": str,
          "image_path": str,       # single image (MSSBench is single-image)
          "is_safe": bool,
          "category": str,
          "pair_id": int           # safe and unsafe share the same pair_id
        }

        We adapt to dual-image format by using the single image as both image1/image2.
        """
        mssbench_path = Path(mssbench_path)

        if mssbench_path.suffix == ".json":
            with open(mssbench_path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = list(raw.values())
        else:
            raw = []
            with open(mssbench_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))

        records = []
        for i, r in enumerate(raw):
            img = r.get("image_path", r.get("image", ""))
            records.append({
                "id": r.get("id", i),
                "question": r.get("question", r.get("instruction", "")),
                "image_path1": str(img),
                "image_path2": str(img),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": "MSSBench",
                "is_safe": r.get("is_safe", r.get("safe", None)),
                "pair_id": r.get("pair_id", None),
                "probe_type": "mssbench",
            })

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

        return records

    @staticmethod
    def split_mssbench_by_safety(records: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split MSSBench records into unsafe and safe subsets."""
        unsafe = [r for r in records if r.get("is_safe") is False]
        safe   = [r for r in records if r.get("is_safe") is True]
        return unsafe, safe
