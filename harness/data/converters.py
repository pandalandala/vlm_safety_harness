"""
Format converters between DREAMS/MIS formats and downstream consumers.

  our_format  →  llamafactory_sharegpt  (training)
  our_format  →  mis_eval_format        (inference / evaluation)
  mis_eval_format  →  our_format        (loading MIS test sets as HarnessDataset)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from harness.config.schema import GeneralDataConfig


# ── LLaMA-Factory ──────────────────────────────────────────────────────────


def to_llamafactory_format(
    records: list[dict],
    image_root: Optional[Path] = None,
    use_cot: bool = True,
    cot_format: str = "structured",
) -> list[dict]:
    """
    Convert DREAMS records to LLaMA-Factory sharegpt multi-image format.

    LF expects:
    {
      "conversations": [
        {"from": "human", "value": "<image>\\n<image>\\n{question}"},
        {"from": "gpt",   "value": "{response}"}
      ],
      "images": ["/abs/path/img1.png", "/abs/path/img2.png"]
    }

    Args:
        records: list of HarnessDataset.__getitem__ dicts
        image_root: prepend to relative image paths (None = paths already absolute)
        use_cot: include cot_response as assistant turn (train mode)
        cot_format: "structured" wraps CoT in XML tags; "free_text" passes raw
    """
    out = []
    for r in records:
        question = r["question"]
        human_value = f"<image>\n<image>\n{question}"

        if use_cot and r.get("cot_response"):
            response = _format_cot(r["cot_response"], cot_format)
        else:
            response = ""

        images = []
        for key in ("image_path1", "image_path2"):
            p = r.get(key, "")
            if p:
                path = Path(p)
                if not path.is_absolute() and image_root:
                    path = image_root / path
                images.append(str(path))

        out.append({
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt",   "value": response},
            ],
            "images": images,
            # Preserve metadata for debugging
            "id": r.get("id"),
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
        })
    return out


def build_mixed_llamafactory_dataset(
    primary_records: list[dict],
    general_data_cfg: Optional[GeneralDataConfig],
    image_root: Optional[Path] = None,
    use_cot: bool = True,
    cot_format: str = "structured",
) -> list[dict]:
    primary = to_llamafactory_format(primary_records, image_root, use_cot, cot_format)
    if general_data_cfg is None:
        return primary

    general = load_general_sharegpt_records(general_data_cfg)
    return primary + general


def load_general_sharegpt_records(cfg: GeneralDataConfig) -> list[dict]:
    if cfg.format != "sharegpt":
        raise ValueError(f"Unsupported general_data format: {cfg.format}")

    all_records: list[dict] = []
    for source in cfg.sources:
        all_records.extend(_load_sharegpt_records_from_source(Path(source)))

    if not all_records:
        raise ValueError("No general-data samples loaded from configured sources.")

    rng = random.Random(cfg.shuffle_seed)
    rng.shuffle(all_records)
    return all_records


def resolve_general_sample_count(primary_count: int, cfg: GeneralDataConfig) -> int:
    if cfg.max_samples is not None:
        return cfg.max_samples
    if cfg.ratio is None:
        raise ValueError("general_data requires either max_samples or ratio.")

    if cfg.ratio_mode == "final":
        sample_count = int(round(primary_count * cfg.ratio / (1.0 - cfg.ratio)))
    else:
        sample_count = int(round(primary_count * cfg.ratio))
    return max(sample_count, 1)


def _load_sharegpt_records_from_source(source: Path) -> list[dict]:
    if source.is_file():
        return _load_sharegpt_records_from_file(source)

    if not source.exists():
        raise FileNotFoundError(
            f"Configured general_data source does not exist: {source}"
        )

    candidate_files = sorted(
        [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".parquet"}]
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"No JSON/JSONL/Parquet files found under general_data source: {source}"
        )

    loaded: list[dict] = []
    for path in candidate_files:
        loaded.extend(_load_sharegpt_records_from_file(path))
    return loaded


def _load_sharegpt_records_from_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                data = [data]
        return [_normalize_sharegpt_record(record, path) for record in data if _looks_like_sharegpt_record(record)]

    if suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if _looks_like_sharegpt_record(record):
                    records.append(_normalize_sharegpt_record(record, path))
        return records

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        records = []
        for record in dataset:
            record = dict(record)
            if _looks_like_sharegpt_record(record):
                records.append(_normalize_sharegpt_record(record, path))
        return records

    return []


def _looks_like_sharegpt_record(record: dict) -> bool:
    return isinstance(record, dict) and isinstance(record.get("conversations"), list)


def _normalize_sharegpt_record(record: dict, source_path: Path) -> dict:
    normalized = {
        "conversations": record["conversations"],
        "images": _normalize_images(record.get("images") or record.get("image") or []),
    }
    if "id" in record:
        normalized["id"] = record["id"]
    else:
        normalized["id"] = f"general::{source_path.name}::{len(normalized['images'])}::{hash(str(record.get('conversations')))}"
    if "category" in record:
        normalized["category"] = record["category"]
    if "sub_category" in record:
        normalized["sub_category"] = record["sub_category"]
    return normalized


def _normalize_images(images: object) -> list[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        return [str(x) for x in images if x]
    return []


def _format_cot(raw_cot: str, fmt: str) -> str:
    if fmt == "structured":
        return (
            "<safety_analysis>\n"
            f"{raw_cot.strip()}\n"
            "</safety_analysis>"
        )
    return raw_cot.strip()


def save_llamafactory_dataset(
    records: list[dict],
    output_path: Path,
    image_root: Optional[Path] = None,
    use_cot: bool = True,
    cot_format: str = "structured",
    general_data_cfg: Optional[GeneralDataConfig] = None,
) -> Path:
    """Convert and write to JSON file for LLaMA-Factory."""
    lf_records = build_mixed_llamafactory_dataset(
        records,
        general_data_cfg=general_data_cfg,
        image_root=image_root,
        use_cot=use_cot,
        cot_format=cot_format,
    )

    if general_data_cfg is not None:
        primary_count = len(to_llamafactory_format(records, image_root, use_cot, cot_format))
        target_general = resolve_general_sample_count(primary_count, general_data_cfg)
        base_general = lf_records[primary_count:]
        if not base_general:
            raise ValueError("General-data mix requested but zero general records were loaded.")
        if len(base_general) < target_general:
            raise ValueError(
                f"Requested {target_general} general samples, but only {len(base_general)} are available from configured sources."
            )
        lf_records = lf_records[:primary_count] + base_general[:target_general]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(lf_records, f, ensure_ascii=False, indent=2)
    return output_path


def register_in_dataset_info(
    lf_root: Path,
    dataset_name: str,
    data_file: Path,
) -> None:
    """
    Register dataset in LLaMA-Factory/data/dataset_info.json.

    Adds entry in sharegpt multi-image format:
    {
      "dataset_name": {
        "file_name": "/abs/path/train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
          "role_tag": "from", "content_tag": "value",
          "user_tag": "human", "assistant_tag": "gpt"
        }
      }
    }
    """
    info_path = lf_root / "data" / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    info[dataset_name] = {
        "file_name": str(data_file.resolve()),
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    with open(info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ── MIS Evaluation Format ───────────────────────────────────────────────────


def to_mis_eval_format(records: list[dict]) -> list[dict]:
    """
    Convert HarnessDataset records to MIS inference input format.

    Output per record:
    {
      "id": int,
      "question": str,
      "image_path1": str,
      "image_path2": str,
      "category": str,
      "sub_category": str,
    }
    """
    return [
        {
            "id": r.get("id"),
            "question": r.get("question", ""),
            "image_path1": r.get("image_path1", ""),
            "image_path2": r.get("image_path2", ""),
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
        }
        for r in records
    ]


def from_mis_test_format(
    mis_json: Path,
    image_dir: Path,
) -> list[dict]:
    """
    Load MIS test JSON (mis_easy/hard/real.json) into our unified record format.

    MIS test fields: category, sub_category, question, image_path1, image_path2, id
    Returns records compatible with HarnessDataset iteration output.
    """
    with open(mis_json) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())

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
            "vlm_score": r.get("vlm_score", -1),
            "cot_response": "",
        })
    return records


# ── Inference Output / Eval Input ──────────────────────────────────────────


def inference_to_eval_input(inference_jsonl: Path) -> list[dict]:
    """
    Read inference output JSONL and return list ready for GPT-4o evaluator.

    Inference JSONL fields: id, question, response, image_path1, image_path2, category
    """
    records = []
    with open(inference_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_inference_jsonl(records: list[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path
