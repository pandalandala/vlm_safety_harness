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

M4_INSTRUCT_SNAPSHOT_ROOT = Path(
    "/mnt2/xuran_hdd/.cache/huggingface/hub/"
    "datasets--lmms-lab--M4-Instruct-Data/snapshots/"
    "9078d03bb7442cef5a71771a29a2af9ec0f63134"
)


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

        if use_cot:
            response = _format_cot(r["cot_response"], cot_format) if r.get("cot_response") else ""
        else:
            response = (r.get("assistant_response") or "").strip()

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
            # Keep metadata type-stable across mixed sources so HF datasets
            # does not fail Arrow schema inference on int-vs-str ids.
            "id": str(r.get("id")) if r.get("id") is not None else None,
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

    target_count = cfg.max_samples
    compatible_records: list[dict] = []
    for source in cfg.sources:
        for record in _iter_compatible_sharegpt_records_from_source(Path(source)):
            compatible_records.append(record)
            if target_count is not None and len(compatible_records) >= target_count:
                break
        if target_count is not None and len(compatible_records) >= target_count:
            break

    if not compatible_records:
        raise ValueError("No general-data samples loaded from configured sources.")

    rng = random.Random(cfg.shuffle_seed)
    rng.shuffle(compatible_records)
    if target_count is not None:
        compatible_records = compatible_records[:target_count]
    return compatible_records


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


def _iter_compatible_sharegpt_records_from_source(source: Path):
    if source.is_file():
        for record in _iter_sharegpt_records_from_file(source):
            if _is_mm_record_compatible(record):
                yield record
        return

    if not source.exists():
        raise FileNotFoundError(
            f"Configured general_data source does not exist: {source}"
        )

    found_candidate = False
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".parquet"}:
            continue
        found_candidate = True
        for record in _iter_sharegpt_records_from_file(path):
            if _is_mm_record_compatible(record):
                yield record

    if not found_candidate:
        raise FileNotFoundError(
            f"No JSON/JSONL/Parquet files found under general_data source: {source}"
        )


def _iter_sharegpt_records_from_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                data = [data]
        for record in data:
            if _looks_like_sharegpt_record(record):
                yield _normalize_sharegpt_record(record, path)
        return

    if suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if _looks_like_sharegpt_record(record):
                    yield _normalize_sharegpt_record(record, path)
        return

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        for record in dataset:
            record = dict(record)
            if _looks_like_sharegpt_record(record):
                yield _normalize_sharegpt_record(record, path)
        return

    return


def _looks_like_sharegpt_record(record: dict) -> bool:
    return isinstance(record, dict) and isinstance(record.get("conversations"), list)


def _normalize_sharegpt_record(record: dict, source_path: Path) -> dict:
    normalized = {
        "conversations": record["conversations"],
        "images": _normalize_images(
            record.get("images") or record.get("image") or [],
            source_path=source_path,
        ),
    }
    if "id" in record:
        normalized["id"] = str(record["id"])
    else:
        normalized["id"] = f"general::{source_path.name}::{len(normalized['images'])}::{hash(str(record.get('conversations')))}"
    if "category" in record:
        normalized["category"] = record["category"]
    if "sub_category" in record:
        normalized["sub_category"] = record["sub_category"]
    return normalized


def _normalize_images(images: object, source_path: Path) -> list[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [_resolve_image_path(images, source_path)]
    if isinstance(images, list):
        return [_resolve_image_path(str(x), source_path) for x in images if x]
    return []


def _resolve_image_path(image: str, source_path: Path) -> str:
    path = Path(image)
    if path.is_absolute():
        return str(path)

    # M4-Instruct annotations use paths relative to the snapshot root. Handle
    # them first to avoid repeated ancestor traversal on a very large dataset.
    if source_path.name.startswith("m4_instruct"):
        candidate = M4_INSTRUCT_SNAPSHOT_ROOT / path
        if candidate.exists():
            return str(candidate)

    # Try to resolve relative media paths against the dataset file directory
    # first, then progressively against ancestor directories up to the snapshot
    # root. If no real file is found, keep the original relative string so later
    # compatibility filtering can drop the sample.
    for base in (source_path.parent, *source_path.parents):
        candidate = base / path
        if candidate.exists():
            return str(candidate)
    return image


def _is_mm_record_compatible(record: dict) -> bool:
    """
    Filter out ShareGPT samples that LLaMA-Factory's multimodal plugin will reject.

    In practice the mixed M4-Instruct cache contains some examples where the number
    of `<image>` placeholders in messages does not match the `images` field (for
    example video-only or malformed multimodal rows). Those samples crash tokenizer
    preprocessing during dataset.map(), so drop them early.
    """
    conversations = record.get("conversations")
    images = record.get("images") or []
    if not isinstance(conversations, list):
        return False

    image_token_count = 0
    for turn in conversations:
        if not isinstance(turn, dict):
            return False
        value = turn.get("value", "")
        if not isinstance(value, str):
            return False
        image_token_count += value.count("<image>")

    if image_token_count != len(images):
        return False

    for image in images:
        image_path = Path(image)
        if not image_path.is_absolute() or not image_path.exists():
            return False

    return True


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
