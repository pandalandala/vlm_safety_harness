"""
Format converters between DREAMS/MIS formats and downstream consumers.

  our_format  →  llamafactory_sharegpt  (training)
  our_format  →  mis_eval_format        (inference / evaluation)
  mis_eval_format  →  our_format        (loading MIS test sets as HarnessDataset)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


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
) -> Path:
    """Convert and write to JSON file for LLaMA-Factory."""
    lf_records = to_llamafactory_format(records, image_root, use_cot, cot_format)
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
