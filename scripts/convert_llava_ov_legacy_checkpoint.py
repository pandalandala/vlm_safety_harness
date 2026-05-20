#!/usr/bin/env python3
"""Convert legacy LLaVA-OneVision checkpoint key prefixes to the 1.5 remote-code layout.

This fixes checkpoints whose text tower is saved under:
  - model.layers.*
  - model.embed_tokens.*
  - model.norm.*
and whose vision tower is saved under:
  - visual.*

The newer LLaVA-OneVision-1.5 remote code expects:
  - model.language_model.layers.*
  - model.language_model.embed_tokens.*
  - model.language_model.norm.*
  - model.visual.*
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _rename_key(key: str) -> str:
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "model.language_model.layers.", 1)
    if key.startswith("model.embed_tokens."):
        return key.replace("model.embed_tokens.", "model.language_model.embed_tokens.", 1)
    if key.startswith("model.norm."):
        return key.replace("model.norm.", "model.language_model.norm.", 1)
    if key.startswith("visual."):
        return key.replace("visual.", "model.visual.", 1)
    if key.startswith("multi_modal_projector."):
        return key.replace("multi_modal_projector.", "model.visual.merger.", 1)
    return key


def _copy_metadata(src: Path, dst: Path) -> None:
    skip_names = {
        "model.safetensors",
        "model.safetensors.index.json",
    }
    skip_prefixes = (
        "model-",
        "pytorch_model-",
        "global_step",
    )
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        if any(child.name.startswith(prefix) for prefix in skip_prefixes):
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _patch_config(dst: Path) -> None:
    config_path = dst / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", {})

    if config.get("image_token_index") is None and text_config.get("image_token_id") is not None:
        config["image_token_index"] = text_config["image_token_id"]
    if config.get("video_token_index") is None and text_config.get("video_token_id") is not None:
        config["video_token_index"] = text_config["video_token_id"]
    if config.get("vision_feature_layer") is None:
        config["vision_feature_layer"] = -2
    if config.get("vision_feature_select_strategy") is None:
        config["vision_feature_select_strategy"] = "default"

    config_path.write_text(json.dumps(config, indent=2) + "\n")


def _convert_single_file(src_file: Path, dst_file: Path) -> list[str]:
    renamed_keys: list[str] = []
    renamed_tensors = {}
    with safe_open(str(src_file), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            new_key = _rename_key(key)
            renamed_tensors[new_key] = handle.get_tensor(key)
            renamed_keys.append(new_key)
    save_file(renamed_tensors, str(dst_file), metadata=metadata)
    return renamed_keys


def convert_checkpoint(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    _copy_metadata(src, dst)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        new_weight_map = {}
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_name in shard_files:
            converted_keys = _convert_single_file(src / shard_name, dst / shard_name)
            for key in converted_keys:
                new_weight_map[key] = shard_name
        new_index = dict(index)
        new_index["weight_map"] = new_weight_map
        (dst / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    elif (src / "model.safetensors").exists():
        _convert_single_file(src / "model.safetensors", dst / "model.safetensors")
    else:
        raise FileNotFoundError(f"No safetensors weights found in {src}")

    _patch_config(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Source checkpoint directory")
    parser.add_argument("--dst", required=True, type=Path, help="Output checkpoint directory")
    args = parser.parse_args()
    convert_checkpoint(args.src.resolve(), args.dst.resolve())


if __name__ == "__main__":
    main()
