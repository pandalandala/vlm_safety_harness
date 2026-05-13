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

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CF_ID_OFFSET = 1_000_000


def collect_benign_pool(pool_dir: Path) -> list[Path]:
    """Recursively collect image files from `pool_dir`."""
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        raise FileNotFoundError(f"benign pool dir does not exist: {pool_dir}")
    images = sorted(p for p in pool_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not images:
        raise ValueError(f"no images found under {pool_dir}")
    return images


def resolve_image_path(image_path: str, dataset_root: Optional[Path] = None) -> Path:
    """Resolve an image path stored in dataset json into a concrete file path."""
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    if dataset_root is not None:
        candidate = Path(dataset_root) / image_path
        if candidate.exists():
            return candidate
    if p.exists():
        return p
    raise FileNotFoundError(f"could not resolve image path: {image_path}")


class SemanticBenignRetriever:
    """Retrieve semantically similar benign images using image embeddings."""

    def __init__(
        self,
        benign_pool: Iterable[Path],
        model_name: str = "google/siglip2-so400m-patch14-384",
        cache_path: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        seed: int = 0,
        local_files_only: bool = True,
    ):
        self.benign_pool = [Path(p) for p in benign_pool]
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._rng = random.Random(seed)

        self._processor = None
        self._model = None
        self._pool_paths: list[Path] = []
        self._pool_embeddings: Optional[torch.Tensor] = None
        self._pool_embeddings_device: Optional[torch.Tensor] = None

        self._load_or_build_index()

    def retrieve(self, query_image_path: Path, top_k: int = 1) -> tuple[Path, float]:
        """Return a semantically similar benign image and its cosine score."""
        query_path = Path(query_image_path)
        if not query_path.exists():
            raise FileNotFoundError(f"query image does not exist: {query_path}")

        query_embedding = self._encode_images([query_path])
        if query_embedding.shape[0] != 1:
            raise RuntimeError(f"failed to encode query image: {query_path}")

        pool_embeddings = self._get_pool_embeddings_for_search()
        query_embedding = query_embedding.to(
            device=pool_embeddings.device,
            dtype=pool_embeddings.dtype,
        )

        similarities = torch.matmul(pool_embeddings, query_embedding[0])
        k = max(1, min(top_k, similarities.shape[0]))
        top_scores, top_indices = torch.topk(similarities, k=k)

        if k == 1:
            chosen = 0
        else:
            chosen = self._rng.randrange(k)

        best_idx = int(top_indices[chosen].item())
        best_score = float(top_scores[chosen].item())
        return self._pool_paths[best_idx], best_score

    def _get_model_components(self):
        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            ).to(self.device)
            self._model.eval()
        return self._processor, self._model

    def _load_or_build_index(self) -> None:
        pool_path_strings = [str(p) for p in self.benign_pool]
        if self.cache_path and self.cache_path.exists():
            cache = torch.load(self.cache_path, map_location="cpu")
            if (
                cache.get("model_name") == self.model_name
                and cache.get("paths") == pool_path_strings
            ):
                self._pool_paths = [Path(p) for p in cache["paths"]]
                self._pool_embeddings = cache["embeddings"]
                return

        self._pool_paths, self._pool_embeddings = self._build_pool_index(self.benign_pool)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": self.model_name,
                    "paths": [str(p) for p in self._pool_paths],
                    "embeddings": self._pool_embeddings,
                },
                self.cache_path,
            )

    def _build_pool_index(self, image_paths: list[Path]) -> tuple[list[Path], torch.Tensor]:
        encoded_batches: list[torch.Tensor] = []
        kept_paths: list[Path] = []

        total = len(image_paths)
        print(
            f"[semantic] building benign image index with {total} images "
            f"(model={self.model_name}, device={self.device}, batch_size={self.batch_size})"
        )

        for start in range(0, total, self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]
            batch_embeddings = self._encode_images(batch_paths)
            if batch_embeddings.shape[0] != len(batch_paths):
                raise RuntimeError("batch embedding count mismatch while building index")
            encoded_batches.append(batch_embeddings.cpu())
            kept_paths.extend(batch_paths)
            end = min(start + self.batch_size, total)
            if end % (self.batch_size * 20) == 0 or end == total:
                print(f"[semantic] indexed {end}/{total} benign images")

        if not encoded_batches:
            raise ValueError("no benign images could be encoded for semantic retrieval")
        embeddings = torch.cat(encoded_batches, dim=0).to(dtype=torch.float16)
        return kept_paths, embeddings

    def _encode_images(self, image_paths: list[Path]) -> torch.Tensor:
        processor, model = self._get_model_components()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if torch.is_tensor(v)
        }
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        features = features.to(dtype=torch.float32)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu()

    def _get_pool_embeddings_for_search(self) -> torch.Tensor:
        if self._pool_embeddings is None:
            raise RuntimeError("semantic benign pool index is not initialized")
        if self._pool_embeddings_device is None:
            target = self._pool_embeddings
            if self.device.startswith("cuda"):
                try:
                    target = target.to(self.device)
                except RuntimeError:
                    print("[semantic] failed to place pool embeddings on GPU; falling back to CPU")
                    target = target.float()
                    self.device = "cpu"
            else:
                target = target.float()
            self._pool_embeddings_device = target
        return self._pool_embeddings_device


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
        dataset_root: Optional[Path] = None,
        retrieval_method: str = "random",
        semantic_retriever: Optional[SemanticBenignRetriever] = None,
        semantic_top_k: int = 1,
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
        if retrieval_method not in {"random", "semantic"}:
            raise ValueError(f"unknown retrieval_method: {retrieval_method}")
        if retrieval_method == "semantic" and semantic_retriever is None:
            raise ValueError("semantic retrieval requires a semantic_retriever instance")

        out: list[dict] = []
        for r in records:
            orig_id = r["id"]
            cf_id = orig_id + _CF_ID_OFFSET
            if retrieval_method == "semantic":
                query_key = "image_path2" if self.swap_image_idx == 2 else "image_path1"
                query_path = resolve_image_path(r[query_key], dataset_root=dataset_root)
                benign_src, benign_score = semantic_retriever.retrieve(
                    query_path,
                    top_k=semantic_top_k,
                )
            else:
                query_path = None
                benign_score = None
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

            record = {
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
                "benign_selection_method": retrieval_method,
            }
            if query_path is not None:
                record["benign_query_image_path"] = str(query_path)
            if benign_score is not None:
                record["benign_selection_score"] = benign_score
            out.append(record)
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
