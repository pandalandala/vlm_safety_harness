"""
DEPRECATED — kept for backward compatibility with prelim A experiments.

Main experiments + ablations use `harness.inference.lf_backend.LFInferenceBackend`,
which delegates per-arch prompt formatting to LlamaFactory's mm_plugin templates.

This module's per-arch `_prompt_<arch>` methods are no longer the canonical
prompt builders. Do NOT add new architectures here — extend
`harness/training/trainer.py:ARCH_TO_TEMPLATE` instead.

Per-architecture prompt construction adapted from
/mnt/hdd/xuran/MIS/evaluation/inference_vllm.py (validated parameters).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

from PIL import Image

from harness.inference.model_configs import get_model_config, VLLMModelConfig

warnings.warn(
    "harness.inference.vllm_backend.VLLMBackend is deprecated. "
    "Use harness.inference.lf_backend.LFInferenceBackend for new code.",
    DeprecationWarning,
    stacklevel=2,
)


def _open_image(path: str) -> Optional[Image.Image]:
    if not path:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


class VLLMBackend:
    def __init__(
        self,
        model_path: str,
        architecture: str,
        tensor_parallel_size: int = 1,
        gpu_ids: Optional[list[int]] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model_path = model_path
        self.architecture = architecture
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_ids = gpu_ids
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._cfg: VLLMModelConfig = get_model_config(architecture, max_model_len)
        self._cfg = type(self._cfg)(**{
            **self._cfg.__dict__,
            "gpu_memory_utilization": gpu_memory_utilization,
        })

        self._llm = None
        self._processor = None

    def load(self) -> None:
        """Initialize vLLM LLM instance (lazy, call before inference)."""
        from vllm import LLM
        from transformers import AutoTokenizer, AutoProcessor

        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        vllm_kwargs = self._cfg.to_vllm_kwargs(self.tensor_parallel_size)
        self._llm = LLM(model=self.model_path, **vllm_kwargs)

        # Load processor/tokenizer for prompt building
        if self.architecture == "internvl":
            self._processor = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        elif self.architecture == "qwen2vl":
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
        elif self.architecture == "llava":
            # LLaVA-OV is Qwen2-based; use chat template for correct format
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_path)
            except Exception:
                self._processor = None
        # phi, idefics: no processor needed for prompt building

    def generate_batch(self, records: list[dict]) -> list[dict]:
        """Run inference on a batch of records. Returns records with 'response' field."""
        if self._llm is None:
            self.load()

        from vllm import SamplingParams

        inputs = [self._build_prompt(r) for r in records]
        stop_token_ids = self._get_stop_token_ids()
        sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

        outputs = self._llm.generate(inputs, sampling)

        results = []
        for r, out in zip(records, outputs):
            response = out.outputs[0].text.strip()
            results.append({
                "id": r.get("id"),
                "question": r.get("question", ""),
                "response": response,
                "image_path1": r.get("image_path1", ""),
                "image_path2": r.get("image_path2", ""),
                "category": r.get("category", ""),
                "sub_category": r.get("sub_category", ""),
                "img_source": r.get("img_source", ""),
                "benchmark": r.get("benchmark", ""),
            })
        return results

    def _build_prompt(self, r: dict) -> dict:
        question = r.get("question", "")
        img1_path = r.get("image_path1", "")
        img2_path = r.get("image_path2", "")
        arch = self.architecture

        if arch == "qwen2vl":
            return self._prompt_qwen2vl(question, img1_path, img2_path)
        elif arch == "internvl":
            return self._prompt_internvl(question, img1_path, img2_path)
        elif arch == "phi":
            return self._prompt_phi(question, img1_path, img2_path)
        elif arch == "idefics":
            return self._prompt_idefics(question, img1_path, img2_path)
        elif arch == "llava":
            return self._prompt_llava(question, img1_path, img2_path)
        elif arch == "minicpm":
            return self._prompt_minicpm(question, img1_path, img2_path)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _prompt_qwen2vl(self, question: str, img1: str, img2: str) -> dict:
        from qwen_vl_utils import process_vision_info
        images = [img for img in (img1, img2) if img]
        placeholders = [{"type": "image", "image": p} for p in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        return {"prompt": prompt, "multi_modal_data": {"image": image_inputs}}

    def _prompt_internvl(self, question: str, img1: str, img2: str) -> dict:
        imgs = [_open_image(p) for p in (img1, img2) if p]
        placeholders = "\n".join(f"Image-{i}: <image>" for i, _ in enumerate(imgs, 1))
        messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

    def _prompt_phi(self, question: str, img1: str, img2: str) -> dict:
        imgs = [_open_image(p) for p in (img1, img2) if p]
        placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(imgs, 1))
        prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
        return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

    def _prompt_idefics(self, question: str, img1: str, img2: str) -> dict:
        imgs = [_open_image(p) for p in (img1, img2) if p]
        placeholders = "\n".join(f"Image-{i}: <image>" for i, _ in enumerate(imgs, 1))
        prompt = f"<|begin_of_text|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"
        return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

    def _prompt_llava(self, question: str, img1: str, img2: str) -> dict:
        imgs = [_open_image(p) for p in (img1, img2) if p]
        if self._processor is not None and hasattr(self._processor, "apply_chat_template"):
            # LLaVA-OneVision (Qwen2-based): use proper chat template
            content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": question}]
            messages = [{"role": "user", "content": content}]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            placeholders = "\n".join(f"<image>" for _ in imgs)
            prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
        return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

    def _prompt_minicpm(self, question: str, img1: str, img2: str) -> dict:
        imgs = [_open_image(p) for p in (img1, img2) if p]
        placeholders = "(<image>./</image>)\n" * len(imgs)
        prompt = f"<|begin_of_text|><|user|>\n{placeholders}{question}<|end|>\n<|assistant|>\n"
        return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

    def _get_stop_token_ids(self) -> Optional[list[int]]:
        if self.architecture == "internvl" and self._processor:
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            ids = []
            for tok in stop_tokens:
                try:
                    ids.append(self._processor.convert_tokens_to_ids(tok))
                except Exception:
                    pass
            return ids or None
        return None

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        import torch
        self._llm = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
