"""Native vLLM backend for DeepSeek-VL2 inference (inference-only, no training).

DeepSeek-VL2 has no LlamaFactory template, so it cannot go through the standard
LF inference path. vLLM (>=0.7) supports `DeepseekVLV2ForCausalLM` natively, so we
drive it directly here, mirroring the MiniCPMV46NativeBackend interface
(load / unload / restart / generate_batch / build_result).

Prompt format follows the DeepSeek-VL2 convention: one `<image>` placeholder per
image, then the question, wrapped in `<|User|>:` / `<|Assistant|>:` turns.
"""

from __future__ import annotations

import gc
import os
from typing import Optional

from PIL import Image

from harness.gpu.allocator import InferPlan


class DeepseekVL2NativeBackend:
    def __init__(
        self,
        model_path: str,
        infer_plan: Optional[InferPlan] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        max_model_len: int = 4096,
        max_images: int = 2,
    ) -> None:
        self.model_path = model_path
        self.infer_plan = infer_plan
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.max_images = max_images
        self.safe_mode = False
        self._llm = None
        self._sampling = None

    def load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams

        tp = self.infer_plan.tensor_parallel_size if self.infer_plan else 1
        if self.infer_plan and self.infer_plan.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.infer_plan.gpu_ids))

        gpu_util = self.infer_plan.gpu_memory_utilization if self.infer_plan else 0.9
        if self.safe_mode:
            gpu_util = min(gpu_util, 0.8)

        self._llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            enforce_eager=self.safe_mode,
            limit_mm_per_prompt={"image": self.max_images},
            # config.json may omit the architectures field; pin it for vLLM.
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        )
        self._sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def unload(self) -> None:
        self._llm = None
        self._sampling = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def restart(self, safe_mode: Optional[bool] = None, concurrency: Optional[int] = None) -> None:
        del concurrency
        self.unload()
        if safe_mode is not None:
            self.safe_mode = safe_mode
        self.load()

    @staticmethod
    def _build_prompt(question: str, n_images: int) -> str:
        placeholders = "".join("<image>\n" for _ in range(n_images))
        return f"<|User|>: {placeholders}{question}\n\n<|Assistant|>:"

    def generate_batch(self, records: list[dict]) -> list[dict]:
        if self._llm is None:
            self.load()
        assert self._llm is not None

        vllm_inputs: list[dict] = []
        valid_idx: list[int] = []
        outputs: list[Optional[dict]] = [None] * len(records)

        for i, record in enumerate(records):
            try:
                images = []
                for path in (record.get("image_path1"), record.get("image_path2")):
                    if path:
                        images.append(Image.open(path).convert("RGB"))
                prompt = self._build_prompt(record.get("question", ""), len(images))
                entry: dict = {"prompt": prompt}
                if images:
                    entry["multi_modal_data"] = {"image": images}
                vllm_inputs.append(entry)
                valid_idx.append(i)
            except Exception as exc:
                outputs[i] = self.build_result(
                    record, f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
                )

        if vllm_inputs:
            try:
                gen = self._llm.generate(vllm_inputs, self._sampling)
                for j, out in zip(valid_idx, gen):
                    text = out.outputs[0].text.strip() if out.outputs else ""
                    outputs[j] = self.build_result(records[j], text)
            except Exception as exc:
                if len(valid_idx) == 1:
                    j = valid_idx[0]
                    outputs[j] = self.build_result(
                        records[j], f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
                    )
                else:
                    # A single bad sample can poison a whole vLLM batch. Retry
                    # one-by-one here so recovery only targets the true failures.
                    for j, entry in zip(valid_idx, vllm_inputs):
                        if outputs[j] is not None:
                            continue
                        try:
                            single = self._llm.generate([entry], self._sampling)
                            text = single[0].outputs[0].text.strip() if single and single[0].outputs else ""
                            outputs[j] = self.build_result(records[j], text)
                        except Exception as single_exc:
                            outputs[j] = self.build_result(
                                records[j],
                                f"[INFERENCE_ERROR] {type(single_exc).__name__}: {single_exc}",
                            )

        return [o for o in outputs if o is not None]

    @staticmethod
    def build_result(record: dict, response_text: str) -> dict:
        return {
            "id": record.get("id"),
            "question": record.get("question", ""),
            "response": response_text,
            "image_path1": record.get("image_path1") or "",
            "image_path2": record.get("image_path2") or "",
            "category": record.get("category", ""),
            "sub_category": record.get("sub_category", ""),
            "img_source": record.get("img_source", ""),
            "harm_type": record.get("harm_type", ""),
            "img_source_type": record.get("img_source_type", ""),
            "benchmark": record.get("benchmark", ""),
        }
