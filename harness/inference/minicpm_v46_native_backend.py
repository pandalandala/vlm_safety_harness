"""Native Transformers backend for MiniCPM-V-4.6 inference."""

from __future__ import annotations

import gc
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration


class MiniCPMV46NativeBackend:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._processor = None
        self._model = None

    def load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def unload(self) -> None:
        self._processor = None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restart(self, safe_mode: Optional[bool] = None, concurrency: Optional[int] = None) -> None:
        del safe_mode, concurrency
        self.unload()
        self.load()

    def generate_batch(self, records: list[dict]) -> list[dict]:
        if self._model is None or self._processor is None:
            self.load()

        assert self._model is not None
        assert self._processor is not None

        outputs = []
        for record in records:
            try:
                response_text = self._infer_one(record)
            except Exception as exc:
                response_text = f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
            outputs.append(self.build_result(record, response_text))
        return outputs

    def _infer_one(self, record: dict) -> str:
        assert self._model is not None
        assert self._processor is not None

        images = []
        for path in (record.get("image_path1"), record.get("image_path2")):
            if path:
                images.append(Image.open(path).convert("RGB"))

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": record.get("question", "")})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(images=images or None, text=text, return_tensors="pt")
        # Current upstream processor may emit this field even though model.generate
        # rejects it. It is not required for correct generation here.
        inputs.pop("mm_token_type_ids", None)
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        response_ids = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

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
