"""LlamaFactory-backed async inference for multi-image VLM evaluation."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from harness.gpu.allocator import InferPlan


class LFInferenceBackend:
    """ChatModel-based async batch inference for multi-image VLM evaluation."""

    def __init__(
        self,
        model_path: str,
        template: str,
        infer_plan: InferPlan,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        max_model_len: int = 8192,
        concurrency: int = 16,
        trust_remote_code: bool = True,
        extra_chat_args: Optional[dict] = None,
    ) -> None:
        self.model_path = model_path
        self.template = template
        self.adapter_path = adapter_path
        self.infer_plan = infer_plan
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.concurrency = concurrency
        self.trust_remote_code = trust_remote_code
        self.extra_chat_args = extra_chat_args or {}

        self._chat = None
        self._sem: Optional[asyncio.Semaphore] = None

    def load(self) -> None:
        """Instantiate ChatModel. Call this before generate_batch()."""
        if self._chat is not None:
            return

        if self.infer_plan.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.infer_plan.gpu_ids))

        # LlamaFactory's vLLM check rejects newer local installs in this env.
        # The harness opts into the documented bypass so inference can continue.
        os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

        from llamafactory.chat import ChatModel

        args = {
            "model_name_or_path": self.model_path,
            "template": self.template,
            "infer_backend": "vllm",
            "trust_remote_code": self.trust_remote_code,
            "vllm_gpu_util": self.infer_plan.gpu_memory_utilization,
            "vllm_maxlen": self.max_model_len,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        if self.adapter_path:
            args["adapter_name_or_path"] = self.adapter_path
        args.update(self.extra_chat_args)

        try:
            self._chat = ChatModel(args)
        except ImportError as e:
            if "vllm>=0.4.3,<=0.11.0" not in str(e):
                raise
            raise ImportError(
                "LlamaFactory rejected the installed vLLM version even after "
                "DISABLE_VERSION_CHECK=1 was set. Please verify the local vLLM "
                "installation is otherwise compatible."
            ) from e

    def unload(self) -> None:
        """Free GPU memory."""
        import gc

        try:
            import torch  # type: ignore
        except Exception:
            torch = None

        self._chat = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_batch(self, records: list[dict]) -> list[dict]:
        """Run inference on a batch and return records with a response field."""
        if self._chat is None:
            self.load()
        return asyncio.run(self._generate_async(records))

    async def _generate_async(self, records: list[dict]) -> list[dict]:
        self._sem = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(self._one(r) for r in records))
        return list(results)

    async def _one(self, record: dict) -> dict:
        assert self._sem is not None
        async with self._sem:
            question = record.get("question", "")
            img1 = record.get("image_path1") or ""
            img2 = record.get("image_path2") or ""
            images = [p for p in (img1, img2) if p]

            placeholders = "".join("<image>\n" for _ in images)
            user_content = f"{placeholders}{question}"
            messages = [{"role": "user", "content": user_content}]

            try:
                resp = await self._chat.achat(messages, images=images if images else None)
                response_text = resp[0].response_text if resp else ""
            except Exception as e:
                response_text = f"[INFERENCE_ERROR] {type(e).__name__}: {e}"

            return {
                "id": record.get("id"),
                "question": question,
                "response": response_text,
                "image_path1": img1,
                "image_path2": img2,
                "category": record.get("category", ""),
                "sub_category": record.get("sub_category", ""),
                "img_source": record.get("img_source", ""),
                "harm_type": record.get("harm_type", ""),
                "img_source_type": record.get("img_source_type", ""),
                "benchmark": record.get("benchmark", ""),
            }
