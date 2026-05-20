"""LlamaFactory-backed async inference for multi-image VLM evaluation."""

from __future__ import annotations

import asyncio
import gc
import os
from typing import Optional

from harness.gpu.allocator import InferPlan
from PIL import Image


class EngineCrashedError(RuntimeError):
    """Raised when the underlying vLLM engine dies and the batch should be retried."""


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
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        infer_backend: str = "vllm",
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
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.infer_backend = infer_backend
        self.extra_chat_args = extra_chat_args or {}
        self.safe_mode = False

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
        os.environ["MAX_CONCURRENT"] = str(max(self.concurrency, 1))

        from llamafactory.chat import ChatModel

        gpu_util = self.infer_plan.gpu_memory_utilization
        if self.safe_mode:
            gpu_util = min(gpu_util, 0.8)

        args = {
            "model_name_or_path": self.model_path,
            "template": self.template,
            "infer_backend": self.infer_backend,
            "trust_remote_code": self.trust_remote_code,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        if self.infer_backend == "vllm":
            args.update(
                {
                    "vllm_gpu_util": gpu_util,
                    "vllm_maxlen": self.max_model_len,
                    "vllm_enforce_eager": self.safe_mode,
                }
            )
        if self.adapter_path:
            args["adapter_name_or_path"] = self.adapter_path
        if self.image_min_pixels is not None:
            args["image_min_pixels"] = self.image_min_pixels
        if self.image_max_pixels is not None:
            args["image_max_pixels"] = self.image_max_pixels
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
        except Exception as e:
            if self.infer_backend == "vllm" and self._should_fallback_to_hf(e):
                self.infer_backend = "huggingface"
                args["infer_backend"] = self.infer_backend
                args.pop("vllm_gpu_util", None)
                args.pop("vllm_maxlen", None)
                args.pop("vllm_enforce_eager", None)
                self._chat = ChatModel(args)
                return
            raise

    def unload(self) -> None:
        """Free GPU memory."""
        try:
            import torch  # type: ignore
        except Exception:
            torch = None

        chat = self._chat
        self._chat = None
        self._sem = None

        if chat is not None:
            try:
                engine = getattr(chat, "engine", None)
                model = getattr(engine, "model", None)
                shutdown = getattr(model, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                pass

            try:
                loop = getattr(chat, "_loop", None)
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

            try:
                thread = getattr(chat, "_thread", None)
                if thread is not None and thread.is_alive():
                    thread.join(timeout=1.0)
            except Exception:
                pass

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restart(self, safe_mode: Optional[bool] = None, concurrency: Optional[int] = None) -> None:
        """Recreate the underlying ChatModel with optional safer settings."""
        self.unload()
        if safe_mode is not None:
            self.safe_mode = safe_mode
        if concurrency is not None:
            self.concurrency = concurrency
        self.load()

    def generate_batch(self, records: list[dict]) -> list[dict]:
        """Run inference on a batch and return records with a response field."""
        if self._chat is None:
            self.load()
        # vLLM's AsyncLLMEngine binds its background output handler to the event
        # loop active on first use. ChatModel runs that loop in its own daemon
        # thread (self._chat._loop). Using asyncio.run() here would create — and
        # then close — a throwaway loop per batch, tearing down the engine after
        # the first batch (every later batch then raises EngineDeadError). Submit
        # the work to ChatModel's persistent loop instead so the engine survives.
        future = asyncio.run_coroutine_threadsafe(
            self._generate_async(records), self._chat._loop
        )
        return future.result()

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
                if self._should_retry_with_pil(e, images):
                    try:
                        pil_images = [Image.open(path).convert("RGB") for path in images]
                        resp = await self._chat.achat(messages, images=pil_images)
                        response_text = resp[0].response_text if resp else ""
                        return self.build_result(record, response_text)
                    except Exception as retry_error:
                        if self._is_engine_crash(retry_error):
                            raise EngineCrashedError(
                                f"id={record.get('id')} {type(retry_error).__name__}: {retry_error}"
                            ) from retry_error
                        response_text = f"[INFERENCE_ERROR] {type(retry_error).__name__}: {retry_error}"
                        return self.build_result(record, response_text)
                if self._is_engine_crash(e):
                    raise EngineCrashedError(
                        f"id={record.get('id')} {type(e).__name__}: {e}"
                    ) from e
                response_text = f"[INFERENCE_ERROR] {type(e).__name__}: {e}"

            return self.build_result(record, response_text)

    @staticmethod
    def build_result(record: dict, response_text: str) -> dict:
        """Return a normalized output record."""
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

    @staticmethod
    def _is_engine_crash(exc: Exception) -> bool:
        name = type(exc).__name__
        if "EngineDead" in name:
            return True
        return "EngineCore encountered an issue" in str(exc)

    @staticmethod
    def _should_fallback_to_hf(exc: Exception) -> bool:
        text = str(exc)
        return (
            "not supported for now" in text
            or "try to use HuggingFace backend" in text
            or "LLaVAOneVision1_5_ForConditionalGeneration" in text
        )

    @staticmethod
    def _should_retry_with_pil(exc: Exception, images: list[str]) -> bool:
        return bool(images) and type(exc).__name__ == "TypeError" and "invalid data type 'str'" in str(exc)
