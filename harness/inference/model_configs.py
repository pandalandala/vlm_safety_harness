"""
Per-architecture vLLM initialization parameters.
Sourced from /mnt/hdd/xuran/MIS/evaluation/inference_vllm.py (validated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VLLMModelConfig:
    trust_remote_code: bool = False
    max_model_len: int = 4096
    max_num_seqs: int = 5
    tensor_parallel_size: int = 1          # overridden by InferPlan at runtime
    limit_mm_per_prompt: dict = field(default_factory=lambda: {"image": 2})
    gpu_memory_utilization: float = 0.9
    mm_processor_kwargs: dict = field(default_factory=dict)
    dtype: Optional[str] = None            # None = auto (bfloat16)

    def to_vllm_kwargs(self, tp: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": tp,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.trust_remote_code:
            out["trust_remote_code"] = True
        if self.mm_processor_kwargs:
            out["mm_processor_kwargs"] = self.mm_processor_kwargs
        if self.dtype:
            out["dtype"] = self.dtype
        return out


# Architecture → default vLLM config
ARCH_CONFIGS: dict[str, VLLMModelConfig] = {
    "internvl": VLLMModelConfig(
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=5,
        mm_processor_kwargs={"max_dynamic_patch": 4},
        gpu_memory_utilization=0.9,
    ),
    "qwen2vl": VLLMModelConfig(
        trust_remote_code=False,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
    ),
    "llava": VLLMModelConfig(
        trust_remote_code=False,
        max_model_len=16384,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
    ),
    "phi": VLLMModelConfig(
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={"max_dynamic_patch": 4},
        gpu_memory_utilization=0.9,
    ),
    "idefics": VLLMModelConfig(
        trust_remote_code=False,
        max_model_len=8192,
        max_num_seqs=16,
        mm_processor_kwargs={"size": {"longest_edge": 4 * 364}},
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    ),
    "minicpm": VLLMModelConfig(
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
    ),
}


def get_model_config(architecture: str, max_model_len: Optional[int] = None) -> VLLMModelConfig:
    cfg = ARCH_CONFIGS.get(architecture)
    if cfg is None:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: {list(ARCH_CONFIGS)}")
    if max_model_len:
        from dataclasses import replace
        cfg = replace(cfg, max_model_len=max_model_len)
    return cfg
