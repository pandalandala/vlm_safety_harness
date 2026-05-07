"""
Pydantic configuration models for the DREAMS harness.
All YAML configs are validated against these schemas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str                          # Short identifier, e.g. "internvl2_5_8b"
    hf_path: str                       # HuggingFace model path or local path
    architecture: Literal["internvl", "qwen2vl", "llava", "phi", "idefics", "minicpm"]
    size_b: float                      # Model size in billions (used for GPU planning)
    trust_remote_code: bool = True

    # vLLM overrides (per-architecture defaults in inference/model_configs.py)
    max_model_len: int = 8192
    max_dynamic_patch: Optional[int] = None   # InternVL-specific
    min_pixels: Optional[int] = None          # Qwen2-VL-specific
    max_pixels: Optional[int] = None          # Qwen2-VL-specific


class DatasetConfig(BaseModel):
    name: str
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    image_root: Optional[Path] = None
    categories: Optional[list[str]] = None      # None = all categories
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # A-experiment controls (do NOT set for main experiments)
    text_only_mode: bool = False                 # A1: replace images with black frames
    filter_img_source: Optional[str] = None      # A3: "AI-generated" | "Web-retrieved"


class TrainingConfig(BaseModel):
    enabled: bool = True
    backend: Literal["llamafactory"] = "llamafactory"
    finetuning_type: Literal["full", "lora", "qlora"] = "full"
    max_gpus: int = 8                            # GPUAllocator caps at min(available, max_gpus)
    use_cot_labels: bool = True
    cot_format: Literal["free_text", "structured"] = "structured"
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = False

    # Training hyperparams (sensible defaults for 7-8B full fine-tune)
    learning_rate: float = 1e-5
    num_train_epochs: float = 3.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    cutoff_len: int = 2048
    save_steps: int = 500
    logging_steps: int = 10

    output_dir: Optional[Path] = None           # Auto-generated if None
    resume_from_checkpoint: Optional[Path] = None


class InferenceConfig(BaseModel):
    backend: Literal["vllm", "hf"] = "vllm"
    benchmarks: list[str] = Field(
        default=["mis_easy", "mis_hard", "mis_real"],
        description=(
            "Supported: mis_easy, mis_hard, mis_real, "
            "figstep, mssbench_safe, mssbench_unsafe, our_test, "
            "probe_text_only, probe_relation_types"
        ),
    )
    batch_size: int = 32
    temperature: float = 0.0
    max_tokens: int = 1024


class EvalConfig(BaseModel):
    model: Literal["gpt-4o", "gpt-4o-mini", "llama_guard"] = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    max_concurrent_requests: int = 20
    max_tokens: int = 128
    temperature: float = 0.0
    compute_per_category: bool = True
    output_format: list[Literal["json", "latex", "markdown"]] = ["json", "latex"]


class TrackingConfig(BaseModel):
    backend: Literal["wandb", "tensorboard", "none"] = "none"
    project: str = "dreams_vlm_safety"
    entity: Optional[str] = None
    run_name: Optional[str] = None


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []
    group: Literal["prelim", "main", "ablation"] = "main"

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
