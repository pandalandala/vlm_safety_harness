"""
HarnessTrainer: wraps LLaMA-Factory SFT for multi-architecture VLM fine-tuning.

Workflow:
  1. register_dataset() → writes to LF data/dataset_info.json
  2. build_llamafactory_yaml() → generates run-specific YAML from template
  3. run_training() → executes torchrun in mis_safety conda env
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from harness.config.schema import ExperimentConfig, TrainingConfig
from harness.gpu.allocator import TrainPlan


LLAMAFACTORY_ROOT = Path("/mnt/hdd/xuran/LlamaFactory")
CONDA_ENV = "mis_safety"

# Maps our architecture IDs to LLaMA-Factory template names.
# Verified against /mnt/hdd/xuran/LlamaFactory/src/llamafactory/data/template.py.
ARCH_TO_TEMPLATE = {
    "internvl":      "intern_vl",
    "qwen2vl":       "qwen2_vl",
    "qwen3_vl":      "qwen3_vl",
    "llava":         "llava_next",
    "kimi_vl":       "kimi_vl",
    "minicpm":       "minicpm_v",
    "minicpm_v_4_6": "minicpm_v_4_6",
    "minicpm_o":     "minicpm_o",
    "gemma_vlm":     "gemma4",
    "glm4v":         "glm4v",
    "glm4_5v":       "glm4_5v",
}


class HarnessTrainer:
    def __init__(self, lf_root: Path = LLAMAFACTORY_ROOT):
        self.lf_root = Path(lf_root)
        self._dataset_info_path = self.lf_root / "data" / "dataset_info.json"

    # ── Public API ────────────────────────────────────────────────────────

    def prepare_and_run(
        self,
        cfg: ExperimentConfig,
        data_file: Path,
        gpu_plan: TrainPlan,
        output_dir: Path,
        dry_run: bool = False,
    ) -> Path:
        """Register dataset, generate YAML, run training. Returns checkpoint dir."""
        dataset_name = self._dataset_name(cfg.name)
        self.register_dataset(dataset_name, data_file)
        yaml_path = self.build_llamafactory_yaml(cfg, dataset_name, gpu_plan, output_dir)

        if dry_run:
            print(f"[dry-run] Would run: torchrun on {yaml_path}")
            print(f"[dry-run] GPUs: {gpu_plan.cuda_visible_devices()}")
            return output_dir / "final_checkpoint"

        self.run_training(yaml_path, gpu_plan)
        return self._find_checkpoint(output_dir)

    def register_dataset(self, dataset_name: str, data_file: Path) -> None:
        """Add/update entry in LLaMA-Factory/data/dataset_info.json."""
        with open(self._dataset_info_path) as f:
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

        with open(self._dataset_info_path, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def build_llamafactory_yaml(
        self,
        cfg: ExperimentConfig,
        dataset_name: str,
        gpu_plan: TrainPlan,
        output_dir: Path,
    ) -> Path:
        """
        Generate a LLaMA-Factory YAML config from ExperimentConfig + GPUPlan.
        Based on the qwen2_5vl_full_sft.yaml template structure.
        """
        tc: TrainingConfig = cfg.training
        mc = cfg.model

        template = ARCH_TO_TEMPLATE.get(mc.architecture)
        if template is None:
            raise ValueError(f"Unknown architecture: {mc.architecture}")

        lf_params = gpu_plan.to_llamafactory_params()
        deepspeed_cfg = self.lf_root / "examples" / "deepspeed" / lf_params["deepspeed"]

        data = {
            # model
            "model_name_or_path": mc.hf_path,
            "trust_remote_code": mc.trust_remote_code,
            # method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": tc.finetuning_type,
            "freeze_vision_tower": tc.freeze_vision_tower,
            "freeze_multi_modal_projector": tc.freeze_multi_modal_projector,
            "freeze_language_model": False,
            "deepspeed": str(deepspeed_cfg),
            # dataset
            "dataset": dataset_name,
            "template": template,
            "cutoff_len": tc.cutoff_len,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "dataloader_num_workers": 4,
            # output
            "output_dir": str(output_dir),
            "logging_steps": tc.logging_steps,
            "save_steps": tc.save_steps,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": cfg.tracking.backend if cfg.tracking.backend != "none" else "none",
            # training
            "per_device_train_batch_size": lf_params["per_device_train_batch_size"],
            "gradient_accumulation_steps": lf_params["gradient_accumulation_steps"],
            "learning_rate": tc.learning_rate,
            "num_train_epochs": tc.num_train_epochs,
            "lr_scheduler_type": tc.lr_scheduler_type,
            "warmup_ratio": tc.warmup_ratio,
            "bf16": tc.bf16,
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": str(tc.resume_from_checkpoint) if tc.resume_from_checkpoint else None,
        }

        # Architecture-specific overrides
        if mc.architecture in ("qwen2vl", "qwen3_vl"):
            data["image_max_pixels"] = mc.max_pixels or 262144
        if mc.architecture == "internvl" and mc.max_dynamic_patch:
            data["max_dynamic_patch"] = mc.max_dynamic_patch

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        yaml_path = output_dir / "llamafactory_config.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return yaml_path

    def run_training(self, yaml_path: Path, gpu_plan: TrainPlan) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_plan.cuda_visible_devices()

        cmd = [
            "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
            "torchrun",
            f"--nproc_per_node={gpu_plan.num_gpus}",
            "-m", "llamafactory.launcher",
            str(yaml_path),
        ]

        subprocess.run(cmd, env=env, check=True, cwd=str(self.lf_root))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _dataset_name(experiment_name: str) -> str:
        return f"dreams_{experiment_name}".replace("-", "_").replace("/", "_")

    @staticmethod
    def _find_checkpoint(output_dir: Path) -> Path:
        """Return latest checkpoint dir or output_dir itself."""
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        return checkpoints[-1] if checkpoints else output_dir
