"""
Dynamic GPU resource detection and allocation.
Detects available GPUs at runtime and computes optimal training/inference plans.
"""
from __future__ import annotations

import os
import subprocess
import argparse
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def is_available(self) -> bool:
        """GPU is available if utilization < 20% and free memory > 90%."""
        mem_ratio = self.memory_used_mb / max(self.memory_total_mb, 1)
        return self.utilization_pct < 20 and mem_ratio < 0.10


@dataclass
class TrainPlan:
    gpu_ids: list[int]
    num_gpus: int
    per_device_batch: int
    grad_accum: int
    deepspeed_config: str  # "ds_z2_config" | "ds_z3_config"
    effective_batch: int

    def to_llamafactory_params(self) -> dict:
        return {
            "per_device_train_batch_size": self.per_device_batch,
            "gradient_accumulation_steps": self.grad_accum,
            "deepspeed": f"examples/deepspeed/{self.deepspeed_config}.json",
        }

    def cuda_visible_devices(self) -> str:
        return ",".join(map(str, self.gpu_ids))


@dataclass
class InferPlan:
    gpu_ids: list[int]
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.9

    def cuda_visible_devices(self) -> str:
        return ",".join(map(str, self.gpu_ids))


class GPUAllocator:
    @staticmethod
    def _visible_gpu_ids() -> set[int] | None:
        """Return CUDA-visible GPU ids if the environment restricts them."""
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None or raw.strip() == "":
            return None

        ids: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                # Non-integer entries (for example UUIDs) are not handled here.
                return None
        return ids

    def detect(self) -> list[GPUInfo]:
        """Query nvidia-smi for current GPU status."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        visible_ids = self._visible_gpu_ids()
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_used_mb=int(parts[2]),
                        memory_total_mb=int(parts[3]),
                        utilization_pct=int(parts[4]),
                    )
                )
            except (ValueError, IndexError):
                continue
        if visible_ids is None:
            return gpus
        return [g for g in gpus if g.index in visible_ids]

    def get_available(self) -> list[GPUInfo]:
        return [g for g in self.detect() if g.is_available]

    def plan_training(self, model_size_b: float) -> TrainPlan:
        """
        Compute optimal training plan based on model size and available GPUs.
        Target effective batch size ~ 16 tokens.
        """
        available = self.get_available()
        if not available:
            # Fallback: assume all GPUs available (user will set CUDA_VISIBLE_DEVICES)
            all_gpus = self.detect()
            available = all_gpus if all_gpus else [GPUInfo(0, "RTX A6000", 0, 49152, 0)]

        n = len(available)
        gpu_ids = [g.index for g in available]

        if model_size_b <= 4:
            # 4B: 1–2 GPUs, ZeRO-2
            use_n = min(n, 2)
            return TrainPlan(
                gpu_ids=gpu_ids[:use_n],
                num_gpus=use_n,
                per_device_batch=2,
                grad_accum=max(1, 8 // use_n),
                deepspeed_config="ds_z2_config",
                effective_batch=use_n * 2 * max(1, 8 // use_n),
            )
        elif model_size_b <= 9:
            # 7–8B: 2–4 GPUs, ZeRO-3
            use_n = min(n, 8)
            deepspeed_config = "ds_z3_offload_config" if use_n <= 2 else "ds_z3_config"
            return TrainPlan(
                gpu_ids=gpu_ids[:use_n],
                num_gpus=use_n,
                per_device_batch=1,
                grad_accum=max(1, 4),
                deepspeed_config=deepspeed_config,
                effective_batch=use_n * 1 * 4,
            )
        elif model_size_b <= 30:
            # 26B: 4–8 GPUs, ZeRO-3
            use_n = min(n, 8)
            return TrainPlan(
                gpu_ids=gpu_ids[:use_n],
                num_gpus=use_n,
                per_device_batch=1,
                grad_accum=max(1, 8 // use_n),
                deepspeed_config="ds_z3_config",
                effective_batch=use_n * 1 * max(1, 8 // use_n),
            )
        else:
            # 72B+: all GPUs, ZeRO-3 (inference only recommended)
            return TrainPlan(
                gpu_ids=gpu_ids,
                num_gpus=n,
                per_device_batch=1,
                grad_accum=8,
                deepspeed_config="ds_z3_config",
                effective_batch=n * 8,
            )

    def plan_inference(self, model_size_b: float) -> InferPlan:
        """Compute optimal inference (vLLM) plan."""
        available = self.get_available()
        if not available:
            all_gpus = self.detect()
            available = all_gpus if all_gpus else [GPUInfo(0, "RTX A6000", 0, 49152, 0)]

        gpu_ids = [g.index for g in available]

        if model_size_b <= 9:
            # Single GPU for ≤9B
            return InferPlan(gpu_ids=gpu_ids[:1], tensor_parallel_size=1)
        elif model_size_b <= 30:
            use_n = min(len(available), 4)
            return InferPlan(gpu_ids=gpu_ids[:use_n], tensor_parallel_size=use_n)
        else:
            use_n = min(len(available), 8)
            return InferPlan(gpu_ids=gpu_ids[:use_n], tensor_parallel_size=use_n)

    def plan_cot_generation(self) -> InferPlan:
        """Use remaining GPUs for CoT label generation with the largest available model."""
        return self.plan_inference(model_size_b=72)

    def status_report(self, plan_for: float | None = None) -> str:
        lines = []
        all_gpus = self.detect()
        available = [g for g in all_gpus if g.is_available]

        lines.append("[GPU Status]")
        for g in all_gpus:
            avail_marker = "← available" if g.is_available else "← busy"
            lines.append(
                f"  GPU{g.index}: {g.name} | {g.memory_used_mb}MB/{g.memory_total_mb}MB"
                f" | util={g.utilization_pct}%  {avail_marker}"
            )
        lines.append(f"\nAvailable GPUs: {[g.index for g in available]} ({len(available)} GPUs)")

        if plan_for is not None:
            train = self.plan_training(plan_for)
            infer = self.plan_inference(plan_for)
            lines.append(f"\n[Training Plan for {plan_for}B model]")
            lines.append(f"  gpu_ids: {train.gpu_ids}  ({train.num_gpus} GPUs)")
            lines.append(f"  deepspeed: {train.deepspeed_config}")
            lines.append(f"  per_device_batch: {train.per_device_batch}")
            lines.append(f"  grad_accum: {train.grad_accum}")
            lines.append(f"  effective_batch: {train.effective_batch}")
            lines.append(f"\n[Inference Plan for {plan_for}B model]")
            lines.append(f"  gpu_ids: {infer.gpu_ids}  (tensor_parallel={infer.tensor_parallel_size})")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU status and allocation planner")
    parser.add_argument("--status", action="store_true", help="Print GPU status")
    parser.add_argument("--plan-for", type=float, default=None,
                        help="Compute training/inference plan for model of this size (in B)")
    args = parser.parse_args()

    allocator = GPUAllocator()
    print(allocator.status_report(plan_for=args.plan_for))
