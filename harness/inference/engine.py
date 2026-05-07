"""
InferenceEngine: unified entry for running inference across all benchmarks.

Orchestrates:
  - Benchmark loading
  - VLLMBackend initialization with InferPlan GPU settings
  - Batch inference with progress tracking
  - JSONL output per benchmark
  - Resume support
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from harness.config.schema import ExperimentConfig, InferenceConfig
from harness.gpu.allocator import InferPlan
from harness.inference.vllm_backend import VLLMBackend

# Benchmark name → loader function
BENCHMARK_REGISTRY: dict[str, str] = {
    "mis_easy":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    "mis_hard":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    "mis_real":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    "mssbench_safe":   "harness.data.benchmarks.mssbench.MSSBenchmark",
    "mssbench_unsafe": "harness.data.benchmarks.mssbench.MSSBenchmark",
    "figstep":    "harness.data.benchmarks.figstep.FigStepBenchmark",
    "our_test":   "_our_test",
    "probe_text_only":      "_probe",
    "probe_text_only_hard": "_probe",
    "probe_relation_types": "_probe",
}


class InferenceEngine:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model_path: str,           # trained checkpoint or base model
        infer_plan: InferPlan,
        output_dir: Path,
        batch_size: int = 32,
    ):
        self.cfg = cfg
        self.model_path = model_path
        self.infer_plan = infer_plan
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        self._backend: Optional[VLLMBackend] = None

    def run_all_benchmarks(
        self,
        benchmarks: Optional[list[str]] = None,
        resume: bool = True,
        max_samples: Optional[int] = None,
    ) -> dict[str, Path]:
        """
        Run inference on all configured benchmarks.
        Returns dict of {benchmark_name: output_jsonl_path}.
        """
        bmarks = benchmarks or self.cfg.inference.benchmarks
        results: dict[str, Path] = {}

        self._init_backend()
        try:
            for bname in bmarks:
                out_path = self.output_dir / "responses" / f"{bname}.jsonl"
                if resume and out_path.exists():
                    print(f"[skip] {bname}: output exists at {out_path}")
                    results[bname] = out_path
                    continue

                records = self._load_benchmark(bname, max_samples)
                print(f"[infer] {bname}: {len(records)} samples")
                out_path = self._run_benchmark(bname, records, out_path)
                results[bname] = out_path
        finally:
            if self._backend:
                self._backend.unload()

        return results

    def _init_backend(self) -> None:
        mc = self.cfg.model
        ic: InferenceConfig = self.cfg.inference
        self._backend = VLLMBackend(
            model_path=self.model_path,
            architecture=mc.architecture,
            tensor_parallel_size=self.infer_plan.tensor_parallel_size,
            gpu_ids=self.infer_plan.gpu_ids,
            max_model_len=mc.max_model_len,
            gpu_memory_utilization=self.infer_plan.gpu_memory_utilization,
            temperature=ic.temperature,
            max_tokens=ic.max_tokens,
        )
        self._backend.load()

    def _run_benchmark(self, name: str, records: list[dict], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: find already-processed IDs
        done_ids: set = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line.strip())["id"])
                    except Exception:
                        pass

        pending = [r for r in records if r.get("id") not in done_ids]
        if not pending:
            return out_path

        with open(out_path, "a") as f:
            for i in range(0, len(pending), self.batch_size):
                batch = pending[i: i + self.batch_size]
                outputs = self._backend.generate_batch(batch)
                for out in outputs:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{name}] {min(i + self.batch_size, len(pending))}/{len(pending)}")

        return out_path

    def _load_benchmark(self, name: str, max_samples: Optional[int]) -> list[dict]:
        from harness.config.schema import DatasetConfig

        ds_cfg: DatasetConfig = self.cfg.dataset

        if name in ("mis_easy", "mis_hard", "mis_real"):
            from harness.data.benchmarks.mis_benchmark import MISBenchmark
            return MISBenchmark(subset=name, max_samples=max_samples).load()

        elif name in ("mssbench_safe", "mssbench_unsafe"):
            from harness.data.benchmarks.mssbench import MSSBenchmark
            if ds_cfg.test_path is None:
                raise ValueError("dataset.test_path must point to MSSBench data for mssbench benchmarks")
            split = "safe" if name == "mssbench_safe" else "unsafe"
            return MSSBenchmark(
                data_path=ds_cfg.test_path,
                split=split,
                max_samples=max_samples,
            ).load()

        elif name == "figstep":
            from harness.data.benchmarks.figstep import FigStepBenchmark
            if ds_cfg.test_path is None:
                raise ValueError("dataset.test_path must point to FigStep data for figstep benchmark")
            return FigStepBenchmark(
                data_path=ds_cfg.test_path,
                max_samples=max_samples,
            ).load()

        elif name == "our_test":
            from harness.data.dataset import HarnessDataset
            ds = HarnessDataset.from_config(ds_cfg, mode="eval")
            return ds.to_eval_records()[:max_samples] if max_samples else ds.to_eval_records()

        elif name in ("probe_text_only", "probe_text_only_hard", "probe_relation_types"):
            # Probes live at results/prelim/probes/; output_dir is results/prelim/<exp>/<ts>/
            probe_path = self.output_dir.parent.parent / "probes" / f"{name}.json"
            if not probe_path.exists():
                raise FileNotFoundError(f"Probe file not found: {probe_path}. Run ProbeBuilder first.")
            with open(probe_path) as f:
                records = json.load(f)
            return records[:max_samples] if max_samples else records

        else:
            raise ValueError(f"Unknown benchmark: {name}")
