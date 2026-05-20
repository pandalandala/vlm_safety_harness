"""
InferenceEngine: unified entry for running inference across all benchmarks.

Orchestrates:
  - Benchmark loading
  - LFInferenceBackend initialization with InferPlan GPU settings
  - Batch inference (async via ChatModel.achat)
  - JSONL output per benchmark
  - Resume support
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from harness.config.schema import ExperimentConfig, InferenceConfig
from harness.gpu.allocator import InferPlan
from harness.inference.lf_backend import EngineCrashedError, LFInferenceBackend
from harness.training.trainer import ARCH_TO_TEMPLATE

# Benchmark name → loader spec.
# Strings prefixed with "_" are special-cased in `_load_benchmark`.
BENCHMARK_REGISTRY: dict[str, str] = {
    # MIS test sets
    "mis_easy":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    "mis_hard":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    "mis_real":   "harness.data.benchmarks.mis_benchmark.MISBenchmark",
    # MSSBench
    "mssbench_safe":   "harness.data.benchmarks.mssbench.MSSBenchmark",
    "mssbench_unsafe": "harness.data.benchmarks.mssbench.MSSBenchmark",
    "mss":             "harness.data.benchmarks.mssbench.MSSBenchmark",
    # Existing safety bench
    "figstep":    "harness.data.benchmarks.figstep.FigStepBenchmark",
    # Phase 4 — new safety
    "advbench":   "harness.data.benchmarks.advbench.AdvBenchBenchmark",
    "safebench":  "harness.data.benchmarks.safebench.SafeBenchBenchmark",
    "mm_safety":  "harness.data.benchmarks.mm_safety.MMSafetyBenchmark",
    "jailbreakv": "harness.data.benchmarks.jailbreakv.JailbreakVBenchmark",
    "siuo":       "harness.data.benchmarks.siuo.SIUOBenchmark",
    # Phase 4 — capability
    "mmstar":     "harness.data.benchmarks.mmstar.MMStarBenchmark",
    "mmmu":       "harness.data.benchmarks.mmmu.MMMUBenchmark",
    "muirbench":  "harness.data.benchmarks.muirbench.MuirBenchBenchmark",
    "blink":      "harness.data.benchmarks.blink.BLINKBenchmark",
    "mmt":        "harness.data.benchmarks.mmt.MMTBenchmark",
    # DREAMS test (with E1/E2 slicing)
    "our_test":   "_our_test",
    # A-experiment probes (kept for backward compat)
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

        self._backend = None
        self._safe_mode_enabled = False

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

        template = ARCH_TO_TEMPLATE.get(mc.architecture)
        if template is None:
            raise ValueError(
                f"No LF template registered for architecture '{mc.architecture}'. "
                f"Update ARCH_TO_TEMPLATE in harness/training/trainer.py."
            )

        if mc.architecture == "minicpm_v_4_6":
            from harness.inference.minicpm_v46_native_backend import MiniCPMV46NativeBackend

            self._backend = MiniCPMV46NativeBackend(
                model_path=self.model_path,
                max_new_tokens=ic.max_tokens,
                temperature=ic.temperature,
            )
            self._backend.load()
            return

        self._backend = LFInferenceBackend(
            model_path=self.model_path,
            template=template,
            infer_plan=self.infer_plan,
            max_new_tokens=ic.max_tokens,
            temperature=ic.temperature,
            max_model_len=mc.max_model_len,
            concurrency=ic.concurrency,
            trust_remote_code=mc.trust_remote_code,
            image_min_pixels=mc.min_pixels,
            image_max_pixels=mc.max_pixels,
            infer_backend=self._resolve_lf_infer_backend(),
        )
        self._backend.load()

    def _resolve_lf_infer_backend(self) -> str:
        backend = self.cfg.inference.backend
        architecture = self.cfg.model.architecture

        if backend == "hf":
            return "huggingface"

        if backend == "lf_vllm" and architecture in {"internvl", "llava", "llava_ov_1_5"}:
            # In the current stack, InternVL responds coherently with HF while the
            # vLLM path emits garbled text; LLaVA-OneVision-1.5 is outright
            # unsupported by vLLM. Keep both the legacy `llava` alias and the
            # explicit `llava_ov_1_5` architecture on the HF path.
            return "huggingface"

        return "vllm"

    def _run_benchmark(self, name: str, records: list[dict], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume only keeps successful records. Failed inference rows are retried.
        existing_successes = self._load_existing_successes(out_path)
        if existing_successes:
            print(f"[resume] {name}: keeping {len(existing_successes)} successful rows from {out_path}")

        pending = [r for r in records if r.get("id") not in existing_successes]
        if not pending:
            return out_path

        generated_by_id: dict = {}
        processed = 0
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i: i + self.batch_size]
            outputs = self._generate_with_recovery(batch)
            for out in outputs:
                generated_by_id[out["id"]] = out
            processed += len(batch)
            print(f"  [{name}] {processed}/{len(pending)}")

        with open(out_path, "w") as f:
            for record in records:
                rid = record.get("id")
                out = existing_successes.get(rid) or generated_by_id.get(rid)
                if out is None:
                    raise RuntimeError(f"Missing inference output for id={rid} in benchmark {name}.")
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

        return out_path

    def _load_existing_successes(self, out_path: Path) -> dict:
        successes: dict = {}
        if not out_path.exists():
            return successes

        with open(out_path) as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue
                rid = row.get("id")
                if rid is None or self._is_retryable_error_output(row):
                    continue
                successes[rid] = row
        return successes

    @staticmethod
    def _is_retryable_error_output(row: dict) -> bool:
        return "[INFERENCE_ERROR]" in str(row.get("response", ""))

    def _generate_with_recovery(self, records: list[dict]) -> list[dict]:
        assert self._backend is not None
        try:
            outputs = self._backend.generate_batch(records)
        except EngineCrashedError as e:
            print(f"[recover] engine crash in batch of {len(records)}: {e}")
            return self._recover_records(records)

        error_rows = [out for out in outputs if self._is_retryable_error_output(out)]
        if not error_rows:
            return outputs

        error_ids = [out.get("id") for out in error_rows]
        print(f"[recover] retrying {len(error_rows)} inference errors in safe mode: ids={error_ids}")
        recovered = self._recover_records(
            [record for record in records if record.get("id") in set(error_ids)]
        )
        recovered_by_id = {row["id"]: row for row in recovered}
        return [recovered_by_id.get(out["id"], out) for out in outputs]

    def _recover_records(self, records: list[dict]) -> list[dict]:
        self._enable_safe_mode()
        recovered: list[dict] = []
        for record in records:
            recovered.append(self._infer_single_with_retries(record))
        return recovered

    def _enable_safe_mode(self) -> None:
        assert self._backend is not None
        if self._safe_mode_enabled:
            return

        self._safe_mode_enabled = True
        print("[recover] switching backend to safe mode (vllm_enforce_eager=true, concurrency=1)")
        self._backend.restart(safe_mode=True, concurrency=1)

    def _restart_safe_backend(self) -> None:
        assert self._backend is not None
        self._safe_mode_enabled = True
        print("[recover] restarting safe-mode backend")
        self._backend.restart(safe_mode=True, concurrency=1)

    def _infer_single_with_retries(self, record: dict, max_attempts: int = 3) -> dict:
        assert self._backend is not None
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                [out] = self._backend.generate_batch([record])
            except EngineCrashedError as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_attempts:
                    self._restart_safe_backend()
                    continue
                break

            if not self._is_retryable_error_output(out):
                return out

            last_error = str(out.get("response", ""))
            if attempt < max_attempts:
                self._restart_safe_backend()

        msg = (
            f"[INFERENCE_ERROR] RecoveryFailed: exhausted {max_attempts} safe-mode attempts"
            f" for id={record.get('id')}"
        )
        if last_error:
            msg = f"{msg}; last_error={last_error}"
        return LFInferenceBackend.build_result(record, msg)

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
