# DREAMS Main Experiments — Initial Code Framework Plan (v3, LlamaFactory backbone)

> **Active version**. Supersedes v1 (vLLM-centric) and v2 (LF outdated path). Mirror file at `/mnt/hdd/xuran/.claude/plans/ai-vlm-safety-plan-yes-partitioned-tarjan.md`.
> **Pivot from v2**: LlamaFactory pulled fresh to `/mnt/hdd/xuran/LlamaFactory` (commit 53e77a9b, MiniCPM-V-4.6 added). 19 VLM templates available — much wider than the old LF audit indicated. `llamafactory-cli eval` deprecated; recommended path is `ChatModel.achat()` async batch.

---

## 0. Context

User directive: **all main experiments and ablation experiments run through LlamaFactory** (`/mnt/hdd/xuran/LlamaFactory/`). The `harness/` package becomes a thin layer providing only:
- DREAMS dataset registration / slicing / CF pair index
- Custom benchmark loaders (10 new — LF has no plug-in mechanism)
- GPT-4o judge wrapper (LF has no safety eval) — **implemented** via `/mnt/hdd/xuran/vlm_safety_harness/harness/evaluation/gpt4o_evaluator.py` and benchmark-aware routing in `run_experiment.py` / `run_eval_only.py`
- Safety-specific metrics (ASR/RSR/RR/HR + PD/PC/VS) — **implemented** in `/mnt/hdd/xuran/vlm_safety_harness/harness/evaluation/metrics.py`, including automatic `harm_type` / `img_source_type` slice emission when fields are present
- Paper tables
- Top-level orchestration scripts

LF research findings (`/mnt/hdd/xuran/LlamaFactory/src/llamafactory/`):
- **Templates**: 19 VLM families registered in `data/template.py` — including `intern_vl`, `qwen2_vl` (deprecated), `qwen3_vl` + `qwen3_vl_nothink` (new), `llava_next` (LLaVA-OneVision-1.5), `kimi_vl`, `minicpm_v` / `minicpm_v_4_6` / `minicpm_o`, `gemma4` (new vision support), `glm4v` / `glm4_5v`, `granite3_vision`, `pixtral`, `video_llava`, `lfm2_vl`. Multi-image natively supported (`limit_mm_per_prompt={"image": 4}` in `chat/vllm_engine.py:94`)
- **Still missing**: Idefics2-8B, Phi-4-multimodal, DeepSeek-VL2, Ovis2/2.5 — drop from initial cohort
- **CLI**: `train`, `chat`, `api`, `export`, `webchat`, `webui`, `env`, `version` — **no `batch`/`predict`/`infer`** subcommand. `eval` is deprecated
- **Inference path**: `from llamafactory.chat import ChatModel`; `ChatModel.achat(messages, images=[...])` async; combine with `asyncio.gather` for batch
- **Latest commit (2026-05-09)**: `53e77a9b` adds MiniCPM-V-4.6
- **`harness/training/trainer.py:24`** still points to old path `/mnt/hdd/xuran/LLaMA-Factory` — must update to `/mnt/hdd/xuran/LlamaFactory`

User-confirmed open questions (resolved 2026-05-10):
- Qwen3.5-9B confirmed at `https://huggingface.co/Qwen/Qwen3.5-9B` (user-verified). Plan uses this HF id directly. LF template still maps to `qwen3_vl` (since Qwen3.5 series shares Qwen3-VL architecture conventions per LF registry).
- 4 unsupported archs (Idefics2 / Phi-4-MM / Ovis2 / DeepSeek-VL2) → **dropped from Tier B**. User will substitute LF-supported alternatives later if cohort expansion needed.
- LLaVA-OV-1.5 → `llava_next` template (confirmed).
- GLM-4.6V-Flash → use `glm4_5v` template (user-confirmed: 4.5V template runs 4.6V).
- Closed-source (Tier C): API ids + model names provided by user at run time.
- E4 V2/V4 general-data mix: resolved. **V2 uses 500 M4-Instruct samples; V4 uses M4-Instruct at 11% final-data ratio.** Runtime should only remind about local data availability before launch.
- CF pair construction rule: user inspects `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json` directly to confirm `path_name` → pair convention. Phase 2 implementation locks the rule after user confirmation.

---

## 1. Reuse Map (final v3 distribution)

| Layer | Owner | File / Component |
|-------|-------|-------------------|
| Multi-arch chat templates + mm_plugin (19 VLM families) | **LF** | `LlamaFactory/src/llamafactory/data/{template.py, mm_plugin.py}` |
| Multi-image prompt formatting + image_token expansion | **LF** | `mm_plugin.py:_make_batched_images` |
| SFT training (full / LoRA / QLoRA / DPO) | **LF** | `llamafactory-cli train <yaml>` → `train/tuner.py:run_exp` |
| DeepSpeed configs (ZeRO 0/2/3 + offload variants) | **LF** | `LlamaFactory/examples/deepspeed/ds_z{0,2,3}_config.json` |
| Batch inference (vLLM async + HF + SGLang) | **LF** | `chat/{chat_model.py:ChatModel.achat, vllm_engine.py}` |
| Checkpoint mgmt | **LF** | HF Trainer auto-output `checkpoint-*` dirs |
| Dataset registration (sharegpt JSON + `dataset_info.json`) | LF + harness | harness writes entries via `harness/training/trainer.py:register_dataset` |
| GPU planning (TrainPlan / InferPlan) | **harness** | `harness/gpu/allocator.py` |
| Config schema + `_extends` inheritance + registry | **harness** | `harness/config/{schema, loader, registry}.py` |
| DREAMS dataset loading + slicing | **harness** | `harness/data/dataset.py` (extend Phase 2) |
| Benchmark loaders (DREAMS + MIS + MSSBench + FigStep + 10 new) | **harness** | `harness/data/benchmarks/` (extend Phase 4) |
| GPT-4o judge | **harness** | `harness/evaluation/gpt4o_evaluator.py` (reuse) |
| Safety metrics (ASR/RSR/RR/HR + PD/PC/VS + slicing) | **harness** | `harness/evaluation/metrics.py` (extend Phase 6) |
| Reporting tables (E1/E2/E3/E4/E5) | **harness** | `harness/reporting/table_generator.py` (extend Phase 7) |
| End-to-end orchestration | **harness** | `scripts/run_*.py` (extend Phase 8) |

**Deleted in v3** (LF subsumes):
- ❌ `harness/inference/vllm_backend.py:_prompt_<arch>` per-arch builders (~120 LOC) — replaced by single `LFInferenceBackend` wrapping `ChatModel.achat`
- ❌ `harness/inference/model_configs.py:ARCH_CONFIGS` 8 new arch entries from v1
- ❌ Architecture-specific dispatch in `_build_prompt`

---

## 2. Architecture Coverage (v3 — based on new LlamaFactory)

| Tier | Model | LF template | Status | Notes |
|------|-------|-------------|--------|-------|
| **A** | InternVL3.5-8B | `intern_vl` | ✅ | Tier A SFT target #1 |
| **A** | Qwen3.5-9B | `qwen3_vl` | ✅ | HF id `Qwen/Qwen3.5-9B` confirmed by user |
| **A** | LLaVA-OV-1.5-8B | `llava_next` | ✅ | One of 9 LLaVA-Next variants |
| **B 7-9B** | Kimi-VL-A3B | `kimi_vl` | ✅ | image-only, no video token |
| **B 7-9B** | MiniCPM-o-4.5 | `minicpm_o` | ✅ | image+video+audio |
| **B 7-9B** | MiniCPM-V-4.6 | `minicpm_v_4_6` (added 2026-05-09) | ✅ | latest LF commit |
| **B 7-9B** | Gemma-4-E4B (vision) | `gemma4` | ✅ | vision support new in this LF |
| **B 7-9B** | GLM-4.6V-Flash | `glm4_5v` | ✅ | user-confirmed: 4.5V template runs 4.6V |
| ~~**B 7-9B**~~ | ~~Phi-4-multimodal~~ | — | ❌ DROPPED | user-approved removal |
| ~~**B 7-9B**~~ | ~~Idefics2-8B~~ | — | ❌ DROPPED | user-approved removal |
| ~~**B 7-9B**~~ | ~~Ovis2.5-9B~~ | — | ❌ DROPPED | user-approved removal |
| ~~**B 7-9B**~~ | ~~DeepSeek-VL2-Tiny~~ | — | ❌ DROPPED | user-approved removal |
| **B 4B** | Qwen3-VL-4B (or Qwen3.5-4B per user list) | `qwen3_vl` | ✅ | |
| **B 4B** | InternVL3.5-4B | `intern_vl` | ✅ | |
| **B 4B** | LLaVA-OV-1.5-4B | `llava_next` | ✅ | |
| **B 4B** | Gemma-4-E2B | `gemma4` | ✅ | |
| — | (optional new in v3 from new LF) Pixtral-12B | `pixtral` | ⚪ | not in user roster; available if cohort expands |
| — | Granite3-Vision | `granite3_vision` | ⚪ | not in user roster |
| — | Video-LLaVA | `video_llava` | ⚪ | not in user roster |
| **C** | GPT-5.5 / Gemini-3.1-Pro / Claude-Opus-4.7 | (API) | ✅ | direct SDK calls, Phase 8 |

**Initial framework cohort**:
- Tier A: 3 archs × 3 variants (DREAMS / MIRage-data / base) = **9 SFT or inference targets**
- Tier B (LF-supported): **8 inference-only baselines** (Kimi-VL, MiniCPM-o-4.5, MiniCPM-V-4.6, Gemma-4-E4B, GLM-4.6V-Flash + 3 4B-class)
- Tier C: **3 closed-source** via direct SDK (API ids supplied by user at run time)
- ~~Deferred 4 archs~~ — **dropped per user**. Will swap in LF-supported alternatives if cohort expansion needed.

---

## 3. Gap Inventory (v3)

| # | Gap | Files | Phase |
|---|-----|-------|-------|
| G1 | Update `LLAMAFACTORY_ROOT` path: `LLaMA-Factory` → `LlamaFactory` | `harness/training/trainer.py:24` | 1 |
| G2 | Fix `ARCH_TO_TEMPLATE` template-name bugs + add new arch IDs (`qwen3_vl`, `gemma4`, `glm4v`, `kimi_vl`, `minicpm_o`, `minicpm_v_4_6`); remove `idefics`, `phi` (text-only) | `harness/training/trainer.py:28` | 1 |
| G3 | Replace `harness/inference/vllm_backend.py` with `LFInferenceBackend` wrapping `ChatModel.achat` (async batch via `asyncio.gather`) | `harness/inference/lf_backend.py` (new); `harness/inference/vllm_backend.py` (thin re-export) | 1 |
| G4 | Update Pydantic `architecture` Literal to LF-supported arch IDs (validates configs at load time) | `harness/config/schema.py:18` | 1 |
| G5a | DREAMS test slicing helpers — `img_source_type` rule-based; `harm_type` already in `test.json` | `harness/data/dataset.py` | 2 |
| G5b | CF pair **synthesizer** — DREAMS test has no native CF pairs (audit confirmed: no `cf_id` / `pair_id` field, no `cf_pairs.json`). For each unsafe test record, swap one image with a benign one from a public pool (ImageNet / OpenImages / COCO) → produce `(orig_unsafe, cf_safe)` pair. Output: `dataset/test_cf.json` consumed by E5. | new `harness/data/cf_synthesizer.py`, new `scripts/build_cf_pairs.py` | 2 |
| G6 | Base model configs (3 new) | `configs/base/model_{internvl3_5_8b, qwen3_vl_8b, llava_ov_1_5_8b}.yaml` | 3 |
| G7 | DREAMS SFT configs (3 archs × harness YAML; LF training YAML auto-generated by `HarnessTrainer.build_llamafactory_yaml`) | `configs/experiments/main/main_dreams_{internvl3_5, qwen3_vl, llava_ov}.yaml` | 3 |
| G8 | MIRage-data SFT configs (3 archs, mirror of G7 with `dataset.train_path → mis_train.json`) | same dir | 3 |
| G9 | Tier B baseline inference configs (8 LF-supported models) | template-driven via `scripts/generate_baseline_configs.py` + `_tier_b_models.csv` | 3 |
| G10 | E4 V0–V4 variant configs (5 per baseline; InternVL3.5-8B first; V2/V4 marked `enabled: false` until general-data mix decided) | `configs/experiments/main/E4_V{0..4}_internvl3_5.yaml` | 3 |
| G11 | Benchmark loaders (10 new): AdvBench, SafeBench, MM-Safety, JailbreakV, SIUO, MSS, MMStar, MMMU, MuirBench, BLINK, MMT — emit sharegpt JSON + `dataset_info.json` entry so LF batch infer handles them. Full impl: AdvBench / MM-Safety / MMStar. Stubs: 7 others | `harness/data/benchmarks/<name>.py` | 4 |
| G12 | InferenceEngine BENCHMARK_REGISTRY entries | `harness/inference/engine.py:22` | 4 |
| G13 | Pluggable evaluator interface (`BenchmarkEvaluator` ABC + 4 backends: gpt4o, rule, harmbench, accuracy) | `harness/evaluation/benchmark_evaluator.py` (new); `harness/evaluation/evaluators/{gpt4o, rule_based, harmbench, accuracy}.py` | 5 |
| G14 | `--evaluator-type {auto, gpt4o, rule, harmbench, accuracy}` routing | `scripts/run_eval_only.py` | 5 |
| G15 | Metrics: `slice_fields=['harm_type'/'img_source_type']` + `compute_pair_metrics` (PD/PC/VS, replace existing `compute_counterfactual_metrics`) | `harness/evaluation/metrics.py:64,118` | 6 |
| G16 | 5 new TableGenerator methods (`e1_table`, `e2_table`, `e3_table`, `e4_table`, `e5_table`) | `harness/reporting/table_generator.py` | 7 |
| G17 | `scripts/run_main.py` (Tier A + B over E1+E2+E5; calls `llamafactory-cli train` for SFT and `LFInferenceBackend` for inference) | new | 8 |
| G18 | `scripts/run_closed_source.py` (Tier C SDK path) | new | 8 |
| G19 | `scripts/run_capability.py` (E4; uses accuracy evaluator) | new | 8 |

---

## 4. Phased Implementation

### Phase 1 — Harness ↔ LF integration (G1–G4)

**Goal**: harness's training + inference paths route entirely through LF, no per-arch code.

**Tasks**:

1. **Path fix** in `harness/training/trainer.py:24`:
   ```python
   LLAMAFACTORY_ROOT = Path("/mnt/hdd/xuran/LlamaFactory")
   ```

2. **`ARCH_TO_TEMPLATE`** update (verified template names from new LF `template.py`):
   ```python
   ARCH_TO_TEMPLATE = {
       "internvl":         "intern_vl",            # was "internvl2_5" (wrong)
       "qwen2vl":          "qwen2_vl",
       "qwen3_vl":         "qwen3_vl",             # NEW
       "llava":            "llava_next",
       "kimi_vl":          "kimi_vl",              # NEW
       "minicpm":          "minicpm_v",
       "minicpm_v_4_6":    "minicpm_v_4_6",        # NEW (latest LF commit)
       "minicpm_o":        "minicpm_o",            # NEW
       "gemma_vlm":        "gemma4",               # NEW (LF gained vision Gemma)
       "glm4v":            "glm4v",                # NEW
   }
   # Removed: "idefics" (LF lacks Idefics2), "phi" (text-only — not VLM)
   ```

3. **New `harness/inference/lf_backend.py`** — `LFInferenceBackend` wrapping `ChatModel.achat`:
   ```python
   import asyncio
   from llamafactory.chat import ChatModel

   class LFInferenceBackend:
       def __init__(self, model_path, adapter_path=None, template,
                    infer_plan, max_new_tokens=1024, temperature=0.0,
                    concurrency=16):
           self._chat = ChatModel({
               "model_name_or_path": model_path,
               "adapter_name_or_path": adapter_path,
               "template": template,
               "infer_backend": "vllm",
               "vllm_gpu_util": infer_plan.gpu_memory_utilization,
               "vllm_maxlen": 8192,
               "max_new_tokens": max_new_tokens,
               "temperature": temperature,
           })
           self._sem = asyncio.Semaphore(concurrency)

       async def _one(self, record):
           async with self._sem:
               messages = [{"role": "user",
                            "content": "<image>\n<image>\n" + record["question"]}]
               resp = await self._chat.achat(
                   messages,
                   images=[record["image_path1"], record["image_path2"]],
               )
               return {**record, "response": resp[0].response_text}

       def generate_batch(self, records: list[dict]) -> list[dict]:
           return asyncio.run(asyncio.gather(*(self._one(r) for r in records)))
   ```

4. **Delete** `vllm_backend.py:_prompt_qwen2vl`, `_prompt_internvl`, `_prompt_phi`, `_prompt_idefics`, `_prompt_llava`, `_prompt_minicpm` (~120 LOC). Keep `vllm_backend.py` as a thin file that re-exports `LFInferenceBackend` for backward compat.

5. **`harness/config/schema.py:18`** — restrict `architecture: Literal[...]` to the 10 ARCH_TO_TEMPLATE keys above, plus `closed_source` for Tier C placeholder configs.

6. **`harness/inference/engine.py:36`** — instantiate `LFInferenceBackend` instead of `VLLMBackend`.

**Smoke test (Phase 1)**:
```bash
python -c "
import asyncio
from harness.inference.lf_backend import LFInferenceBackend
from harness.gpu.allocator import GPUAllocator
plan = GPUAllocator().plan_inference(model_size_b=8.0)
b = LFInferenceBackend(
    model_path='OpenGVLab/InternVL3_5-8B',
    template='intern_vl',
    infer_plan=plan,
)
out = b.generate_batch([{
    'id': 1, 'question': 'What do you see?',
    'image_path1': 'tests/fixtures/img1.jpg',
    'image_path2': 'tests/fixtures/img2.jpg',
}])
print(out[0]['response'][:200])
"
```

### Phase 2 — DREAMS test slicing + CF pair synthesis (G5a + G5b)

**Goal**: HarnessDataset slices `test.json` by `img_source_type` (rule-derived) and `harm_type` (already in test.json). New `cf_synthesizer.py` builds CF pairs offline by image-swap with benign pool — DREAMS has no native CF metadata (audit confirmed).

#### G5a — Slicing helpers (`harness/data/dataset.py`)

- `derive_img_source_type(record) -> Literal["synth","real","mix","unknown"]`:
  ```python
  def derive_img_source_type(rec):
      a, b = rec.get("img1_source", ""), rec.get("img2_source", "")
      is_synth = lambda x: x in ("AI-generated", "SD-3.5", "Stable Diffusion")
      is_real  = lambda x: x in ("Web-crawled", "Web-retrieved", "local", "web")
      if is_synth(a) and is_synth(b): return "synth"
      if is_real(a)  and is_real(b):  return "real"
      if (is_synth(a) and is_real(b)) or (is_real(a) and is_synth(b)): return "mix"
      return "unknown"
  ```
- Pass through `harm_type` from JSON (already populated).
- New constructor params: `filter_harm_type`, `filter_img_source_type`.

#### G5b — CF synthesizer (`harness/data/cf_synthesizer.py` + `scripts/build_cf_pairs.py`)

**Input**: `test.json` records (all unsafe by construction).
**Output**: `dataset/test_cf.json` — for each `orig` record, one `cf_safe` partner where one image is swapped with a benign image. Structure:
```json
[
  {"orig_id": 15320, "cf_id": 1015320, "swap_idx": 2, "benign_image_path": "..."},
  ...
]
```

**Synthesizer interface**:
```python
class CFSynthesizer:
    def __init__(self, benign_pool: Path, swap_image_idx: int = 2, seed: int = 0):
        """benign_pool = directory of benign images.
        swap_image_idx = which image (1 or 2) to replace; default 2."""

    def synthesize(self, test_records: list[dict], output_dir: Path) -> Path:
        """For each unsafe record, sample 1 benign image, copy to output_dir/cf_images/<cf_id>.png,
        write a CF record with cf_id = orig_id + 1_000_000 (collision-safe offset).
        Returns path to test_cf.json."""

    def to_pair_index(self, cf_records: list[dict]) -> dict[int, int]:
        """{orig_id: cf_id} mapping for compute_pair_metrics."""
```

**Benign pool source — TBD before Phase 2 starts** (user input or auto-pick):
| Source | Pros | Cons | Status |
|--------|------|------|--------|
| OpenImages (`pip` already installed at `/mnt/hdd/xuran/anaconda3/lib/python3.13/site-packages/openimages/`) | diverse, real-world, license-clean | 9M+ images — must subsample | **Recommended** |
| ImageNet (already in HF cache) | already local, large | object-centric, less context-rich | fallback |
| COCO val2017 | mid-size, common-objects | not local; needs download | last choice |

**Quality control**: a tiny VLM judge pass (e.g., GPT-4o-mini) on each synthesized `(orig_question, benign_image, retained_image)` triple to verify the pair is genuinely safe (no accidental harm by question + benign-image combo). Reject + re-sample if judged unsafe. Keep loop bounded (max 3 retries per record; drop if still unsafe).

**Build script** (`scripts/build_cf_pairs.py`):
```bash
python scripts/build_cf_pairs.py --test-json /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json --benign-source openimages --benign-pool-size 5000 --output /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test_cf.json --quality-judge gpt-4o-mini --max-retries 3
```

Run **once offline**; output committed (or symlinked into `data_links/`). E5 reads `test_cf.json` directly.

#### Smoke test (Phase 2)

```bash
# G5a slicing
python -c "
from harness.data.dataset import HarnessDataset
ds = HarnessDataset(
    data_path='/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json',
    image_root='/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset',
    mode='eval', filter_img_source_type='real')
print('real:', len(ds))
ds = HarnessDataset(data_path='/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json',
    image_root='/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset',
    mode='eval', filter_harm_type='implicit')
print('implicit:', len(ds))
"

# G5b CF synth — 50-sample dry run
python scripts/build_cf_pairs.py --test-json /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json --benign-source openimages --benign-pool-size 100 --output /tmp/test_cf_smoke.json --limit 50 --skip-judge
```

### Phase 3 — Configs (G6–G10)

**Goal**: every experiment cell has a YAML config. Each main experiment has TWO YAMLs:
- LF-native training YAML (`configs/experiments/main/lf_training/<name>.yaml`) shaped after `LlamaFactory/examples/train_full/qwen3vl_full_sft.yaml` — auto-generated by `HarnessTrainer.build_llamafactory_yaml` (existing logic; just verify against Qwen3-VL example).
- Harness orchestration YAML (`configs/experiments/main/<name>.yaml`) — top-level config consumed by `scripts/run_experiment.py`.

**Hand-written**:
- `configs/base/model_{internvl3_5_8b, qwen3_vl_8b, llava_ov_1_5_8b}.yaml` (3)
- `configs/experiments/main/main_dreams_{internvl3_5, qwen3_vl, llava_ov}.yaml` (3) — `_extends` base, `dataset.train_path → train_annotated.json`, `inference.benchmarks: [our_test]`
- `configs/experiments/main/main_baseline_mirage_data_{...}.yaml` (3) — same shape, `dataset.train_path → mis_train.json`
- `configs/experiments/main/E4_V{0..4}_internvl3_5.yaml` (5)

**Generated**:
- `_baseline_template.yaml` + `_tier_b_models.csv` (8 LF-supported rows) → `scripts/generate_baseline_configs.py` emits 8 inference-only YAMLs.

**Smoke test (Phase 3)**:
```bash
python -c "
from harness.config.loader import ConfigLoader
import glob
errs = []
for p in glob.glob('configs/experiments/main/*.yaml'):
    if p.split('/')[-1].startswith('_'): continue
    try: ConfigLoader.load(p); print('OK', p)
    except Exception as e: errs.append((p, str(e)))
assert not errs, errs
"
```

### Phase 4 — Benchmark loaders (G11–G12)

**Goal**: 10 new benchmark loaders that emit sharegpt-format JSON for LF batch inference. 3 priority impls (AdvBench / MM-Safety / MMStar); 7 stubs.

**Per-loader contract** (extending existing `Benchmark` ABC):
```python
class FooBench(Benchmark):
    name = "foo_bench"
    metric_name: str = "ASR"
    metric_direction: Literal["↑","↓"] = "↓"
    evaluator_type: Literal["gpt4o","rule","harmbench","accuracy"] = "gpt4o"

    def __init__(self, data_path, ...): ...
    def load(self) -> list[dict]: ...
    def to_sharegpt(self, output_path: Path) -> Path: ...   # for LF inference
```

**Decision-by-inspection**: Phase 4 first step is `huggingface-cli download <id> --repo-type dataset --local-dir data_links/<name>/`, then read each repo's README to fix `evaluator_type` per benchmark.

**Engine wiring** (`harness/inference/engine.py:22`): extend BENCHMARK_REGISTRY dict.

**Smoke test (Phase 4)**: per-impl record load + sharegpt JSON write.

### Phase 5 — Pluggable evaluator interface (G13–G14)

**Goal**: `run_eval_only.py` routes responses through evaluator declared by benchmark loader.

- New `harness/evaluation/benchmark_evaluator.py` (`BenchmarkEvaluator` ABC)
- New `harness/evaluation/evaluators/{gpt4o, rule_based, harmbench, accuracy}.py`
- `run_eval_only.py` adds `--evaluator-type {auto, ...}` flag

**Smoke test**: AdvBench responses → rule evaluator → ASR.

### Phase 6 — Metrics extensions (G15)

- Extend `compute_metrics(slice_fields=['harm_type', 'img_source_type'])` → returns `per_harm_type`, `per_img_source_type`
- Replace `compute_counterfactual_metrics` (`metrics.py:118`) with `compute_pair_metrics(orig, cf, pair_index, vs_threshold=0.3)` returning **PD / PC / VS**
- Update `MetricsDict` dataclass with new optional fields

**Smoke test**: stub records → compute_metrics returns slicing dicts.

### Phase 7 — Reporting tables (G16)

5 new TableGenerator methods matching handoff §6 layouts: `e1_table` (21×8), `e2_table` (24×12), `e3_table` (model × benchmark canonical metric, header w/ ↑/↓), `e4_table` (5 V0–V4 rows × 5 capability benchmarks), `e5_table` (6 rows × PD/PC/VS).

**Smoke test**: `generate_report.py --experiment-set e1 --format markdown` on stub data.

### Phase 8 — Orchestration scripts (G17–G19)

- `scripts/run_main.py`: `--experiment-id E1|E2|E3|E5|all`. Drives Tier A (SFT via `llamafactory-cli train` from `HarnessTrainer.prepare_and_run`) + Tier B (inference-only via `LFInferenceBackend`). Reads cohort definition from `configs/experiments/main/_cohort.yaml`.
- `scripts/run_closed_source.py`: direct SDK calls (`openai`, `google.generativeai`, `anthropic`); responses → harness GPT-4o eval → metrics.
- `scripts/run_capability.py`: trains E4 V1/V3 (V2/V4 only after general-data mix decision); runs capability benchmarks via accuracy evaluator.

**Smoke test**: `run_main.py --experiment-id E1 --dry-run --limit 5` completes.

### Phase 9 — End-to-end smoke verification

5-sample run for one Tier A model: train → infer → eval → table. Pass criteria: E1 markdown table cell populated for InternVL3.5+DREAMS row (8 cells: explicit ASR/RS/HR/FPR + implicit ASR/RS/HR/FPR), no exceptions, `metrics.json` contains `per_harm_type` block.

---

## 5. Resolved Questions + Active Reminders

**Resolved (2026-05-10)**:

1. ✅ **Qwen3.5-9B**: `https://huggingface.co/Qwen/Qwen3.5-9B` (user-confirmed). Plan uses this id with LF template `qwen3_vl`.
2. ✅ **CF pair source**: DREAMS test has no native CF pair metadata (audit confirmed). User chose **Option B — runtime synthesis**: image-swap from benign public pool (OpenImages preferred, ImageNet fallback) → `dataset/test_cf.json` produced offline once via `scripts/build_cf_pairs.py`. E5 reads this file. See Phase 2 G5b for implementation contract.
3. ✅ **4 unsupported archs dropped**: Idefics2, Phi-4-multimodal, Ovis2.5, DeepSeek-VL2. User will pick LF-supported alternatives if cohort needs expansion.
4. ✅ **GLM-4.6V**: use `glm4_5v` template (user-confirmed: 4.5V template runs 4.6V).
5. ✅ **LLaVA-OV-1.5**: `llava_next` template confirmed.
6. ✅ **E4 V2/V4 general data mix**: resolved. V2 uses **500 M4-Instruct** samples; V4 uses **M4-Instruct at 11% final-data ratio**. `scripts/run_capability.py` still guards only for a populated `general_data` block and local source availability.
7. ✅ **Tier C API ids**: supplied by user at runtime; runtime check guards launch.

**Active reminders** (plan must surface these at the right phase):

| Trigger | Reminder | Owner |
|---------|----------|-------|
| Before E4 V2 / V4 training run | Surface to user: "local M4-Instruct source must exist at `/mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct` before launch; V2 is fixed at 500 samples and V4 is fixed at 11% final-data ratio." | `scripts/run_capability.py` checks V2/V4 config has `enabled: true` + `general_data` block populated; `harness/data/converters.py` raises a clear error if the configured source path is missing or empty |
| Before Tier C inference run | Surface to user: "supply API key + model id for GPT-5.5 / Gemini-3.1-Pro / Claude-Opus-4.7." | `scripts/run_closed_source.py --models` flag must be non-empty; raises clear error if Tier C requested but no models supplied |
| Before E5 inference | `dataset/test_cf.json` must exist — built once by `scripts/build_cf_pairs.py`. If missing, halt with usage hint. | `scripts/run_main.py --experiment-id E5` precheck |
| Before `build_cf_pairs.py` run | Confirm benign pool source + size with user (OpenImages 5K default; user can override to ImageNet / COCO) | `build_cf_pairs.py` requires `--benign-source` flag explicitly; no default to force the choice |

---

## 6. Verification Summary

| Phase | Smoke pass criterion |
|-------|---------------------|
| 1 | `LFInferenceBackend` produces non-empty response on InternVL3.5 + 2-image prompt |
| 2 | `HarnessDataset` slices return non-zero counts for real/synth/mix/explicit/implicit; `build_cf_pairs.py --limit 50 --skip-judge` produces a valid `test_cf.json` with 50 entries |
| 3 | All `configs/experiments/main/*.yaml` load via `ConfigLoader.load()` without errors |
| 4 | 3 priority benchmark loaders emit valid sharegpt JSON; LF batch infer runs ≥1 sample |
| 5 | rule evaluator returns ASR on AdvBench responses; accuracy evaluator returns acc on MMStar responses |
| 6 | `compute_metrics(slice_fields=['harm_type'])` returns `per_harm_type`; `compute_pair_metrics` returns PD/PC/VS |
| 7 | `generate_report.py` produces markdown for each new table method (e1..e5) |
| 8 | `run_main.py --experiment-id E1 --dry-run --limit 5` completes |
| 9 | End-to-end Tier A smoke run produces markdown E1 table cell + `metrics.json` with `per_harm_type` block |

---

## 7. File-Modification Summary (v3)

**New files** (~21):
- `configs/base/model_{internvl3_5_8b, qwen3_vl_8b, llava_ov_1_5_8b}.yaml` (3)
- `configs/experiments/main/main_dreams_{internvl3_5, qwen3_vl, llava_ov}.yaml` (3)
- `configs/experiments/main/main_baseline_mirage_data_{...}.yaml` (3)
- `configs/experiments/main/_baseline_template.yaml`, `_tier_b_models.csv`, `_cohort.yaml` (3)
- `configs/experiments/main/E4_V{0..4}_internvl3_5.yaml` (5)
- `harness/inference/lf_backend.py` (1)
- `harness/data/cf_synthesizer.py` (1)
- `scripts/build_cf_pairs.py` (1)
- `harness/data/benchmarks/{advbench, safebench, mm_safety, jailbreakv, siuo, mmstar, mmmu, muirbench, blink, mmt}.py` (10)
- `harness/evaluation/benchmark_evaluator.py` + `harness/evaluation/evaluators/{gpt4o, rule_based, harmbench, accuracy}.py` (5)
- `scripts/{run_main, run_closed_source, run_capability, generate_baseline_configs}.py` (4)

**Modified files** (~7):
- `harness/training/trainer.py` (LLAMAFACTORY_ROOT path + ARCH_TO_TEMPLATE rewrite)
- `harness/inference/vllm_backend.py` (deletions; thin re-export shim)
- `harness/inference/engine.py` (use LFInferenceBackend; BENCHMARK_REGISTRY extension)
- `harness/config/schema.py` (architecture Literal restricted)
- `harness/data/dataset.py` (slicing helpers)
- `harness/evaluation/metrics.py` (slice_fields + PD/PC/VS)
- `harness/reporting/table_generator.py` (5 new methods)
- `scripts/run_eval_only.py` (evaluator-type routing)

---

## 8. Plan Version History

| Version | Path | Status | Pivot reason |
|---------|------|--------|--------------|
| v1 | `docs/main_experiments/initial_framework_plan_v1_vllm.md` | Superseded | Per-arch prompt builders in vllm_backend; LF only used for SFT |
| v2 | `docs/main_experiments/initial_framework_plan_v2_llamafactory_old_path.md` | Superseded | Used outdated LF clone at `/mnt/hdd/xuran/LLaMA-Factory` (failed pull) |
| **v3** | **this file** | **Active** | Fresh LF clone at `/mnt/hdd/xuran/LlamaFactory`; 19 templates audited; Qwen3-VL surfaced; LF eval deprecation noted |

v1 / v2 archived for reference. v3 incorporates all user clarifications:
- harm_type already in test.json (user)
- LF as backbone for all main + ablation experiments (user)
- Plans must be visible in project tree, not just `.claude/plans/` (user — memory)
