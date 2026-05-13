# A Experiments: MIS Diagnostic Study (Section 2)

**Constraint**: All A experiments run **before** DREAMS dataset is built.  
Data sources: MIS test sets + public benchmarks only. No DREAMS data, no custom-trained models.

---

## Model Roster for A Experiments

### Base VLMs (no safety SFT)

| Model | HF ID | Size | Multi-Image | vLLM | Arch key |
|-------|-------|------|-------------|------|----------|
| **InternVL3.5-8B** | `OpenGVLab/InternVL3_5-8B` | 8.5B | ✓ | ✓ | `internvl` |
| **Qwen3.5-9B** | `Qwen/Qwen3.5-9B` | 9B | ✓ | ✓ | `qwen2vl` |
| **LLaVA-OV-1.5-8B** | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | 8B | ✓ | ✓ | `llava` |

### Safety-SFT Comparison

| Model | Description | Multi-Image | vLLM | Role |
|-------|-------------|-------------|------|------|
| **MIRage** | InternVL2.5-8B + MIS fine-tune | ✓ | ✓ | MIS safety SFT baseline |

### Excluded from A experiments

| Model | Reason |
|-------|--------|
| Janus-Pro-7B | Single-image generative model, incompatible with dual-image task |
| DeepSeek-VL2-Tiny | Reserved for main experiment only |
| idefics2-8b | No vLLM support — reserved for main experiment (HF pipeline) |

### Architecture for inference
- All models: vLLM backend, `harness/inference/vllm_backend.py`
- 1 GPU per model (A6000 48GB sufficient for ≤9B)

---

## Narrative Structure

Following MIS Section 2 style: propose hypothesis → design experiment → "discover" weakness → motivate our improvement.

> "MIS 是多图安全推理的奠基工作，但我们怀疑其数据集在以下方面存在根本性局限。
> 为此设计 A1–A4 四组诊断实验，全部使用 MIS 已有数据和公开基准，
> 结果显示……这四个发现共同揭示了更优质数据集的必要性。"

Using 3 modern base VLMs instead of 1 demonstrates that the MIS weaknesses are **architecture-agnostic**, not InternVL-specific.

---

## A1: Textual Shortcut Diagnosis (P2)

### Hypothesis
MIRage 的安全收益部分来自**文本模板识别**，而非真正的视觉推理。  
MIS 文本指令遵循固定句式（"use the object in the first image to..."），模型无需看图即可触发安全行为。

### Data Sources
| Split | Path | Samples |
|-------|------|---------|
| MIS-easy | `/mnt/hdd/xuran/MIS/mis_test/mis_easy.json` | 1,675 |
| MIS-hard | `/mnt/hdd/xuran/MIS/mis_test/mis_hard.json` | 510 |

**Probe construction** (`scripts/run_prelim.py --build-probes --experiment A1`):
- Condition A: original dual-image input
- Condition B: both images replaced with 224×224 black frames, text unchanged
- Stored at: `results/prelim/probes/probe_text_only.json`

### Models
| Model | Condition A | Condition B |
|-------|-------------|-------------|
| InternVL3.5-8B (base) | ✓ | ✓ |
| Qwen3.5-9B (base) | ✓ | ✓ |
| LLaVA-OV-1.5-8B (base) | ✓ | ✓ |
| MIRage (InternVL2.5-8B+MIS) | ✓ | ✓ |

### Evaluation Protocol
GPT-4o structured eval (identical to `MIS/evaluation/gpt_eval.py`):
- `label_perception`: Perception Level1 / Level2 / Level3
- `label_str`: Safe / Unsafe
- Metrics: **ASR** (primary), RSR, RR, HR

### Key Metric
```
ΔASR = ASR_full_image - ASR_text_only
```
If ΔASR ≈ 0 for MIRage → safety from text, not vision.

### Expected Findings
| Condition | Base (any arch) | MIRage |
|-----------|-----------------|--------|
| Condition A (full image) | ~80-90% | ~0-5% |
| Condition B (text-only) | ~75-85% | ~0-5% |
| ΔASR | ≈5-10% (small) | ≈0% (**safety from text**) |

MIRage 在 text-only 条件下 ASR 仍接近 0%，**三种 base 模型的 ΔASR 均较小**，定量揭示文本捷径问题是架构无关的。

### Motivates
DREAMS 中将危险意图更深地藏于视觉语义，而非文本模板。

### Config
`configs/experiments/prelim/A1_textual_shortcut.yaml`

---

## A2: Relation Pattern Coverage Diagnosis (P3)

### Hypothesis
MIS 危害样本 90%+ 遵循单一 **tool→target** 组合模式（图1=手段，图2=目标）。  
MIRage 在非标准关系类型上缺乏安全推理能力。

### Data Sources
**Step 1 — Statistical analysis** (GPT-4o relation type annotation):
- Input: MIS-hard 510 samples
- GPT-4o labels each sample with `relation_type`
- Output: `results/prelim/probes/mis_hard_relation_annotated.json`

**Step 2 — Relation Probe evaluation**:
| Relation Type | Source | N |
|--------------|--------|---|
| tool→target (MIS standard) | MIS-hard direct sample | 50 |
| before→after (temporal causal) | COCO + hand-written text | ~50 |
| identity-linking (identity→behavior) | OpenImages + hand-written | ~50 |
| context-shift (safe→sensitive context) | COCO + hand-written | ~50 |

Hand-crafted probe JSONL: `results/prelim/probes/extra_relation_probe.jsonl`

**Image sources for hand-crafted probes**:
- COCO: not found in HF cache (`~/.cache/huggingface/datasets/` only has ImageNet).
  Use `fiftyone` (already installed in base env) or `pycocotools` to download subset on demand.
- OpenImages: use `openimages` package (installed at `/mnt/hdd/xuran/anaconda3/lib/python3.13/site-packages/openimages/`) to pull ~50 images per category.
- Alternative: search within DREAMS `mis_dataset_builder` image pool — but only for images **not** in our train/test split to avoid data contamination.

### Models
| Model | Distribution Analysis | ASR by Type |
|-------|----------------------|-------------|
| InternVL3.5-8B (base) | — | ✓ |
| Qwen3.5-9B (base) | — | ✓ |
| LLaVA-OV-1.5-8B (base) | — | ✓ |
| MIRage (InternVL2.5-8B+MIS) | — | ✓ |

(Distribution analysis is data-side only, not model-dependent)

### Key Metrics
- **Relation distribution**: % of each type in MIS-hard (GPT-4o annotation result)
- **ASR per relation type**: per model × relation type

### Expected Findings
1. MIS-hard: tool→target ~90% (confirms distribution is narrow)
2. MIRage: ASR ≈ 0% on tool→target, but **ASR matches base model on other 3 types**
3. All 3 base VLMs show similar vulnerability pattern on non-tool→target types

### Motivates
DREAMS systematically covers all 4 relation types.

### Config
`configs/experiments/prelim/A2_pattern_coverage.yaml`

---

## A3: Synthetic-Real Distribution Gap (P4)

### Hypothesis
MIRage 在 Stable Diffusion 合成图上训练，其安全能力**无法泛化到真实世界图像**。

### Data Sources
All from MIS test sets — zero additional data needed.

| Subset | Path | N | Image Source |
|--------|------|---|-------------|
| MIS-easy | `.../mis_test/mis_easy.json` | 1,675 | AI-generated (SD 3.5) |
| MIS-hard | `.../mis_test/mis_hard.json` | 510 | AI-generated (SD 3.5) |
| **MIS-real** | `.../mis_test/mis_real.json` | **100** | Web-retrieved |

### Models
| Model | easy ASR | hard ASR | real ASR | Gap (real−easy) |
|-------|---------|---------|---------|----------------|
| InternVL3.5-8B (base) | — | — | — | — |
| Qwen3.5-9B (base) | — | — | — | — |
| LLaVA-OV-1.5-8B (base) | — | — | — | — |
| MIRage (InternVL2.5-8B+MIS) | — | — | — | — |

### Key Metric
```
Synthetic-Real Gap = ASR_real - ASR_easy
```
Per-category breakdown to control for category distribution differences.

### Expected Findings
| Subset | Base ASR (avg) | MIRage ASR |
|--------|---------------|-----------|
| mis_easy | ~80-90% | ~0-5% |
| mis_hard | ~80-90% | ~0-5% |
| **mis_real** | ~75-85% | **~15-25%** (significant degradation) |

MIRage gap ~15-20% vs. base gap ~0-5% → **gap是 safety SFT 引入的泛化问题**。

### Motivates
DREAMS 包含更高比例真实图像，提升真实场景泛化能力。

### Config
`configs/experiments/prelim/A3_synthetic_real_gap.yaml`

### Status: Ready to run (no additional data needed)
```bash
python scripts/run_prelim.py --experiment A3 --model OpenGVLab/InternVL3_5-8B
# + Qwen/Qwen3.5-9B + lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
# + re-run with MIRage checkpoint once path known
```

---

## A4: Counterfactual Safety Boundary Diagnosis (P7)

### Hypothesis
MIRage 无法可靠区分"真正危险的图像组合"与"仅改变一个图像即变安全的 counterfactual"——  
说明安全判断基于**对象共现 pattern**，而非真正的因果推理。

### Data Sources
**MSSBench** (public benchmark): natural safe/unsafe paired samples sharing same text instruction.

| Split | N | Description |
|-------|---|-------------|
| MSS-unsafe | ~_TBD_ | Same question + harmful image |
| MSS-safe | ~_TBD_ | Same question + safe image (counterfactual) |

- HF dataset: `kzhou35/mssbench`
- Download: `huggingface-cli download kzhou35/mssbench --repo-type dataset --local-dir data_links/mssbench`
- MIS-hard 50 samples as `unsafe baseline reference`.

### Models
| Model | MSS-Unsafe ASR↓ | MSS-Safe FPR↓ | Pair Consistency↑ | Visual Sensitivity↑ |
|-------|----------------|---------------|------------------|---------------------|
| InternVL3.5-8B (base) | — | — | — | — |
| Qwen3.5-9B (base) | — | — | — | — |
| LLaVA-OV-1.5-8B (base) | — | — | — | — |
| MIRage (InternVL2.5-8B+MIS) | — | — | — | — |

### Key Metrics
| Metric | Definition | Target |
|--------|-----------|--------|
| MSS-Unsafe ASR | % harmful responses on unsafe samples | Lower is better |
| **MSS-Safe FPR** | % refused safe samples (false alarm) | Lower is better |
| **Pair Consistency** | Both: unsafe→Unsafe AND safe→Safe | Higher is better |
| Visual Sensitivity | ASR_unsafe − FPR_safe | Higher = better visual discrimination |

Implementation: `harness/evaluation/metrics.py::compute_counterfactual_metrics()`

### Expected Findings (reference: MIS Table 5)
| Model | MSS-Unsafe ASR↓ | MSS-Safe FPR↓ | Consistency↑ |
|-------|----------------|---------------|-------------|
| Base (avg) | ~80% | ~10% | ~15% |
| MIRage | ~40% | ~5% | ~55% |

MIRage 的 Consistency 仍低 (~55%)，**揭示其"看对象"而非"理解语境"**。  
三种 base 模型 Consistency 均低，证明问题系统性存在。

### Motivates
DREAMS 设计 counterfactual-aware 训练 pairs，提升细粒度安全边界判断。

### Config
`configs/experiments/prelim/A4_counterfactual.yaml`

---

## Summary Table (Paper Section 2)

| Exp | Defect | Data | Models | Key Metric | Expected Finding | Motivates |
|-----|--------|------|--------|-----------|-----------------|-----------|
| **A1** | P2 文本捷径 | MIS easy+hard (text-only probe) | 3×base + MIRage | ΔASR(text-only) | all base ΔASR≈0, MIRage ΔASR≈0 | 多样化文本结构 |
| **A2** | P3 关系单一 | MIS-hard + hand probe | 3×base + MIRage | ASR per relation type | non-tool→target: MIRage ASR不降 | 多关系类型覆盖 |
| **A3** | P4 合成-真实 | MIS easy/hard/real | 3×base + MIRage | Gap=ASR_real−ASR_easy | MIRage gap>>base gap | 更多真实图 |
| **A4** | P7 无CF | MSSBench (kzhou35/mssbench) | 3×base + MIRage | Consistency, FPR | Consistency低，FPR高 | CF-aware训练 |

3×base = InternVL3.5-8B + Qwen3.5-9B + LLaVA-OV-1.5-8B

---

## Execution Order

```
A3 (can run now — base models only, no MIRage needed)
  ↓
A1 (build probes first: run_prelim.py --build-probes --experiment A1)
  ↓
A4 (download MSSBench first — blocker)
  ↓
A2 (requires hand-crafted probe JSONL + GPT-4o relation annotation)
```

**Blocking info needed**:
1. MIRage checkpoint path/HF ID (A1/A2/A3/A4 MIRage rows)
2. MSSBench HF dataset repo ID (A4)
3. COCO/OpenImages access for A2 relation probe images

**Can run immediately** (base models only):
```bash
python scripts/run_prelim.py --experiment A3 --models OpenGVLab/InternVL3_5-8B Qwen/Qwen3.5-9B lmms-lab/LLaVA-OneVision-1.5-8B-Instruct deepseek-ai/deepseek-vl2-tiny
```

**Download MSSBench** (unblocks A4):
```bash
huggingface-cli download kzhou35/mssbench --repo-type dataset --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mssbench
```
