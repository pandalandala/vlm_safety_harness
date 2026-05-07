# A Experiment Result Tables

**Models**: IVL=InternVL3.5-8B, Qwen=Qwen3.5-9B, LLaVA=LLaVA-OV-1.5-8B, MIRage=InternVL2.5-8B+MIS  
**Metrics**: ASR=Attack Success Rate, RSR=Relation-Safe Rate, RR=Refusal Rate, HR=Harmless Rate (↓=lower better, ↑=higher better)  
**MSSBench**: `kzhou35/mssbench` | **MIRage path**: TBD

---

## A1: Textual Shortcut

**One table**: full-image vs. text-only (black frame), per model per split.

| Model | Split | Cond-A ASR↓ | Cond-A RSR↑ | Cond-A RR↑ | Cond-A HR↑ | Cond-B ASR↓ | **ΔASR** |
|-------|-------|------------|------------|-----------|-----------|------------|---------|
| IVL | easy | — | — | — | — | — | — |
| IVL | hard | — | — | — | — | — | — |
| Qwen | easy | — | — | — | — | — | — |
| Qwen | hard | — | — | — | — | — | — |
| LLaVA | easy | — | — | — | — | — | — |
| LLaVA | hard | — | — | — | — | — | — |
| **MIRage** | easy | — | — | — | — | — | **—** |
| **MIRage** | hard | — | — | — | — | — | **—** |

Cond-A = full dual-image input. Cond-B = both images replaced with black 224×224 frames. ΔASR = Cond-A − Cond-B. Small ΔASR for MIRage → safety from text pattern, not vision.

---

## A2: Relation Pattern Coverage

**Separate data-stats table** (model-independent):

| Relation Type | Count in MIS-hard | % |
|--------------|------------------|---|
| tool→target | — | —% |
| before→after | — | —% |
| identity-linking | — | —% |
| context-shift | — | —% |
| other | — | —% |
| **Total** | 510 | 100% |

**Main result table**: ASR per model × relation type (one big matrix).

| Model | tool→target ASR↓ | before→after ASR↓ | identity-linking ASR↓ | context-shift ASR↓ | Overall ASR↓ | Gap† |
|-------|-----------------|------------------|----------------------|-------------------|-------------|------|
| IVL | — | — | — | — | — | — |
| Qwen | — | — | — | — | — | — |
| LLaVA | — | — | — | — | — | — |
| **MIRage** | — | — | — | — | — | **—** |

†Gap = avg(non-tool→target ASR) − tool→target ASR. Positive for MIRage → MIS training doesn't generalize beyond its dominant pattern.

---

## A3: Synthetic-Real Distribution Gap

**One table**: all models × all three splits × full metrics on real subset.

| Model | easy ASR↓ | hard ASR↓ | real ASR↓ | **Gap (real−easy)** | real RSR↑ | real RR↑ | real HR↑ |
|-------|----------|----------|----------|---------------------|----------|---------|---------|
| IVL | — | — | — | — | — | — | — |
| Qwen | — | — | — | — | — | — | — |
| LLaVA | — | — | — | — | — | — | — |
| **MIRage** | — | — | — | **—** | — | — | — |

**Per-category breakdown** (MIRage only — separate because category structure doesn't align with model rows):

| Category | easy ASR | real ASR | Gap |
|----------|---------|---------|-----|
| Weapons | — | — | — |
| Drugs | — | — | — |
| Violence | — | — | — |
| Privacy | — | — | — |
| Self-harm | — | — | — |
| Other | — | — | — |
| **Overall** | — | — | — |

---

## A4: Counterfactual Safety Boundary

**One table**: MSS metrics + MIS-hard reference ASR merged.

| Model | MSS-Unsafe ASR↓ | MSS-Safe FPR↓ | Pair Consistency↑ | Visual Sensitivity↑ | MIS-hard ref ASR↓ |
|-------|----------------|---------------|------------------|--------------------|--------------------|
| IVL | — | — | — | — | — |
| Qwen | — | — | — | — | — |
| LLaVA | — | — | — | — | — |
| **MIRage** | — | — | — | — | — |

Visual Sensitivity = MSS-Unsafe ASR − MSS-Safe FPR (higher = model better discriminates safe vs. unsafe by vision).  
Pair Consistency = P(correctly identifies unsafe pair AND safe pair) for matched pairs.  
MIS-hard ref = ASR on 50 MIS-hard samples (ground truth "should refuse").

---

## Paper Summary Table (Section 2, forward reference)

Fill after all A experiments complete. Becomes Table 2 in paper.

| Exp | MIS Defect | Key Finding (MIRage) | Key Finding (Base avg) |
|-----|-----------|---------------------|----------------------|
| A1 | P2 Textual shortcut | ΔASR = — | ΔASR = — |
| A2 | P3 Relation monoculture | tool→target ASR=—, other=— | Gap=— |
| A3 | P4 Synth-real gap | Gap=— | Gap=— |
| A4 | P7 No CF reasoning | Consistency=—, FPR=— | Consistency=—, FPR=— |

---

## Main Experiment (Forward Reference)

Full model roster for main experiment. To be updated when main exp runs.

### Open-Source — Train + Eval

| Model | HF ID | Size | Priority |
|-------|-------|------|---------|
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | 8.5B | Primary |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 9B | Primary |
| LLaVA-OV-1.5-8B | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | 8B | Primary |
| DeepSeek-VL2-Tiny | `deepseek-ai/deepseek-vl2-tiny` | 4.1B MoE | Secondary |
| MiniCPM-o-4.5 | `openbmb/MiniCPM-o-4_5` | 9B | Secondary |
| Kimi-VL-A3B | `moonshotai/Kimi-VL-A3B-Instruct` | 3B MoE | Secondary |
| GLM-4.6V-Flash | `zai-org/GLM-4.6V-Flash` | 9B | Secondary |
| Ovis2.5-9B | `AIDC-AI/Ovis2.5-9B` | 9B | Secondary |
| Gemma-4-4B | `google/gemma-4-E4B-it` | 4.5B | Secondary |
| Phi-4-multimodal | `microsoft/Phi-4-multimodal-instruct` | ~14.8B | Secondary |

### Open-Source — Eval Only

| Model | HF ID | Size | Note |
|-------|-------|------|------|
| InternVL3.5-4B | `OpenGVLab/InternVL3_5-4B` | 4B | Size ablation |
| LLaVA-OV-1.5-4B | `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` | 4B | Size ablation |
| Gemma-4-2B | `google/gemma-4-E2B-it` | 2B | Size ablation |
| idefics2-8b | `HuggingFaceM4/idefics2-8b` | 8B | HF pipeline (no vLLM) |

### Closed-Source — API Eval Only

| Model | Note |
|-------|------|
| GPT-5.5 | Upper bound reference |
| Gemini-3.1-Pro | Upper bound reference |
| Claude-Opus-4.7 | Upper bound reference |

### Excluded

| Model | Reason |
|-------|--------|
| Janus-Pro-7B | Single-image generative model |
