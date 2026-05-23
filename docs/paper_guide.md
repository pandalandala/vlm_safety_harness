# DREAMS 论文章节结构与实验命令指南

> **说明**: 本文档覆盖论文各章节的逻辑、内容、对应表格，以及每个子实验的具体运行命令（按模型拆分到最细粒度）。
> 除 LLaVA 系列命令外，其余命令均在 `mis_safety` conda 环境中运行；LLaVA 命令需区分两类：SFT 训练仍使用 `mis_safety_llava`，而实际推理 / smoke 验证（包括官方 base 与本地导出的 LLaVA checkpoint）应优先使用兼容环境 `mis_safety_llava453`。`mis_safety_llava` 仅保留给训练与旧 checkpoint 排查，因为 2026-05-19 的排查表明它会让 LLaVA inference 产生空字符串、`Question: What is the image?` 等假坏结果。工作目录统一为 `/mnt/hdd/xuran/vlm_safety_harness/`。
> **78server 分工**: 推理/测试专用清单见 `docs/main_experiments/78server_inference_only_assignment.md`。本文中的 `[78server]` 表示可直接分配给 78server，`[78server 条件]` 表示等 checkpoint/API 到位后可分配，`[训练机]` 表示不要分给 78server。
> **E5 状态**: E5 counterfactual consistency 暂时取消，不纳入当前论文主实验和执行队列；相关代码、配置、数据路径和历史说明仅归档保留，不删除。

---

## 目录

- [1 论文整体叙事链](#1-论文整体叙事链)
- [2 Section 1 — Introduction](#2-section-1--introduction)
- [3 Section 2 — Related Work](#3-section-2--related-work)
- [4 Section 3 — DREAMS Dataset](#4-section-3--dreams-dataset)
- [5 Section 4 — Preliminary Analysis (A1–A4)](#5-section-4--preliminary-analysis-a1a4)
- [6 Section 5 — Main Experiments (E1–E4; E5 archived)](#6-section-5--main-experiments-e1e4-e5-archived)
- [7 Section 6 — Ablation Studies](#7-section-6--ablation-studies)
- [8 Appendix](#8-appendix)
- [9 模型速查表](#9-模型速查表)

---

## 1 论文整体叙事链

```
MIS (ICLR 2026) 提出多图安全推理
       ↓
Section 2: 诊断 MIS 数据集的四个根本缺陷 (A1–A4)
       ↓
Section 3: 基于诊断结论，构建更优质数据集 DREAMS
       ↓
Section 5: E1–E4 主实验证明 DREAMS 在安全性和能力上均优于 MIRage；E5 反事实一致性暂时取消，仅归档保留
       ↓
Section 6: 消融实验分析关键设计决策
```

**核心对比方案（贯穿全文）**:

| 对比轴 | 我们 (DREAMS SFT) | 基线 (MIRage-data SFT) | 说明 |
|--------|------|------|------|
| 同架构 InternVL3.5-8B | InternVL3.5 + DREAMS | InternVL3.5 + mis_train.json | 公平对比，必须重训练 |
| 同架构 Qwen3.5-9B | Qwen3.5 + DREAMS | Qwen3.5 + mis_train.json | 同上 |
| 同架构 LLaVA-OV-1.5-8B | LLaVA-OV + DREAMS | LLaVA-OV + mis_train.json | 同上 |

> ⚠️ **关键约束**: MIRage 对比列必须使用 Tier A 重训练变体（`main_baseline_mirage_data_*.yaml`），
> 禁止使用 `Tuwhy/InternVL2.5-8B-MIRage` 等公开 checkpoint 作为 "DREAMS > MIRage" 的对照。

---

## 2 Section 1 — Introduction

### 2.1 内容逻辑

1. **背景**: VLM 在多图场景下存在安全漏洞（用两张图共同构造危险指令）。
2. **现有工作 (MIS)**: MIRage 是首个针对此问题的安全微调方法，但存在局限性。
3. **问题陈述**: 四句话引入四个缺陷（对应 A1–A4，但不展开）。
4. **贡献**: ① DREAMS 数据集 ② DREAMS 训练框架 ③ E1–E4 实验体系 ④ 多架构/跨 benchmark 验证。E5 counterfactual consistency 暂时取消，不作为当前贡献主线。
5. **结果预告**: DREAMS 在所有 Tier A 架构上的 ASR 降低 X%，同时通用能力不下降。

### 2.2 无需运行实验

---

## 3 Section 2 — Related Work

### 3.1 内容逻辑

1. **VLM Safety**: 单图 jailbreak（FigStep、MM-Safety 等），文本 jailbreak（AdvBench）。
2. **多图安全**: MIS 作为直接前驱，重点比较。
3. **安全微调**: RLHF / SFT 方案综述，capability-safety tradeoff 问题。
4. **反事实推理**: VLM 视觉依赖性评测相关工作。当前论文主线不再运行 E5 counterfactual consistency，仅在相关工作和归档附录中保留背景说明。

### 3.2 无需运行实验

---

## 4 Section 3 — DREAMS Dataset

### 4.1 内容逻辑

1. **构建动机**: 从 A1–A4 结论推导出 DREAMS 的四个设计原则:
   - A1 → 危险意图深藏视觉语义，去除文本模板依赖
   - A2 → 覆盖多种图像关系类型（tool→target / before→after / identity-linking / context-shift）
   - A3 → 真实图像和合成图像均衡混合
   - A4 → 视觉依赖性/过度拒绝诊断保留在 prelim；原 CF-pair 主实验扩展（E5）暂时取消并归档

2. **数据统计**（对应 Table 1）。
3. **与 MIS 比较**（对应 Table 2）。
4. **标注流程**: Qwen3.5-122B-A10B 生成响应 → GPT-4o 质量过滤。

### 4.2 对应表格

**Table 1 — DREAMS 数据集统计**

| 字段 | DREAMS | MIS |
|------|--------|-----|
| 总样本数 | 17,022 | 5,622 |
| 每条图像数 | 2 | 2 |
| 危害类别数 | 12 | 12 |
| Synth 图比例 | ~50% | ~100% |
| Real 图比例 | ~50% | ~0% |
| CF 对比例 | 100% | 0% |
| Train / Test 划分 | ~15K / ~2K | 3,927 / 2,185 |

**Table 2 — DREAMS vs MIS 关键差异对比**

| 维度 | MIS | DREAMS |
|------|-----|--------|
| 图像来源 | 纯 AI 合成 (SD 3.5) | Synth + Real 混合 |
| 关系类型覆盖 | 主要 tool→target | 4 类关系均衡 |
| 文本模板依赖 | 固定句式 | 多样化指令 |
| CF pair 支持 | 无 | 归档保留：当前不作为主实验要求 |

### 4.3 相关命令

- [x] **生成训练集响应标注**（一次性，需要 Qwen3.5-122B 或等价大模型）

```bash
conda activate mis_safety && python scripts/generate_responses.py --input /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/train.json --output /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/train_annotated.json --backend vllm --model Qwen/Qwen3.5-122B-A10B --resume --gpu-ids 0,1,2,3
```

> **[归档保留 / 当前不运行]** 构建 E5 反事实对（需要 benign image 目录）；E5 counterfactual consistency 暂时取消，以下命令仅保留给未来恢复实验时参考：
>
> ```bash
> conda activate mis_safety && python scripts/build_cf_pairs.py --test-json /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json --benign-pool /path/to/openimages_benign_pool --output /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test_cf.json --cf-images-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/cf_images --swap-idx 2 --seed 0
> ```

---

## 5 Section 4 — Preliminary Analysis (A1–A4)

> 使用 MIS 已有数据和公开基准，**不使用 DREAMS 数据**，不使用自定义训练模型。
> 四个实验共同揭示 MIRage 的结构性缺陷，为 DREAMS 数据集设计提供实证依据。
> **[78server]** A1-A4 都是 probe/推理/评测/标注任务，不需要 SFT 训练，可分配给 78server。

### 5.1 A1 — 文本捷径诊断 (Textual Shortcut)

**核心问题**: MIRage 的安全行为来自文本模板识别，还是视觉推理？

**设计**: 将两张输入图像替换为全黑帧，文本不变，观察 ASR 变化。

**关键指标**:
```
ΔASR = ASR(全图输入) - ASR(纯文本/黑帧输入)
```
ΔASR ≈ 0 for MIRage → 安全来自文本，非视觉。

**对应表格** (Table 3 / Table in Section 4.1):

| 模型 | Condition A ASR (全图) | Condition B ASR (黑帧) | ΔASR |
|------|----------------------|----------------------|------|
| InternVL3.5-8B (base) | ~85% | ~80% | ~5% |
| Qwen3.5-9B (base) | ~85% | ~80% | ~5% |
| LLaVA-OV-1.5-8B (base) | ~85% | ~80% | ~5% |
| **InternVL3.5 + MIRage-data** | ~3% | **~3%** | **≈0%** ← 关键发现 |
| **Qwen3.5 + MIRage-data** | ~3% | ~3% | ≈0% |
| **LLaVA-OV + MIRage-data** | ~3% | ~3% | ≈0% |

**命令 — 构建 Probe**:

- [x] **构建黑帧 probe**（一次性；默认写入 results/prelim/probes/）

```bash
conda activate mis_safety && python scripts/run_prelim.py --build-probes --build-probes-experiment A1
```

**命令 — 推理（按模型逐条运行）**:

> A1 config 已同时包含 original-image 与 black-frame benchmark。

- [x] **InternVL3.5-8B base — A1 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A1 --skip-eval --models OpenGVLab/InternVL3_5-8B --cuda-visible-devices 0
```

- [x] **Qwen3.5-9B base — A1 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A1 --skip-eval --models Qwen/Qwen3.5-9B --cuda-visible-devices 0
```

- [ ] **LLaVA-OV-1.5-8B base — A1 推理** （0）

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A1 --skip-eval --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct --cuda-visible-devices 0
```

- [x] **InternVL3.5 + MIRage-data — A1 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A1 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 --cuda-visible-devices 0
```

- [x] **Qwen3.5 + MIRage-data — A1 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A1 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — A1 推理** （1）

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A1 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov --cuda-visible-devices 0
```

**Step 2 — 评分（全部 A1 responses 就绪后，调 GPT-4o API）**:

- [ ] **A1 全部 run 批量评分**（自动遍历最新 run）

```bash
ROOT=/mnt/hdd/xuran/vlm_safety_harness
for cfg in A1_textual_shortcut; do
    for run_dir in $(ls -td "${ROOT}/results/prelim/${cfg}/"*/ 2>/dev/null); do
        echo "=== evaluating: ${run_dir} ==="
        conda activate mis_safety && python scripts/run_eval_only.py --responses "${run_dir}responses/" --evaluator-type auto
    done
done
```

---

### 5.2 A2 — 关系类型覆盖诊断 (Relation Pattern Coverage)

**核心问题**: MIS 数据集关系模式单一，MIRage 对非 tool→target 关系类型无安全能力。

**设计**:
1. 实地核查 MIS-hard 510 条样本的关系类型：经逐条检查，全部 510 条均为 `tool_target` 模式（图1=工具/手段，图2=目标）。MIS 构造管线的句式模板决定了这一结构性限制（见 `docs/.claude/MIS_shortcomes_final.md` §三）。
2. 为覆盖另外三种关系类型，从 DREAMS 数据集中用 GPT-4o 筛选样本：遍历 DREAMS `scored.json`（按 vlm_score 降序），分类每条样本的图像关系类型，直到 `before_after` / `identity_linking` / `context_shift` 各≥50 条。
3. 最终 Probe = MIS-hard 前50条（`tool_target`）+ DREAMS 筛选样本（三类各≥50条），共≥200条。

**对应表格** (Table 4 / Figure in Section 4.2):

| 关系类型 | Probe 样本来源 | 样本数 | Base ASR (预期) | MIRage-data SFT ASR (预期) |
|---------|-------------|-------|----------------|--------------------------|
| tool→target | MIS-hard 前50条 | 50 | ~85% | ~3% |
| before→after | DREAMS（GPT-4o筛选）| ≥50 | ~80% | ~50% ← gap |
| identity-linking | DREAMS（GPT-4o筛选）| ≥50 | ~80% | ~45% ← gap |
| context-shift | DREAMS（GPT-4o筛选）| ≥50 | ~80% | ~40% ← gap |

**命令 — Probe 构建（一次性，两步）**:

- [x] **Step 1：从 DREAMS 筛选非 tool_target 样本**（写入 results/prelim/probes/extra_relation_probe.jsonl）

```bash
conda activate mis_safety &&  python scripts/build_a2_dreams_probe.py --resume --skip-n 5000 --max-candidates 12022 --min-per-type 50 --model gpt-5.4-nano-2026-03-17 --use-batch --batch-max-bytes 52428800

```

> `build_a2_dreams_probe.py` 遍历 DREAMS scored.json（默认最多5000条），每条调 GPT-4o 分类关系类型，三类各满50条后自动停止。需要 `OPENAI_API_KEY` 环境变量。dry-run 验证：`python scripts/build_a2_dreams_probe.py --dry-run --max-candidates 5`

- [ ] **Step 2：合并 MIS-hard + DREAMS，重建 relation_type_probe.json**

```bash
conda activate mis_safety && python scripts/run_prelim.py --build-probes --build-probes-experiment A2
```

**命令 — 推理（按模型逐条运行）**:

- [ ] **InternVL3.5-8B base — A2 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A2 --skip-eval --models OpenGVLab/InternVL3_5-8B --cuda-visible-devices 0
```

- [ ] **Qwen3.5-9B base — A2 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A2 --skip-eval --models Qwen/Qwen3.5-9B --cuda-visible-devices 0
```

- [ ] **LLaVA-OV-1.5-8B base — A2 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A2 --skip-eval --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data — A2 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A2 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data — A2 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A2 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — A2 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A2 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov --cuda-visible-devices 0
```

---

### 5.3 A3 — 合成-真实分布差距 (Synthetic-Real Gap)

**核心问题**: MIRage 在 AI 合成图上训练，真实图像上 ASR 显著回升。

**设计**: 三个 MIS test split（easy=AI合成 / hard=AI合成 / real=真实网络图）直接对比。

**关键指标**:
```
Synth-Real Gap = ASR_real - ASR_easy
```

**对应表格** (Table 5 / Section 4.3):

| 模型 | mis_easy ASR | mis_hard ASR | mis_real ASR | Gap (real−easy) |
|------|------------|------------|------------|----------------|
| InternVL3.5-8B (base) | ~85% | ~85% | ~80% | ~−5% |
| Qwen3.5-9B (base) | ~85% | ~85% | ~80% | ~−5% |
| LLaVA-OV-1.5-8B (base) | ~85% | ~85% | ~80% | ~−5% |
| **InternVL3.5 + MIRage-data** | ~3% | ~5% | **~20%** | **+17%** ← 关键 |
| **Qwen3.5 + MIRage-data** | ~3% | ~5% | ~20% | +17% |
| **LLaVA-OV + MIRage-data** | ~3% | ~5% | ~20% | +17% |

**命令 — 推理（按模型逐条运行）**:

> A3 config 已包含 mis_easy / mis_hard / mis_real，一次 run 覆盖全部三个 split。

- [x] **InternVL3.5-8B base — A3 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A3 --skip-eval --models OpenGVLab/InternVL3_5-8B --cuda-visible-devices 0
```

- [x] **Qwen3.5-9B base — A3 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A3 --skip-eval --models Qwen/Qwen3.5-9B --cuda-visible-devices 0
```

- [ ] **LLaVA-OV-1.5-8B base — A3 推理**（2）

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A3 --skip-eval --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct --cuda-visible-devices 0
```

- [x] **InternVL3.5 + MIRage-data — A3 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A3 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 --cuda-visible-devices 0
```

- [x] **Qwen3.5 + MIRage-data — A3 推理**

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A3 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — A3 推理**（3）

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A3 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov --cuda-visible-devices 0
```

---

### 5.4 A4 — MSSBench FPR 与视觉依赖诊断

**核心问题**: MIRage 面对安全样本（CF pair）时过度拒绝，视觉依赖性不足。

> **注**: A4 现在保留为 prelim 中的 MSSBench FPR（过度拒绝率）诊断，不再上升为当前主实验 E5。E5/PD/PC/VS 方案暂时取消并归档，原因是与黑图/视觉依赖诊断思路重叠，当前不纳入论文主线。
> 此处仅运行 MSSBench 上的 FPR（过度拒绝率）分析，作为动机陈述。

**对应表格** (Table 6 / Section 4.4):

| 模型 | MSSBench FPR ↓ (safe 样本被拒率) |
|------|--------------------------------|
| InternVL3.5-8B (base) | ~5% |
| Qwen3.5-9B (base) | ~5% |
| LLaVA-OV-1.5-8B (base) | ~5% |
| **InternVL3.5 + MIRage-data** | **~30%** ← 关键 |
| **Qwen3.5 + MIRage-data** | ~30% |
| **LLaVA-OV + MIRage-data** | ~30% |

**命令 — 推理（按模型逐条运行）**:

> A4 config 已包含 mssbench_safe / mssbench_unsafe / mis_hard，一次 run 覆盖全部三个 split。

- [ ] **InternVL3.5-8B base — A4 推理**（4）

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A4 --skip-eval --models OpenGVLab/InternVL3_5-8B --cuda-visible-devices 0
```

- [ ] **Qwen3.5-9B base — A4 推理**（5）

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A4 --skip-eval --models Qwen/Qwen3.5-9B --cuda-visible-devices 0
```

- [ ] **LLaVA-OV-1.5-8B base — A4 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A4 --skip-eval --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data — A4 推理**（6）

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A4 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data — A4 推理**（7）

```bash
conda activate mis_safety && python scripts/run_prelim.py --experiment A4 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — A4 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment A4 --skip-eval --models /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov --cuda-visible-devices 0
```

---

## 6 Section 5 — Main Experiments (E1–E4; E5 archived)

> **前置条件**: 需要先完成以下准备工作：
> 1. DREAMS train_annotated.json 已生成（`generate_responses.py`）
> 2. Tier A 6 个 SFT checkpoint 已训练（3 DREAMS + 3 MIRage-data）
> 3. Tier B 8 个基线模型已下载
> 4. E5 counterfactual consistency 暂时取消；`test_cf.json` / `build_cf_pairs.py` 仅作为归档能力保留，当前主实验不需要生成。

### 6.1 E1 — DREAMS 分布内安全（Headline，Section 5.1）

**核心问题**: DREAMS 训练的模型是否在 DREAMS 测试集上显著降低 ASR？explicit/implicit 两类危害均有效？

**当前执行状态**:
- 审计清单见 [`6_1_e1_inference_audit.md`](./6_1_e1_inference_audit.md)
- 仍待真实执行的命令见 [`6_1_e1_pending_commands.md`](./6_1_e1_pending_commands.md)

**指标**: ASR ↓ / PR ↑ / BR ↓ / FPR ↓，按 `harm_type∈{explicit, implicit}` 分层 → 8 列。

**对应表格** (Table 7 — Main Results, Section 5.1):

| 模型 | Expl ASR↓ | Expl PR↑ | Expl BR↓ | Expl FPR↓ | Impl ASR↓ | Impl PR↑ | Impl BR↓ | Impl FPR↓ |
|------|----------|---------|---------|---------|----------|---------|---------|---------|
| **— Tier A base (无 SFT) —** | | | | | | | | |
| InternVL3.5-8B | | | | | | | | |
| Qwen3.5-9B | | | | | | | | |
| LLaVA-OV-1.5-8B | | | | | | | | |
| **— Tier B 7-9B —** | | | | | | | | |
| Kimi-VL-A3B | | | | | | | | |
| MiniCPM-o-4.5 | | | | | | | | |
| Gemma-4-E4B | | | | | | | | |
| GLM-4.6V-Flash | | | | | | | | |
| **— Tier B 4B —** | | | | | | | | |
| Qwen3.5-4B | | | | | | | | |
| InternVL3.5-4B | | | | | | | | |
| LLaVA-OV-1.5-4B | | | | | | | | |
| DeepSeek-VL2-Tiny | | | | | | | | |
| **— Tier C 闭源上限 —** | | | | | | | | |
| GPT-5.5 | | | | | | | | |
| Gemini-3.1-Pro | | | | | | | | |
| Claude-Opus-4.7 | | | | | | | | |
| **— Tier A MIRage-data SFT (同架构对照) —** | | | | | | | | |
| InternVL3.5 + MIRage-data | | | | | | | | |
| Qwen3.5 + MIRage-data | | | | | | | | |
| LLaVA-OV + MIRage-data | | | | | | | | |
| **— Ours (DREAMS SFT) —** | | | | | | | | |
| **InternVL3.5 + DREAMS** | | | | | | | | |
| **Qwen3.5 + DREAMS** | | | | | | | | |
| **LLaVA-OV + DREAMS** | | | | | | | | |

#### 6.1.1 Tier A SFT 训练命令

> **[训练机]** 本小节会触发 DREAMS/MIRage-data SFT 训练，不分配给 78server。
>
> **LLaVA 训练注意**: 2026-05-19 的排查已确认，直接从原始 Hub 路径
> `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` 开始训练时，当前
> `LLaVAOneVision1_5` remote code 会把整套 `model.language_model.*`
> 语言塔标成 `MISSING` 并重新初始化，导致后续 checkpoint 生成异常。
> 训练配置现已改为本地兼容基座
> `/mnt/hdd/xuran/vlm_safety_harness/models/llava_ov_1_5_8b_base_hfcompat`；
> 这个问题同时影响 `main_dreams_llava_ov` 与
> `main_baseline_mirage_data_llava_ov` 的历史训练 run；如需重训这两条线，
> 不要再手动覆盖回原始 Hub 路径。

**DREAMS SFT（按架构逐条运行）**:

- [x] **InternVL3.5-8B + DREAMS SFT**（训练 + 推理；GPT 评测由 78server 单独跑）

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [x] **Qwen3.5-9B + DREAMS SFT**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [x] **LLaVA-OV-1.5-8B + DREAMS SFT**（仅训练；推理必须单独在 `mis_safety_llava453` 跑）

> LLaVA-OV-1.5 训练需 `mis_safety_llava`（transformers dev），推理需 `mis_safety_llava453`
> （transformers 4.57.6）。两 env 互斥，故训练命令加 `--skip-inference`，推理见下方 E1 推理小节。

```bash
conda activate mis_safety_llava && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-eval --skip-inference --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

**MIRage-data SFT（按架构逐条运行，用于 E2/E3 对比；E5 暂停不纳入当前对比）**:

- [x] **InternVL3.5-8B + MIRage-data SFT**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [x] **Qwen3.5-9B + MIRage-data SFT**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [x] **LLaVA-OV-1.5-8B + MIRage-data SFT**（仅训练；推理单独在 `mis_safety_llava453` 跑）

```bash
conda activate mis_safety_llava && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-eval --skip-inference --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

> **⚠️ 训练完成后必须将 checkpoint 放到 `models/` 目录，78server 的 `--skip-train --model-path` 才能正确加载。**  
> 详见 [`models/MODELS.md`](../models/MODELS.md)（含 symlink 用法）。

**Checkpoint 放置（训练机执行，自动查找最新 run，无需手动输入时间戳）**：

```bash
# 自动符号链接 6 个 SFT checkpoint 到 models/ 目录
ROOT=/mnt/hdd/xuran/vlm_safety_harness

link_latest() {
    local cfg=$1 dst=$2
    local target link
    # New layout: results/main/{cfg}/{model_tag}/checkpoint — pick newest checkpoint by mtime.
    target=$(ls -td "${ROOT}/results/main/${cfg}/"*/checkpoint 2>/dev/null | head -1)
    [ -z "$target" ] && { echo "[skip] no checkpoint found: ${cfg}"; return 1; }
    target=$(realpath "$target")
    link="${ROOT}/models/${dst}"
    if [ -d "$link" ] && [ ! -L "$link" ]; then
        if find "$link" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            echo "[skip] destination directory is not empty: ${link}"
            return 1
        fi
        rmdir "$link"
    elif [ -e "$link" ] || [ -L "$link" ]; then
        rm -f "$link"
    fi
    ln -s "$target" "$link"
    echo "[ok] models/${dst} → ${target}"
}

link_latest main_dreams_internvl3_5               dreams_internvl3_5
link_latest main_dreams_qwen3_5                   dreams_qwen3_5
link_latest main_dreams_llava_ov                  dreams_llava_ov
link_latest main_baseline_mirage_data_internvl3_5 mirage_data_internvl3_5
link_latest main_baseline_mirage_data_qwen3_5     mirage_data_qwen3_5
link_latest main_baseline_mirage_data_llava_ov    mirage_data_llava_ov
```

#### 6.1.1b Tier A SFT 推理命令（78server，checkpoint 就绪后）

> **[78server]** `models/` 目录 checkpoint 放置完成后执行。  
> Step 1 生成 responses（不调 API）；Step 2 评分（调 GPT-4o API）。  
> E1 输出指标：`Expl ASR↓  Expl PR↑  Expl BR↓  Expl FPR↓  Impl ASR↓  Impl PR↑  Impl BR↓  Impl FPR↓`

**Step 1 — DREAMS SFT 推理**：

> **当前状态（2026-05-19）**:
> `InternVL3.5 + DREAMS` 与 `Qwen3.5 + DREAMS` 的 smoke 推理已通过。
> `LLaVA-OV + DREAMS` 现有 checkpoint 仍不可用，根因不是推理环境，而是其训练 run
> 在加载 base 时把整套 `model.language_model.*` 标成了 `MISSING` 并重新初始化；
> 即使把该 canonical checkpoint 放到正确的 `mis_safety_llava453` 推理环境里，
> 也仍会产出病态长文本而非正常安全回复；
> 需要基于本地 `llava_ov_1_5_8b_base_hfcompat` 重新训练后，才值得重新跑本节命令。

- [x] **InternVL3.5 + DREAMS 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5 \
    --cuda-visible-devices 0
```

- [x] **Qwen3.5 + DREAMS 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_qwen3_5 \
    --cuda-visible-devices 0
```

- [x] **LLaVA-OV + DREAMS 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — DREAMS SFT 评分**：

- [ ] **InternVL3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_internvl3_5/dreams_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_qwen3_5/dreams_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + DREAMS 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_llava_ov/dreams_llava_ov/responses/ \
    --evaluator-type auto
```

**Step 1 — MIRage-data SFT 推理（E1 对照基线）**：

> **当前状态（2026-05-19）**:
> `InternVL3.5 + MIRage-data` 与 `Qwen3.5 + MIRage-data` 的 smoke 推理已通过。
> `LLaVA-OV + MIRage-data` 的 canonical 推理路径已切到
> `models/mirage_data_llava_ov_hfcompat`，解决了旧 checkpoint 的
> `model.layers.*` / `visual.*` 命名空间错配；但历史训练日志同样表明，
> 旧训练 run 在加载官方 base 时把整套 `model.language_model.*` 新初始化了，
> 因此如果要得到真正可靠的 SFT 模型，仍应基于
> `llava_ov_1_5_8b_base_hfcompat` 重新训练。就本节 Step 1 推理命令而言，
> 该 canonical 路径现已在 `mis_safety_llava453` 下 smoke 通过：无
> `[INFERENCE_ERROR]`，response 为正常的安全拒答文本。
> 另外，LLaVA-OV 的 smoke 推理请使用 `mis_safety_llava453`，不要再用
> `mis_safety_llava453` 直接判定 response 质量；后者在 2026-05-19 的对照中
> 会让 base 和本地 checkpoint 都出现空字符串、`Question: What is the image?`
> 等假坏结果。

- [x] **InternVL3.5 + MIRage-data 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 \
    --cuda-visible-devices 0
```

- [x] **Qwen3.5 + MIRage-data 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 \
    --cuda-visible-devices 0
```

- [x] **LLaVA-OV + MIRage-data 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — MIRage-data SFT 评分**：

- [ ] **InternVL3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_internvl3_5/mirage_data_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_qwen3_5/mirage_data_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + MIRage-data 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_llava_ov/mirage_data_llava_ov/responses/ \
    --evaluator-type auto
```

#### 6.1.2 Tier A Base 推理命令（跳过训练）

> **[78server]** 全部带 `--skip-train`，可分配给 78server。  
> Step 1 生成 responses（不调 API）；Step 2 评分（调 GPT-4o API）。  
> E1 输出指标：`Expl ASR↓  Expl PR↑  Expl BR↓  Expl FPR↓  Impl ASR↓  Impl PR↑  Impl BR↓  Impl FPR↓`
>
> **LLaVA base 环境**: 2026-05-19 的 smoke 结果表明，官方
> `LLaVA-OneVision-1.5-8B/4B-Instruct` 应使用 `mis_safety_llava453`
> 推理；旧环境 `mis_safety_llava453` 会触发更高版本 Transformers / flash-attn
> 兼容问题。

**Step 1 — Tier A Base 推理**：

- [x] **InternVL3.5-8B base 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path OpenGVLab/InternVL3_5-8B-HF \
    --cuda-visible-devices 0
```

- [x] **Qwen3.5-9B base 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path Qwen/Qwen3.5-9B \
    --cuda-visible-devices 0
```

- [x] **LLaVA-OV-1.5-8B base 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
    --cuda-visible-devices 0
```

**Step 2 — Tier A Base 评分**：

- [ ] **InternVL3.5-8B base 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_internvl3_5/InternVL3_5-8B-HF/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5-9B base 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_qwen3_5/Qwen3.5-9B/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV-1.5-8B base 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_llava_ov/LLaVA-OneVision-1.5-8B-Instruct/responses/ \
    --evaluator-type auto
```

#### 6.1.3 Tier B 推理命令（按模型逐条运行，无 SFT）

> **[78server]** Tier B 均为无 SFT 推理，可分配给 78server。  
> Step 1 生成 responses（不调 API）；Step 2 评分（调 GPT-4o API）。  
> E1 输出指标：`Expl ASR↓  Expl PR↑  Expl BR↓  Expl FPR↓  Impl ASR↓  Impl PR↑  Impl BR↓  Impl FPR↓`

**Step 1 — Tier B 推理**：

- [x] **Kimi-VL-A3B 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_kimi_vl_a3b.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **MiniCPM-o-4.5 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_minicpm_o_4_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **Gemma-4-E4B 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_gemma_4_e4b.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **GLM-4.6V-Flash 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_glm_4_6v_flash.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **Qwen3.5-4B 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_qwen3_5_4b.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **InternVL3.5-4B 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_internvl3_5_4b.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **LLaVA-OV-1.5-4B 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_llava_ov_1_5_4b.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

- [x] **DeepSeek-VL2-Tiny 推理**（原生 vLLM，无需训练）

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_deepseek_vl2_tiny.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --cuda-visible-devices 0
```

**Step 2 — Tier B 评分**：

- [ ] **Kimi-VL-A3B 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_kimi_vl_a3b/Kimi-VL-A3B-Instruct/responses/ \
    --evaluator-type auto
```

- [ ] **MiniCPM-o-4.5 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_minicpm_o_4_5/MiniCPM-o-4_5/responses/ \
    --evaluator-type auto
```

- [ ] **Gemma-4-E4B 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_gemma_4_e4b/gemma-4-E4B-it/responses/ \
    --evaluator-type auto
```

- [ ] **GLM-4.6V-Flash 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_glm_4_6v_flash/GLM-4.6V-Flash/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5-4B 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_qwen3_5_4b/Qwen3.5-4B/responses/ \
    --evaluator-type auto
```

- [ ] **InternVL3.5-4B 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_internvl3_5_4b/InternVL3_5-4B-HF/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV-1.5-4B 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_llava_ov_1_5_4b/LLaVA-OneVision-1.5-4B-Instruct/responses/ \
    --evaluator-type auto
```

- [ ] **DeepSeek-VL2-Tiny 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_deepseek_vl2_tiny/deepseek-vl2-tiny/responses/ \
    --evaluator-type auto
```

#### 6.1.4 Tier C 闭源推理命令（按模型逐条运行）

> **[78server 条件]** 可分配给 78server，前提是 API key 和闭源模型配置已就绪。

- [ ] **GPT-5.5**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models gpt-5.5 --benchmarks our_test --output-root results/main/closed_source
```

- [ ] **Gemini-3.1-Pro**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models gemini-3.1-pro --benchmarks our_test --output-root results/main/closed_source
```

- [ ] **Claude-Opus-4.7**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models claude-opus-4.7 --benchmarks our_test --output-root results/main/closed_source
```

#### 6.1.5 一键批量运行（队列模式）

> **[训练机/谨慎]** 当前批量命令可能包含训练队列。78server 只运行明确带 `--skip-train` 的单模型推理命令，或确认 orchestrator 不会触发训练后再用。

- [ ] **Tier A DREAMS + E1 完整队列**

```bash
conda activate mis_safety && python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams --cuda-visible-devices 0
```

- [ ] **全队列（含 Tier B）**

```bash
conda activate mis_safety && python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams,tier_a_mirage_data,tier_a_base,tier_b --cuda-visible-devices 0
```

---

### 6.2 E2 — 合成/真实/混合图像分层（Section 5.2，A3 验证）

> **[78server]** E2 复用 E1 输出，只重算切片 metrics，不需要训练。

**核心问题**: DREAMS 训练是否解决了 MIRage 在真实图像上的泛化缺口？

**设计**: E1 同一次推理输出，按 `img_source_type∈{synth, real, mix}` 重新切片。**不需要重新推理**，只需重算 metrics。

**对应表格** (Table 8 / Section 5.2): 4 指标 × 3 图像类型 = 12 列，24 行（含 MIRage-data Tier A）。

**模型行**（比 E1 多出 Tier A MIRage-data 3行）:

| 模型 | Synth ASR↓ | Synth PR↑ | Synth BR↓ | Synth FPR↓ | Real ASR↓ | Real PR↑ | Real BR↓ | Real FPR↓ | Mix ASR↓ | Mix PR↑ | Mix BR↓ | Mix FPR↓ |
|------|---|---|---|---|---|---|---|---|---|---|---|---|
| Tier A base (3 行) | | | | | | | | | | | | |
| Tier A MIRage-data (3 行) | | | | | | | | | | | | |
| Tier B (8 行) | | | | | | | | | | | | |
| Tier C (3 行) | | | | | | | | | | | | |
| **Ours DREAMS (3 行)** | | | | | | | | | | | | |

**命令 — 重算 E2 切片 metrics（E1 推理完成后执行）**:

- [ ] **E2 全队列切片重算**（E2 不需要重新推理，重用 E1 输出；orchestrator 只做 metric 切片）

```bash
conda activate mis_safety && python scripts/run_main.py --experiment-id E2 --cohort tier_a_dreams,tier_a_mirage_data,tier_a_base,tier_b --skip-train --cuda-visible-devices 0
```

- [ ] **E2 单模型切片重算**（示例 InternVL3.5 DREAMS；确定性 model_tag 路径）

```bash
RESP=/mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_internvl3_5/dreams_internvl3_5/
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses "${RESP}responses/" \
    --output-dir "${RESP}eval_results" \
    --evaluator-type auto
```

---

### 6.3 E3 — 跨 Benchmark 安全泛化（Section 5.3）

> **[78server 条件]** E3 可交给 78server 跑跨 benchmark 推理，但 Tier A DREAMS/MIRage-data 行必须等训练机产出 checkpoint 后执行，并且命令必须保留 `--skip-train`。

**核心问题**: DREAMS 的安全提升是否泛化到独立 benchmark，而非过拟合到 DREAMS 测试集？

**指标**: 每个 benchmark 使用其自身的 canonical metric（AdvBench 用 ASR，FigStep 用通过率等）。

**对应表格** (Table 9 / Section 5.3):

| 模型 | AdvBench ASR↓ | SafeBench ASR↓ | FigStep Pass↓ | MM-Safety ASR↓ | JailbreakV ASR↓ | SIUO ↓ | MSS FPR↓ |
|------|---|---|---|---|---|---|---|
| Tier A base (3 行) | | | | | | | |
| Tier A MIRage-data (3 行) | | | | | | | |
| Tier B (8 行) | | | | | | | |
| Tier C (3 行) | | | | | | | |
| **Ours DREAMS (3 行)** | | | | | | | |

**命令 — 按 benchmark 逐条运行**:

**Step 1 — E3 AdvBench 推理**：

- [ ] **InternVL3.5 + DREAMS — AdvBench 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + DREAMS — AdvBench 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + DREAMS — AdvBench 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_llava_ov \
    --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data — AdvBench 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data — AdvBench 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — AdvBench 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks advbench \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — E3 AdvBench 评分**（推理完成后立即运行）：

- [ ] **InternVL3.5 + DREAMS — AdvBench 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_internvl3_5/dreams_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + DREAMS — AdvBench 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_qwen3_5/dreams_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + DREAMS — AdvBench 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_llava_ov/dreams_llava_ov/responses/ \
    --evaluator-type auto
```

- [ ] **InternVL3.5 + MIRage-data — AdvBench 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_internvl3_5/mirage_data_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + MIRage-data — AdvBench 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_qwen3_5/mirage_data_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + MIRage-data — AdvBench 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_llava_ov/mirage_data_llava_ov/responses/ \
    --evaluator-type auto
```

**Step 1 — E3 MM-SafetyBench 推理**：

- [ ] **InternVL3.5 + DREAMS — MM-Safety 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + DREAMS — MM-Safety 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + DREAMS — MM-Safety 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_llava_ov \
    --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data — MM-Safety 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data — MM-Safety 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — MM-Safety 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval --benchmarks mm_safety \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — E3 MM-SafetyBench 评分**（推理完成后立即运行）：

- [ ] **InternVL3.5 + DREAMS — MM-Safety 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_internvl3_5/dreams_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + DREAMS — MM-Safety 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_qwen3_5/dreams_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + DREAMS — MM-Safety 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_llava_ov/dreams_llava_ov/responses/ \
    --evaluator-type auto
```

- [ ] **InternVL3.5 + MIRage-data — MM-Safety 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_internvl3_5/mirage_data_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + MIRage-data — MM-Safety 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_qwen3_5/mirage_data_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + MIRage-data — MM-Safety 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_llava_ov/mirage_data_llava_ov/responses/ \
    --evaluator-type auto
```

> ⚠️ **注意**：E3 每个模型的 responses 落在确定性目录 `results/main/E3/{config}/{model_tag}/responses/`，同一 (config, model) 重跑会原地覆盖。不同 benchmark 写入同一 responses 目录下不同 `{benchmark}.jsonl`，互不干扰；评分命令直接指向该确定性目录即可，无需 `ls -td` 猜测。

**全部 E3 benchmarks 批量运行（推荐，一次跑完全部7个benchmark）**：

- [ ] **E3 全队列批量推理**（不调 API）

```bash
conda activate mis_safety && python scripts/run_main.py --experiment-id E3 --cohort tier_a_dreams,tier_a_mirage_data,tier_b \
    --skip-eval --cuda-visible-devices 0
```

**Tier C 闭源推理命令（当前仅支持 `our_test / mis_easy / mis_hard / mis_real`）**:

> **[78server 条件]** 如果要跑 E3 的完整 benchmark 集，需要先扩展 `run_closed_source.py` 里的 `load_benchmark_records()`。

- [ ] **GPT-5.5 — our_test**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models gpt-5.5 --benchmarks our_test --output-root results/main/closed_source
```

- [ ] **Gemini-3.1-Pro — our_test**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models gemini-3.1-pro --benchmarks our_test --output-root results/main/closed_source
```

- [ ] **Claude-Opus-4.7 — our_test**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models claude-opus-4.7 --benchmarks our_test --output-root results/main/closed_source
```

---

### 6.4 E4 — 通用能力保留（Section 5.4）

> **[78server 部分]** 只有 V0 base 能直接给 78server。V1/V2/V3/V4 是训练变体，除非已确认脚本只复用现有 checkpoint，否则不要分配给 78server。

**核心问题**: DREAMS SFT 是否保留通用多模态能力？与 MIRage-data SFT 相比有无损失？

**设计**: 同一个 base 架构（InternVL3.5-8B，主选）训练 5 种 SFT 变体（V0–V4），在 5 个通用 benchmark 测试准确率。

**变体定义**:
| ID | 变体 | 训练数据 | 状态 |
|----|------|---------|------|
| V0 | base（无 SFT） | — | 可运行 |
| V1 | + MIRage-data | mis_train.json | 可运行 |
| V2 | + MIRage-data + General | mis_train.json + 通用数据 | **BLOCKED**（待指定通用数据） |
| V3 | + DREAMS | train_annotated.json | 可运行 |
| V4 | + DREAMS + General | train_annotated.json + 通用数据 | **BLOCKED** |

**对应表格** (Table 10 / Section 5.4):

| 变体 | MMStar ↑ | MMMU ↑ | MuirBench ↑ | BLink ↑ | MMT ↑ |
|------|---------|--------|------------|--------|-------|
| InternVL3.5 V0 (base) | | | | | |
| V1 (+ MIRage) | | | ← 预计下降 | | |
| V2 (+ MIRage + General) | | | ← 部分恢复 | | |
| **V3 (+ DREAMS)** | | | **← 预计 ≈ V0** | | |
| **V4 (+ DREAMS + General)** | | | **← 预计 ≥ V0** | | |

**命令 — E4 变体训练（按变体逐条运行）**:

- [ ] **V0: base（仅推理，无训练）**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V0 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0
```

- [ ] **V1: InternVL3.5 + MIRage-data**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V1 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0,1,2,3
```

- [ ] **V3: InternVL3.5 + DREAMS**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V3 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0,1,2,3
```

- [ ] **V2: InternVL3.5 + MIRage-data + 500 M4-Instruct**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V2 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0,1,2,3
```

- [ ] **V4: InternVL3.5 + DREAMS + M4-Instruct at 11% final-data ratio**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V4 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0,1,2,3
```

- [ ] **下载 M4-Instruct 通用数据源**（V2/V4 依赖，若本地未就绪）

```bash
conda activate mis_safety && huggingface-cli download lmms-lab/M4-Instruct --repo-type dataset --local-dir /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
```

**命令 — 按 benchmark 逐条运行 E4（更细粒度）**:

- [ ] **V0 — 仅 MMStar**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V0 --baseline internvl3_5 --benchmarks mmstar --cuda-visible-devices 0
```

- [ ] **V0 — 仅 MuirBench**（最关键，DREAMS 是多图数据）

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V0 --baseline internvl3_5 --benchmarks muirbench --cuda-visible-devices 0
```

- [ ] **V3 — 仅 MuirBench**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V3 --baseline internvl3_5 --benchmarks muirbench --cuda-visible-devices 0,1,2,3
```

- [ ] **V1 — 仅 MuirBench**（对比 V3）

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V1 --baseline internvl3_5 --benchmarks muirbench --cuda-visible-devices 0,1,2,3
```

---

### 6.5 E5 — 反事实一致性（已暂时取消，归档保留）

> **当前状态**: 本实验暂时取消，不纳入当前论文主实验、78server 队列、表格生成或结果汇总。
> **原因**: counterfactual image-swap 与 A1 黑图/视觉依赖诊断的核心思想重叠，当前增量贡献不足。
> **保留范围**: `scripts/build_cf_pairs.py`、`harness/data/cf_synthesizer.py`、`compute_pair_metrics`、`test_cf.json` 路径约定和以下命令全部仅作为归档资料保留；不要删除代码或已有数据。

**归档问题**: DREAMS 模型是否真正"看到"图像？面对 CF safe 样本时是否不同于 unsafe 样本？

**归档前置条件（当前不执行）**: 若未来恢复 E5，才需要生成 `test_cf.json`（`build_cf_pairs.py`）。

**归档指标（当前不汇报）**:
- **PD ↑** = Pair Discrimination (orig→Unsafe AND cf→Safe) / 总对数
- **PC ↑** = Pair Consistency (任意标签翻转) / 总对数
- **VS ↑** = Visual Sensitivity (响应文本变化 > 0.3) / 总对数

**归档表格（当前不生成 / 不进入论文主表）** (原 Table 11 / Section 5.5):

| 模型 | PD ↑ | PC ↑ | VS ↑ |
|------|------|------|------|
| InternVL3.5 + MIRage-data | | | ← 预计低 |
| Qwen3.5 + MIRage-data | | | |
| LLaVA-OV + MIRage-data | | | |
| **InternVL3.5 + DREAMS** | | | ← 预计高 |
| **Qwen3.5 + DREAMS** | | | |
| **LLaVA-OV + DREAMS** | | | |

**归档命令 — E5 推理（当前不运行；仅保留给未来恢复实验时参考）**:

```bash
# [归档保留 / 当前不运行] 确认 CF 文件存在
# ls /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test_cf.json

# [归档保留 / 当前不运行] InternVL3.5 + DREAMS — orig 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] InternVL3.5 + DREAMS — CF 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0

# [归档保留 / 当前不运行] Qwen3.5 + DREAMS — orig 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] Qwen3.5 + DREAMS — CF 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0

# [归档保留 / 当前不运行] LLaVA-OV + DREAMS — orig 推理
# conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] LLaVA-OV + DREAMS — CF 推理
# conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0

# [归档保留 / 当前不运行] InternVL3.5 + MIRage-data — orig 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] InternVL3.5 + MIRage-data — CF 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0

# [归档保留 / 当前不运行] Qwen3.5 + MIRage-data — orig 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] Qwen3.5 + MIRage-data — CF 推理
# conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0

# [归档保留 / 当前不运行] LLaVA-OV + MIRage-data — orig 推理
# conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-train --benchmarks our_test --cuda-visible-devices 0

# [归档保留 / 当前不运行] LLaVA-OV + MIRage-data — CF 推理
# conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-train --benchmarks our_test_cf --cuda-visible-devices 0
```

**归档命令 — E5 指标计算（当前不运行；仅保留给未来恢复实验时参考）**:

```bash
# [归档保留 / 当前不运行] 批量运行 E5（使用 run_main.py；78server 上必须跳过训练）
# conda activate mis_safety && python scripts/run_main.py --experiment-id E5 --cohort tier_a_dreams,tier_a_mirage_data --skip-train
```

---

## 7 Section 6 — Ablation Studies

**核心问题**: DREAMS 的哪些设计决策最关键？

> **[训练机]** 本节消融实验会训练新变体，不分配给 78server；78server 只可在训练完成后做已有 checkpoint 的推理/评测。

### 7.1 消融维度

| 消融实验 | 变量 | 核心指标 |
|---------|------|---------|
| Abl-1: 数据规模 | 25% / 50% / 75% / 100% DREAMS 训练数据 | E1 ASR |
| Abl-2: 图像类型混合比 | 纯合成 / 25%真实 / 50%真实 / 纯真实 | E2 Synth-Real Gap |
| Abl-3: 关系类型覆盖 | 仅 tool→target / 全4类 | E1 implicit ASR |
| Abl-4: CF pair 训练 | 已暂时取消；相关配置归档保留 | 不纳入当前消融 |

### 7.2 对应表格 (Table 12 / Section 6)

**Abl-1: 数据规模消融**

| 训练数据量 | E1 ASR ↓ | E3 AdvBench ASR ↓ |
|---------|---------|-----------------|
| 25% (~3.75K) | | |
| 50% (~7.5K) | | |
| 75% (~11K) | | |
| 100% (~15K) | | ← DREAMS full |

### 7.3 命令 — 消融实验（按实验逐条运行）

**Abl-1: 数据规模**

- [ ] **25% 数据**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override dataset.max_train_samples=3750 training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [ ] **50% 数据**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override dataset.max_train_samples=7500 training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [ ] **75% 数据**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override dataset.max_train_samples=11250 training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

> 100% 数据与 E1 共用 checkpoint，无需重复运行。

**Abl-2: 图像类型比例**

- [ ] **仅合成图**（filter_img_source_type=synth）

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override dataset.filter_img_source_type=synth training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

- [ ] **仅真实图**（filter_img_source_type=real）

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override dataset.filter_img_source_type=real training.save_strategy=steps training.save_steps=100 training.save_total_limit=1 --cuda-visible-devices 0,1,2,3
```

> 50/50 混合为默认配置，无需额外运行。

---

## 8 Appendix

> **[78server]** Appendix 表格生成和已有 responses 上的重评测都可以分配给 78server。

### 8.1 Appendix A — 按危害类别细分结果（降级自原 E3）

**内容**: 12 个危害类别 × best DREAMS 模型 vs best MIRage-data 模型，ASR 对比。

**对应输出**: metrics.json 中的 `per_category` 字段，无需额外推理。

- [ ] **从已有 metrics.json 生成预实验汇总表**

```bash
conda activate mis_safety && python scripts/generate_report.py --group prelim --format markdown
```

### 8.2 Appendix B — E3 完整 ASR/PR/BR/FPR 细分

**内容**: E3 中每个 benchmark 的完整 4 指标细分，Table 9 只展示 canonical metric，Appendix B 展示全部。

- [ ] **从已有 main 结果生成汇总表**

```bash
conda activate mis_safety && python scripts/generate_report.py --group main --format markdown
```

### 8.3 Appendix C — E5 VS 阈值敏感性分析（已暂时取消，归档保留）

> 当前不生成 Appendix C；VS 阈值敏感性分析随 E5 一起暂时取消。以下命令仅保留给未来恢复实验时参考。

**归档内容**: VS 指标在不同阈值（0.1 / 0.2 / 0.3 / 0.5）下的值，验证阈值选取的稳健性。

```bash
# 当前版本尚未暴露 `--vs-thresholds` CLI；
# 如需比较不同阈值，请在 `harness/evaluation/metrics.py` 中调用 `compute_pair_metrics(..., vs_threshold=...)`
```

---

## 9 模型速查表

### 9.1 Tier A — 同架构对比（需要 SFT 训练）

| 架构 | DREAMS SFT config | MIRage-data SFT config | Base HF id |
|------|-----------------|----------------------|-----------|
| InternVL3.5-8B | `main/main_dreams_internvl3_5.yaml` | `main/main_baseline_mirage_data_internvl3_5.yaml` | `OpenGVLab/InternVL3_5-8B` |
| Qwen3.5-9B | `main/main_dreams_qwen3_5.yaml` | `main/main_baseline_mirage_data_qwen3_5.yaml` | `Qwen/Qwen3.5-9B` |
| LLaVA-OV-1.5-8B | `main/main_dreams_llava_ov.yaml` | `main/main_baseline_mirage_data_llava_ov.yaml` | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` |

### 9.2 Tier B — 开源基线（仅推理，无 SFT）

| 模型 | HF id | Config | LF 模板 | 参数量 |
|------|-------|--------|---------|-------|
| Kimi-VL-A3B | `moonshotai/Kimi-VL-A3B-Instruct` | `main_baseline_kimi_vl_a3b.yaml` | `kimi_vl` | ~3B |
| MiniCPM-o-4.5 | `openbmb/MiniCPM-o-4_5` | `main_baseline_minicpm_o_4_5.yaml` | `minicpm_o` | ~4.5B |
| Gemma-4-E4B | `google/gemma-4-E4B-it` | `main_baseline_gemma_4_e4b.yaml` | `gemma4` | ~4B |
| GLM-4.6V-Flash | `zai-org/GLM-4.6V-Flash` | `main_baseline_glm_4_6v_flash.yaml` | `glm4_5v` | ~9B |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `main_baseline_qwen3_5_4b.yaml` | `qwen3_vl` | ~4B |
| InternVL3.5-4B | `OpenGVLab/InternVL3_5-4B-HF` | `main_baseline_internvl3_5_4b.yaml` | `intern_vl` | ~4B |
| LLaVA-OV-1.5-4B | `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` | `main_baseline_llava_ov_1_5_4b.yaml` | `llava_next` | ~4B |
| DeepSeek-VL2-Tiny | `deepseek-ai/deepseek-vl2-tiny` | `main_baseline_deepseek_vl2_tiny.yaml` | 原生 vLLM（无 LF 模板） | ~3B |

> **DeepSeek-VL2-Tiny 说明**: 仅推理。LlamaFactory 无对应模板，故走原生 vLLM 后端
> `harness/inference/deepseek_vl2_native_backend.py`（engine 按 `architecture: deepseek_vl2` 特判）。vLLM 0.18.1 原生支持 `DeepseekVLV2ForCausalLM`。
>
> ⚠️ **仍停用**: Phi-4-multimodal / Idefics2-8B / Ovis2.5-9B — LlamaFactory 暂无对应模板，且未接原生后端，从实验计划中移除。

### 9.3 Tier C — 闭源上限（SDK 调用）

| 模型 | `--models` 参数值 | SDK |
|------|----------------|-----|
| GPT-5.5 | `gpt-5.5` | OpenAI SDK |
| Gemini-3.1-Pro | `gemini-3.1-pro` | Google GenAI SDK |
| Claude-Opus-4.7 | `claude-opus-4.7` | Anthropic SDK |

### 9.4 数据路径速查

| 数据 | 路径 |
|------|------|
| DREAMS test | `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test.json` |
| DREAMS train (annotated) | `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/train_annotated.json` |
| E5 CF pairs（归档保留，当前不使用） | `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test_cf.json` |
| MIS-easy | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test/mis_easy.json` |
| MIS-hard | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test/mis_hard.json` |
| MIS-real | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test/mis_real.json` |
| MIS-train | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_train/mis_train.json` |

---

## 10 实验依赖顺序与并行执行图

> 本节帮助多机 / 多人协作分工：哪些任务可以同时启动，哪些必须等待前置产物。
> 当前主线：**A1–A4 + E1–E4 + Appendix/Report**；E5 已暂停，仅归档保留。

---

### 10.1 总体依赖 DAG（Mermaid，支持 GitHub / VS Code Preview 渲染）

```mermaid
graph TD
    classDef data     fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef prelim   fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef baseline fill:#fff9c4,stroke:#f9a825,color:#e65100
    classDef training fill:#fce4ec,stroke:#c62828,color:#b71c1c
    classDef infer    fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef report   fill:#fff3e0,stroke:#e65100,color:#bf360c
    classDef archived fill:#f5f5f5,stroke:#bdbdbd,color:#9e9e9e

    D0["🔧 D0\n环境 & 数据检查\ntest.json / train.json / .env / API keys"]:::data

    subgraph W0["⚡ Wave 0 — 无前置依赖，立即并行启动"]
        direction TB
        D1["📝 D1\n训练响应标注\ngenerate_responses.py\n→ train_annotated.json"]:::data
        D2["⚙️ D2\nTier B 配置生成\ngenerate_baseline_configs.py"]:::data
        A14["🔬 A1–A4\nPrelim 实验\nMIS 缺陷诊断"]:::prelim
        BASE["🤖 E1 Baseline 推理\nTier A base + Tier B open-source\n无 SFT，--skip-train"]:::baseline
        E4V0["📊 E4 V0\n通用能力基线\nbase model，无 SFT"]:::baseline
        G0["⬇️ M4-Instruct 数据\nhuggingface-cli download\nE4 V2/V4 需要"]:::data
        TC["☁️ Tier C\nGPT / Claude / Gemini\nrun_closed_source.py"]:::baseline
    end

    subgraph W1["🔥 Wave 1 — 需要 train_annotated.json，训练机运行"]
        direction TB
        SFT_D["🏋️ DREAMS SFT\n3 × Tier A 架构\nInternVL3.5 / Qwen3.5 / LLaVA-OV"]:::training
        SFT_M["🏋️ MIRage-data SFT\n3 × Tier A 架构\n同架构对照组"]:::training
    end

    subgraph W2["🚀 Wave 2 — 需要 SFT checkpoint，78server 推理+评测"]
        direction TB
        E1["📈 E1\n分布内安全主结果\n--skip-train"]:::infer
        E3["🌐 E3\n跨 benchmark 安全泛化\n--skip-train"]:::infer
        E4V13["⚡ E4 V1/V3\n能力保留 MIRage/DREAMS\n--skip-train"]:::infer
        E4V24["⚡ E4 V2/V4\n+通用数据 M4-Instruct 混入\n需要 G0 + SFT checkpoint"]:::infer
        E2["🔁 E2\nSynth/Real 图像切片\n复用 E1 输出，无需重新推理"]:::infer
    end

    REPORT["📄 Report\ngenerate_report.py\nLaTeX / Markdown 表格"]:::report
    E5X["❌ E5 CF Consistency\n已暂停 / 归档保留\n当前不运行"]:::archived

    %% D0 → Wave 0
    D0 --> D1
    D0 --> D2
    D0 --> A14
    D0 --> BASE
    D0 --> E4V0
    D0 --> G0
    D0 --> TC

    %% Wave 0 → Wave 1
    D1 --> SFT_D
    D1 --> SFT_M
    D2 --> BASE

    %% Wave 1 → Wave 2
    SFT_D --> E1
    SFT_M --> E1
    SFT_D --> E3
    SFT_M --> E3
    SFT_D --> E4V13
    SFT_M --> E4V13
    SFT_D --> E4V24
    SFT_M --> E4V24
    G0    --> E4V24

    %% E1 → E2（切片复用，无需重跑推理）
    E1 --> E2

    %% Wave 2 → Report
    A14   --> REPORT
    BASE  --> REPORT
    E4V0  --> REPORT
    TC    --> REPORT
    E1    --> REPORT
    E2    --> REPORT
    E3    --> REPORT
    E4V13 --> REPORT
    E4V24 --> REPORT

    %% E5 归档，不接入主链
    E5X -. 已暂停，不连入主链 .-> REPORT
```

---

### 10.2 并行执行分组

| 并行组 | 包含任务 | 前置条件 | 机器建议 | GPU 需求 | API 需求 |
|--------|---------|---------|---------|---------|---------|
| **Wave 0-A** 数据/配置 | D1 训练响应标注、D2 Tier B 配置生成、G0 M4-Instruct 下载 | D0 完成 | 任意 | D1 需要 GPU | D1 需要 OpenAI |
| **Wave 0-B** Prelim | A1 黑帧 probe、A2 GPT 关系标注、A3 MIS splits、A4 MSSBench FPR | D0 完成 | 78server | 是（A1/A3/A4） | A2 需要 GPT |
| **Wave 0-C** Baseline | E1 Tier A base 推理、E1 Tier B 推理、E4 V0 能力测试 | D0 完成 | 78server | 是 | 否 |
| **Wave 0-D** 闭源 | Tier C GPT / Claude / Gemini | D0 + API keys | API worker | 否 | 是 |
| **Wave 1** SFT 训练 | DREAMS SFT ×3、MIRage-data SFT ×3 | D1 完成 | **训练机** | 是（全卡） | 否 |
| **Wave 2-A** 安全推理 | E1 SFT 推理、E3 跨 benchmark、E4 V1/V3 | W1 checkpoint 同步 | 78server | 是 | E1 GPT judge |
| **Wave 2-B** 切片/混入 | E2 切片（复用 E1）、E4 V2/V4 | E1 完成 + G0 就绪 | 78server | E4 V2/V4 需要 | E2 无需 |
| **Report** | generate_report.py | 各实验 metrics.json 存在 | 78server | 否 | 否 |

---

### 10.3 严格串行依赖链（关键路径）

```
D0
 └─► D1: generate_responses.py → train_annotated.json
      ├─► SFT_DREAMS: 训练机 × 3 模型（最长瓶颈，数小时至天级）
      │    └─► E1 / E3 / E4 V1 V3 V4  →  Report
      └─► SFT_MIRAGE: 训练机 × 3 模型
           └─► E1 / E3 / E4 V1 V2     →  Report
```

**关键路径**：`D0 → D1 → SFT → E1 → E2 → Report`

**最长瓶颈**：SFT 训练（6 次，每次约 2–8 h 取决于 GPU 数量）。

---

### 10.4 推荐执行队列

**立即启动（Wave 0，可全部同时开跑）**

- [ ] **[78server] A1–A4 prelim**

```bash
# InternVL + Qwen (mis_safety env)
conda activate mis_safety && python scripts/run_prelim.py --experiment all \
    --models OpenGVLab/InternVL3_5-8B Qwen/Qwen3.5-9B \
    --cuda-visible-devices 0
# LLaVA-OV-1.5 (mis_safety_llava453 env — different transformers version required)
conda activate mis_safety_llava453 && python scripts/run_prelim.py --experiment all \
    --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
    --cuda-visible-devices 0
```

- [ ] **[78server] E1 Tier A base + Tier B baseline 推理**（无 SFT）

```bash
conda activate mis_safety && python scripts/run_main.py --experiment-id E1 --cohort tier_a_base,tier_b --cuda-visible-devices 0
```

- [ ] **[78server] E4 V0 能力基线**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V0 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0
```

- [ ] **[API worker / 78server] Tier C 闭源**

```bash
conda activate mis_safety && python scripts/run_closed_source.py --models gpt-5.5 claude-opus-4.7 gemini-3.1-pro --benchmarks our_test mis_easy mis_hard
```

- [ ] **[训练机] D1 训练响应标注**（同时启动）

```bash
conda activate mis_safety && python scripts/generate_responses.py \
  --input  /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/train.json \
  --output /mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/train_annotated.json \
  --backend vllm --model Qwen/Qwen3.5-122B-A10B --resume --gpu-ids 0,1,2,3
```

- [ ] **[任意] G0 M4-Instruct 数据下载**（同时启动，慢慢下）

```bash
conda activate mis_safety && huggingface-cli download lmms-lab/M4-Instruct \
  --repo-type dataset \
  --local-dir /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
```

**Wave 1 — D1 完成后立即启动 SFT（训练机）**

> `--skip-eval`: 训练机只做训练+推理，GPT 评测交给 78server；同时让 registry 不把此 run 标记为 completed，确保 78server `--skip-train` 能正常执行。

DREAMS SFT × 3 架构（可并行，或逐一排队）：

- [ ] **InternVL3.5 + DREAMS SFT**（until 6am）

```bash
conda activate mis_safety && CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

- [x] **Qwen3.5 + DREAMS SFT**

```bash
conda activate mis_safety && CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

- [x] **LLaVA-OV + DREAMS SFT**

```bash
conda activate mis_safety_llava && CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-eval --skip-inference --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

MIRage-data SFT × 3 架构（与 DREAMS SFT 并行，若 GPU 充足）：

- [x] **InternVL3.5 + MIRage-data SFT**

```bash
conda activate mis_safety && CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

- [x] **Qwen3.5 + MIRage-data SFT**

```bash
conda activate mis_safety && CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-eval --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

- [x] **LLaVA-OV + MIRage-data SFT**

```bash
conda activate mis_safety_llava && CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-eval --skip-inference --resume-latest-train --override training.save_strategy=steps training.save_steps=100 training.save_total_limit=1
```

**Wave 2 — SFT checkpoint 同步到 78server 后（可并行）**

**Step 1 — E1 SFT 推理**：

- [ ] **InternVL3.5 + DREAMS 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + DREAMS 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + DREAMS 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_llava_ov \
    --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data 推理**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data 推理**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
    --experiment-id E1 --skip-train --skip-eval \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — E1 SFT 评分**：

- [ ] **InternVL3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_internvl3_5/dreams_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_qwen3_5/dreams_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + DREAMS 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_dreams_llava_ov/dreams_llava_ov/responses/ \
    --evaluator-type auto
```

- [ ] **InternVL3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_internvl3_5/mirage_data_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_qwen3_5/mirage_data_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + MIRage-data 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E1/main_baseline_mirage_data_llava_ov/mirage_data_llava_ov/responses/ \
    --evaluator-type auto
```

**Step 1 — E3 跨 benchmark 推理**（每个 benchmark 推理后立即运行对应 Step 2 评分）：

- [ ] **InternVL3.5 + DREAMS — 全部 7 个 benchmark**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + DREAMS — 全部 7 个 benchmark**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + DREAMS — 全部 7 个 benchmark**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/dreams_llava_ov \
    --cuda-visible-devices 0
```

- [ ] **InternVL3.5 + MIRage-data — 全部 7 个 benchmark**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_internvl3_5 \
    --cuda-visible-devices 0
```

- [ ] **Qwen3.5 + MIRage-data — 全部 7 个 benchmark**

```bash
conda activate mis_safety && python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_qwen3_5 \
    --cuda-visible-devices 0
```

- [ ] **LLaVA-OV + MIRage-data — 全部 7 个 benchmark**

```bash
conda activate mis_safety_llava453 && python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
    --experiment-id E3 --skip-train --skip-eval \
    --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss \
    --model-path /mnt/hdd/xuran/vlm_safety_harness/models/mirage_data_llava_ov \
    --cuda-visible-devices 0
```

**Step 2 — E3 评分**（推理完成后按模型逐一运行）：

- [ ] **InternVL3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_internvl3_5/dreams_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + DREAMS 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_qwen3_5/dreams_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + DREAMS 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_dreams_llava_ov/dreams_llava_ov/responses/ \
    --evaluator-type auto
```

- [ ] **InternVL3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_internvl3_5/mirage_data_internvl3_5/responses/ \
    --evaluator-type auto
```

- [ ] **Qwen3.5 + MIRage-data 评分**

```bash
conda activate mis_safety && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_qwen3_5/mirage_data_qwen3_5/responses/ \
    --evaluator-type auto
```

- [ ] **LLaVA-OV + MIRage-data 评分**

```bash
conda activate mis_safety_llava453 && python scripts/run_eval_only.py \
    --responses /mnt/hdd/xuran/vlm_safety_harness/results/main/E3/main_baseline_mirage_data_llava_ov/mirage_data_llava_ov/responses/ \
    --evaluator-type auto
```

**E4 V1/V2/V3/V4**（`--skip-train`，需要 G0 就绪）：

- [ ] **E4 V1/V2/V3/V4 能力测试**

```bash
conda activate mis_safety && python scripts/run_capability.py --variants V1 V2 V3 V4 \
  --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt --cuda-visible-devices 0,1,2,3
```

**Report — 各实验输出就绪后**

- [ ] **E1 per-category 报告**

```bash
conda activate mis_safety && python scripts/generate_report.py --group main --experiment-set e1_per_category --format markdown
```

- [ ] **E3 完整报告**

```bash
conda activate mis_safety && python scripts/generate_report.py --group main --experiment-set e3_full --format markdown
```

- [ ] **Prelim 报告**

```bash
conda activate mis_safety && python scripts/generate_report.py --group prelim --format markdown
```

---

### 10.5 机器分工建议

| 机器角色 | 运行内容 | 备注 |
|---------|---------|------|
| **训练机**（全 GPU）| D1 大模型标注、Wave 1 SFT × 6 | DeepSpeed ZeRO-3，所有 GPU |
| **78server**（推理机）| Wave 0-B/C、Wave 2 全部推理+评测、Report | 只跑 `--skip-train`；不跑 SFT |
| **API worker** | Tier C 闭源、D1 OpenAI 标注 | 无 GPU 需求，纯 API 调用 |

---

### 10.6 归档任务（当前不进入队列）

| 实验 | 状态 | 原因 |
|------|------|------|
| E5 Counterfactual Consistency | ❌ 已暂停，归档保留 | 与 A1 黑图/视觉依赖诊断思路重叠，当前增量贡献不足 |
| Abl-4 CF pair 训练 | ❌ 已暂停，归档保留 | 依赖 E5，随 E5 一起暂停 |
| Appendix C VS 阈值敏感性 | ❌ 已暂停，归档保留 | 依赖 E5 PD/PC/VS 指标 |
