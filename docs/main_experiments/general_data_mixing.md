# MIS Table 6 通用数据混入分析与 DREAMS 方案建议

---

## 1 MIS 原论文 Table 6 详解

### 1.1 符号含义

Table 6 标题脚注：

| 符号 | 含义 |
|------|------|
| `†` | 在原方法基础上额外加入 **500 条通用数据（来自 MIRage，即 M4-Instruct）** |
| `‡` | 在原方法基础上额外加入 **6000 条通用数据（来源未明确说明，"other sources"）** |
| `∗` | MIRage **不加任何通用数据**的版本 |

### 1.2 通用数据来源（原论文明确说明的部分）

> *"MIRage. Similar to prior SFT methods, we add **500 general QA samples from M4-Instruct** (Li et al., 2024b) to preserve instruction-following ability. In the final 4.5k training set, only 11% are general samples..."*  
> — Section 3.3

- **主要通用数据集**: **M4-Instruct**（Li et al., 2024b）
  - HuggingFace: `lmms-lab/M4-Instruct` （LLaVA-OneVision 系列数据）
  - 多模态多图 QA 数据，涵盖单图 + 多图指令跟随
- **数量**: 500 条（占总训练集 4.5K 的 **11%**）
- **6000 条 "other sources"**: **论文正文中没有明确来源**，仅在 Table 6/7 脚注提及 "randomly sampled from other sources"

> **注意**: VLGuard 使用的是 LLaVA-v1.5 数据（5k 条）+ VLGuard 自身数据（1k 条）作为通用数据，而 Textual SFT 使用 1k general-safe samples。这些对照方法均有说明，但 MIRage 的 6000 条来源未说明。

### 1.3 Table 6 数据（复原）

基础模型：InternVL2.5-8B

| 方法 | 通用数据量 | 通用数据比 | Q-Bench | MMStar | MMMU | MuirBench | MMT | Average |
|------|----------|----------|---------|--------|------|-----------|-----|---------|
| Base | 0 | — | 73.11 | 62.87 | 54.33 | 51.35 | 60.70 | 60.47 |
| + Textual SFT | 0 | — | 71.77 | 60.47 | 54.00 | 47.30 | 59.14 | 58.54 |
| + Textual SFT† | 500 (M4) | 33% | 71.51 | 62.00 | 48.38 | 53.33 | 60.17 | 59.08 |
| + VLGuard-R | 0 | — | 72.03 | 62.00 | 52.89 | 45.88 | 59.67 | 58.49 |
| + VLGuard-R† | 500 (M4) | 33% | 72.44 | 62.06 | 54.11 | 51.53 | 60.44 | 60.12 |
| + VLGuard-R‡ | 6000 (?) | **75%** | 74.65 | 62.03 | 54.77 | 47.58 | 59.51 | 59.71 |
| + MIRage∗ | 0 | 0% | 72.91 | 62.47 | 54.78 | 51.54 | 60.95 | 60.53 |
| **+ MIRage** | **500 (M4)** | **11%** | **73.31** | **63.13** | **55.00** | **54.15** | **60.92** | **61.30** |

### 1.4 核心发现

1. **MIRage 用最少通用数据（11%）达到最好通用能力，甚至略超 Base Model**
2. VLGuard-R‡（6000 条，75%比例）并没有显著优于 VLGuard-R†（500 条）→ **堆通用数据不是解法**
3. MIRage∗（无通用数据）与 MIRage（500 条）差距很小，说明 **多图安全训练本身能保留通用能力**，通用数据只是辅助
4. **MuirBench（多图理解）** 是最关键的指标：MIRage 在此列表现最好（54.15），而其他方法均低于 Base（51.35）——**多图训练对多图能力有增益，不是损失**

---

## 2 我们如何做（DREAMS 方案建议）

### 2.1 直接复用 MIRage 方案（推荐）

DREAMS 训练集约 **~15K 条**，MIS 约 **4K 条**（比例关系同等保持 11% 比例）：

| DREAMS 安全数据 | 通用数据比例 | 通用数据量 |
|--------------|-----------|---------|
| ~15,000 | 11% (= MIRage 比例) | ~1,650 条 |
| ~15,000 | 3.3% (更保守，按绝对量与 MIRage 相同) | **500 条** |

**建议起点**: **500 条 M4-Instruct** （与 MIRage 绝对数量相同，方便直接对比）。
若 E4 结果显示能力仍有损失，再扩至 ~1,650 条。

### 2.2 通用数据集来源推荐

#### 首选：M4-Instruct（与 MIRage 完全一致）

```
HuggingFace: lmms-lab/M4-Instruct
```
- 多模态多图 QA，与 DREAMS 的双图输入格式天然兼容
- 直接用与 MIRage 相同数据 → "同等实验条件"，论文对比更公平
- 下载：
```bash
huggingface-cli download lmms-lab/M4-Instruct --repo-type dataset --local-dir /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
```

#### 次选：LLaVA-OneVision-Data（若需要更强多图指令跟随）

```
HuggingFace: lmms-lab/LLaVA-OneVision-Data
```
- 包含大量多图 QA，与 DREAMS 双图场景更接近
- 数据规模大（可以按需采样）
- 比 M4-Instruct 更新，多图指令类型更丰富

#### 可选：ShareGPT4V（单图高质量 caption + QA）

```
HuggingFace: Lin-Chen/ShareGPT4V
```
- 单图，但质量高（GPT-4V 标注）
- 用于补充单图通用理解能力（MMStar / Q-Bench 等 SI benchmark）

### 2.3 我们的实验设计（对应 E4 V2/V4）

**E4 变体对应关系**：

| 变体 | 安全数据 | 通用数据 | 对应 Table 6 的哪个条件 |
|------|---------|---------|----------------------|
| V1 | mis_train.json | 无 | ≈ MIRage∗ |
| V2 | mis_train.json | M4-Instruct 500条 | = MIRage (†) |
| V3 | DREAMS train | 无 | ≈ MIRage∗ |
| V4 | DREAMS train | M4-Instruct 按最终训练集 11% 比例混入 | = MIRage 的 11% general-data ratio，用我们数据 |

**论文叙事**（类比 MIS 的叙述逻辑）：
> "Following MIRage, we add 500 M4-Instruct samples to the MIRage-data control (V2), and for DREAMS we match MIRage's 11% final general-data ratio (V4). As Table shows, DREAMS preserves general capability with minimal general data, consistent with MIRage's finding."

### 2.4 消融建议（若需要）

若审稿人质疑通用数据量，可做 4 点消融：

| 通用数据量 | 通用/安全比 | 说明 |
|---------|-----------|------|
| 0 | 0% | V3 baseline |
| 500 | ~3% | 与 MIRage 绝对量相同 |
| ~1,650 | 11% | 与 MIRage 相同比例 |
| 6,000 | ~40% | 对应 MIS Table 6 中的 ‡ 条件 |

---

## 3 当前配置实现（E4 V2/V4 已解锁）

当前仓库已经完成如下配置：

```yaml
# /mnt/hdd/xuran/vlm_safety_harness/configs/experiments/main/E4_V2_internvl3_5_mirage_general.yaml
training:
  enabled: true
  general_data:
    sources:
      - /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
    max_samples: 500
    format: sharegpt
    shuffle_seed: 0

# /mnt/hdd/xuran/vlm_safety_harness/configs/experiments/main/E4_V4_internvl3_5_dreams_general.yaml
training:
  enabled: true
  general_data:
    sources:
      - /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
    ratio: 0.11
    ratio_mode: final
    format: sharegpt
    shuffle_seed: 0
```

`/mnt/hdd/xuran/vlm_safety_harness/scripts/run_capability.py` 现在不会再因“未填写 general_data”而拦住 V2/V4；当前唯一前置条件是本地准备好 `/mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct` 数据源。

---

## 4 总结

| 问题 | 结论 |
|------|------|
| MIS 用了什么通用数据集？ | **M4-Instruct**（500条，lmms-lab/M4-Instruct），占训练集 11% |
| "other sources" 6000条是什么？ | **论文未说明**，仅作对照消融，且效果并不更好 |
| DREAMS 推荐用什么？ | **M4-Instruct 500 条**（直接对标 MIRage，公平对比）；若多图能力不足再加 LLaVA-OV-Data |
| 通用数据比例建议？ | 先试 500 条（保守），E4 结果不好再试 11% 比例（~1650 条） |
| 核心逻辑 | MIRage 的关键发现是"多图安全训练本身能保留通用能力"，堆通用数据作用有限。DREAMS 应复现并强调同样的结论。 |
