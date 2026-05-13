# DREAMS VLM Safety Harness — 代码结构与使用指南

## 项目概述

本项目实现多图 VLM 安全微调实验框架，旨在超越 MIS (ICLR 2026)。核心思路：用 DREAMS 数据集（17K+ 样本，12 危害类别，每条 2 图 + 1 文本）训练 VLM，在安全性和通用能力上超越 MIRage baseline。

**训练/推理框架**: LlamaFactory（位于 `/mnt/hdd/xuran/LlamaFactory`）  
**Conda 环境**: `mis_safety`（所有训练/推理/评测命令均在此环境下运行）
**E5 状态**: counterfactual consistency 暂时取消；相关代码、配置和 `test_cf.json` 路径仅归档保留，当前主流程不运行。

---

## 目录结构

```
/mnt/hdd/xuran/vlm_safety_harness/
│
├── CLAUDE.md                          # 项目规则（每次 session 加载）
│
├── docs/
│   ├── overview/
│   │   ├── project_overview.md        # 项目完整规划
│   │   └── Models_List.md             # 所有模型列表
│   ├── dataset/
│   │   ├── dataset_construction_plan.md
│   │   ├── testset_annotation_plan.md
│   │   └── response_generation_plan.md
│   ├── prelim_experiments/
│   │   ├── A_experiments.md           # A1-A4 实验设计
│   │   ├── A_experiments_handoff.md
│   │   ├── A_experiments_run_guide.md
│   │   └── result_tables.md
│   ├── main_experiments/
│   │   ├── main_experiments_handoff.md
│   │   └── initial_framework_plan_v3_llamafactory.md  # 当前实施计划（canonical）
│   ├── ablation_experiments/
│   └── USAGE.md                       # 本文件
│
├── configs/
│   ├── base/                          # 基础模型配置
│   │   ├── model_internvl3_5_8b.yaml
│   │   ├── model_qwen3_5_9b.yaml
│   │   └── model_llava_ov_1_5_8b.yaml
│   └── experiments/
│       ├── main/
│       │   ├── main_dreams_internvl3_5.yaml         # Tier A DREAMS SFT
│       │   ├── main_dreams_qwen3_5.yaml
│       │   ├── main_dreams_llava_ov.yaml
│       │   ├── main_baseline_mirage_data_internvl3_5.yaml  # Tier A MIRage 对照
│       │   ├── main_baseline_mirage_data_qwen3_5.yaml
│       │   ├── main_baseline_mirage_data_llava_ov.yaml
│       │   ├── main_baseline_kimi_vl_a3b.yaml       # Tier B（共 8 个，脚本生成）
│       │   ├── E4_V0_internvl3_5.yaml               # E4 能力保留（无 SFT）
│       │   ├── E4_V1_internvl3_5_mirage.yaml        # E4 V1（MIRage 数据）
│       │   ├── E4_V2_internvl3_5_mirage_general.yaml  # E4 V2（MIRage + 500 M4-Instruct）
│       │   ├── E4_V3_internvl3_5_dreams.yaml        # E4 V3（DREAMS 数据）
│       │   ├── E4_V4_internvl3_5_dreams_general.yaml  # E4 V4（DREAMS + 11% M4-Instruct）
│       │   ├── _cohort.yaml                         # 模型队列定义
│       │   ├── _baseline_template.yaml              # Tier B YAML 模板
│       │   └── _tier_b_models.csv                   # Tier B 模型参数表
│       ├── prelim/                    # A 实验配置
│       └── ablation/                  # 消融实验配置
│
├── harness/                           # 核心 Python 包
│   ├── config/
│   │   ├── schema.py                  # Pydantic 配置模型（ExperimentConfig）
│   │   ├── loader.py                  # YAML 加载 + _extends 继承解析
│   │   └── registry.py               # 实验注册与 config_hash 去重
│   ├── data/
│   │   ├── dataset.py                 # HarnessDataset（支持 harm_type / img_source_type 切片）
│   │   ├── converters.py              # DREAMS ↔ LlamaFactory ShareGPT 格式转换
│   │   ├── cf_synthesizer.py          # E5 反事实样本合成（归档保留，当前不运行）
│   │   ├── probe_builder.py           # A 实验 probe 构建
│   │   └── benchmarks/
│   │       ├── base.py                # BenchmarkBase ABC（metric_name / direction / evaluator_type）
│   │       ├── mis_benchmark.py       # MIS easy/hard/real（完整实现）
│   │       ├── mssbench.py            # MSSBench safe/unsafe（完整实现）
│   │       ├── figstep.py             # FigStep（完整实现）
│   │       ├── advbench.py            # AdvBench（完整实现）
│   │       ├── mm_safety.py           # MM-SafetyBench（完整实现）
│   │       ├── mmstar.py              # MMStar（完整实现）
│   │       ├── safebench.py           # SafeBench（stub — 需下载数据集）
│   │       ├── jailbreakv.py          # JailbreakV（stub）
│   │       ├── siuo.py                # SIUO（stub）
│   │       ├── mmmu.py                # MMMU（stub）
│   │       ├── muirbench.py           # MuirBench（stub）
│   │       ├── blink.py               # BLINK（stub）
│   │       └── mmt.py                 # MMT-Bench（stub）
│   ├── gpu/
│   │   └── allocator.py              # GPUAllocator — 动态检测可用 GPU
│   ├── training/
│   │   ├── trainer.py                # HarnessTrainer（封装 LF torchrun launcher）
│   │   └── cot_generator.py          # CoT 标注生成
│   ├── inference/
│   │   ├── lf_backend.py             # LFInferenceBackend（主推理路径，基于 ChatModel.achat）
│   │   ├── engine.py                 # InferenceEngine（多 benchmark 调度 + BENCHMARK_REGISTRY）
│   │   ├── model_configs.py          # vLLM 模型配置（已废弃，仅保留兼容）
│   │   └── vllm_backend.py           # VLLMBackend（已废弃）
│   ├── evaluation/
│   │   ├── gpt4o_evaluator.py        # GPT-4o 评测（ASR/RSR/RR/HR 标注）
│   │   ├── metrics.py                # MetricsDict + compute_metrics + compute_pair_metrics
│   │   ├── benchmark_evaluator.py    # BenchmarkEvaluator ABC + get_evaluator() 工厂
│   │   ├── llama_guard.py            # LlamaGuard 评测器
│   │   └── evaluators/
│   │       ├── gpt4o.py              # GPT-4o 适配器
│   │       ├── rule_based.py         # 关键词规则（AdvBench 风格）
│   │       ├── accuracy.py           # 多选题正确率（MMStar 等）
│   │       └── harmbench.py          # HarmBench（stub）
│   └── reporting/
│       ├── table_generator.py        # 论文表格生成（E1–E4 + 消融 + prelim；E5 归档保留）
│       └── aggregator.py             # 结果聚合
│
├── scripts/                          # CLI 入口
│   ├── run_experiment.py             # 单实验端到端（训练 → 推理 → 评测）
│   ├── run_main.py                   # E1/E2/E3 队列编排器（E5 入口归档保留）
│   ├── run_capability.py             # E4 能力评测编排器（含 V2/V4 guard）
│   ├── run_closed_source.py          # Tier C 闭源模型推理（OpenAI/Anthropic/Gemini SDK）
│   ├── run_prelim.py                 # A 实验批量运行
│   ├── run_inference_only.py         # 仅推理
│   ├── run_eval_only.py              # 仅评测（支持 --evaluator-type）
│   ├── build_cf_pairs.py             # 离线构建 E5 反事实对（归档保留，当前不运行）
│   ├── generate_baseline_configs.py  # 从 CSV 生成 Tier B YAML 配置
│   ├── generate_cot_labels.py        # 生成 CoT 标注
│   ├── generate_responses.py         # 生成训练响应（大模型标注）
│   └── generate_report.py            # 生成论文表格（LaTeX / Markdown）
│
├── results/                          # 实验输出（自动创建）
│   └── main/{experiment_name}/{YYYYMMDD_HHMMSS}/
│       ├── config_snapshot.yaml
│       ├── config_hash.txt
│       ├── gpu_plan.json
│       ├── responses/{benchmark}.jsonl
│       ├── eval_results/{benchmark}.jsonl
│       └── metrics.json
│
├── models/                           # 微调 checkpoint 输出目录
└── data_links/                       # 数据符号链接
    ├── our_dataset → /mnt/hdd/xuran/mis_dataset_builder/dataset/
    ├── mis_test    → /mnt/hdd/xuran/MIS/mis_test/
    └── mis_train   → /mnt/hdd/xuran/MIS/mis_train/
```

---

## 核心模块说明

### 1. 配置系统（`harness/config/`）

**YAML 继承**（`_extends` 关键字）：
```yaml
# 子配置只覆盖差异部分；路径相对于 configs/ 根目录
_extends: experiments/main/main_dreams_internvl3_5.yaml
dataset:
  train_path: /mnt/hdd/xuran/vlm_safety_harness/data_links/mis_train/mis_train.json
```

**命令行覆盖**（`--override`）：
```bash
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --override training.learning_rate=5e-6 training.num_train_epochs=5
```

### 2. 数据集切片（`harness/data/dataset.py`）

两种 JSON schema 自动识别：
- **Schema A**（train.json）：`conversations`, `image`（列表）, `img_source`
- **Schema B**（test.json）：`question`, `image_path1/2`, `img1_source`, `img2_source`, `harm_type`

`img_source_type` 自动派生规则：
- 两张图都是 AI-generated → `synth`
- 两张图都是 Web-crawled/local → `real`
- 混合 → `mix`

### 3. 推理后端（`harness/inference/lf_backend.py`）

主推理路径通过 `llamafactory.chat.ChatModel.achat()` 异步批量推理：
```python
backend = LFInferenceBackend(
    model_path="OpenGVLab/InternVL3_5-8B",
    template="intern_vl",   # 必须与 ARCH_TO_TEMPLATE 一致
    infer_plan=infer_plan,
    concurrency=16,
)
outputs = backend.generate_batch(records)
```

### 4. 评测与指标（`harness/evaluation/metrics.py`）

**评测路由**：
- `scripts/run_experiment.py` 与 `scripts/run_eval_only.py` 都会先读取 benchmark loader 的 `evaluator_type`
- `gpt4o`：用于 `our_test` 与其他安全 benchmark；每个样本只发 **1 次** GPT 请求，同时返回 `label_str` 与 `label_perception`
- `rule`：用于 `advbench` 等规则判定 benchmark
- `accuracy`：用于 `mmstar` / `mmmu` / `muirbench` / `blink` / `mmt` 等能力 benchmark

**安全指标（DREAMS-strengthened, MIS-compatible；ASR + RSR + RR + HR = 1.0）**：
- **ASR** (↓) = count(`Unsafe`) / total — 攻击成功率
- **RSR** (↑) = count(`Safe` & `Perception Level1`) / total — 感知正确后的安全拒绝
- **RR**  (↓) = count(`Safe` & `Perception Level2`) / total — 未直接描述图像但仍安全拒绝
- **HR**  (↓) = count(`Safe` & `Perception Level3`) / total — 未能看图或未形成有效视觉理解的拒绝

`GPT-4o` judge 位于 `harness/evaluation/gpt4o_evaluator.py`：保持 MIS 兼容输出 schema，但 prompt 已增强为对隐式危害、跨图组合危害、伪装/间接危害更谨慎的 DREAMS 版本。

**E5 反事实指标（归档保留，当前不汇报）**：
- **PD** (↑) = Pair Discrimination — orig→Unsafe AND cf→Safe / 总对数
- **PC** (↑) = Pair Consistency — 标签翻转对数 / 总对数
- **VS** (↑) = Visual Sensitivity — 响应文本变化 > 0.3 / 总对数

### 5. 架构→LF模板映射（`harness/training/trainer.py`）

```python
ARCH_TO_TEMPLATE = {
    "internvl":      "intern_vl",    # InternVL2.5 / InternVL3.5
    "qwen2vl":       "qwen2_vl",     # Qwen2.5-VL
    "qwen3_vl":      "qwen3_vl",     # Qwen3.5-9B (Qwen/Qwen3.5-9B)
    "llava":         "llava_next",   # LLaVA-OneVision-1.5
    "kimi_vl":       "kimi_vl",      # Kimi-VL-A3B
    "minicpm":       "minicpm_v",
    "minicpm_v_4_6": "minicpm_v_4_6",
    "minicpm_o":     "minicpm_o",
    "gemma_vlm":     "gemma4",       # Gemma-4 vision
    "glm4v":         "glm4v",
    "glm4_5v":       "glm4_5v",      # GLM-4.6V
}
```

---

## 使用指南

### 环境准备

```bash
conda activate mis_safety

# 一次性准备：复制模板并填写本机 key（.env 已被 .gitignore 忽略）
cp .env.example .env
$EDITOR .env

# 每次运行前加载
source .env
```

### 数据准备（一次性）

```bash
# Step 1: 生成训练响应标注
python scripts/generate_responses.py --input /mnt/hdd/xuran/mis_dataset_builder/dataset/train.json --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json --backend vllm --model Qwen/Qwen3.5-122B-A10B --resume

# Step 2: [归档保留 / 当前不运行] 构建 E5 反事实对（需要 benign image 目录）
# E5 counterfactual consistency 暂时取消；以下命令仅保留给未来恢复实验时参考。
# python scripts/build_cf_pairs.py --test-json /mnt/hdd/xuran/mis_dataset_builder/dataset/test.json --benign-pool /path/to/openimages_subset --output /mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json --cf-images-dir /mnt/hdd/xuran/mis_dataset_builder/dataset/cf_images --swap-idx 2 --seed 0

# Step 3: 生成 Tier B 推理配置（从 CSV 模板）
python scripts/generate_baseline_configs.py
```

---

## 主实验命令参考

### 单实验端到端

```bash
# dry-run 验证配置
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --dry-run --limit 5

# 完整运行（训练 + 推理 + 评测）
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml

# 跳过训练（使用已有 checkpoint）
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-train --model-path /mnt/hdd/xuran/vlm_safety_harness/models/internvl3_5_dreams/
```

### E1 — DREAMS 分布内安全（explicit / implicit 分层）

```bash
# Tier A DREAMS 训练模型
python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams

# 全队列（Tier A + Tier B）
python scripts/run_main.py --experiment-id E1 --cohort tier_a_dreams,tier_a_mirage_data,tier_a_base,tier_b
```

### E2 — 合成/真实/混合图像分层（验证 A3 假设）

```bash
python scripts/run_main.py --experiment-id E2 --cohort tier_a_dreams,tier_a_mirage_data,tier_a_base,tier_b
```

### E3 — 跨 benchmark 安全泛化

```bash
python scripts/run_main.py --experiment-id E3 --cohort tier_a_dreams,tier_a_mirage_data,tier_b
```

### E4 — 通用能力保留（MMStar / MMMU / MuirBench / BLINK / MMT）

```bash
# 已实现的 V0–V4；V2=500 条 M4-Instruct，V4=最终数据集 11% M4-Instruct
python scripts/run_capability.py --variants V0 V1 V2 V3 V4 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt

# 如本地尚未准备通用数据源，先下载 M4-Instruct 到约定目录
huggingface-cli download lmms-lab/M4-Instruct --repo-type dataset --local-dir /mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct
```

### E5 — 反事实一致性（已暂时取消，归档保留）

```bash
# [归档保留 / 当前不运行] 确认 CF 文件存在
# ls /mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json

# [归档保留 / 当前不运行] 如未来恢复 E5，再运行该队列
# python scripts/run_main.py --experiment-id E5 --cohort tier_a_dreams,tier_a_mirage_data
```

### Tier C — 闭源模型（GPT / Claude / Gemini）

```bash
# --models 必须显式指定，无默认值
python scripts/run_closed_source.py --models gpt-5.5 claude-opus-4.7 gemini-3.1-pro --benchmarks our_test mis_easy mis_hard --limit 100 --dry-run  # 先 dry-run

# 确认无误后去掉 --dry-run
python scripts/run_closed_source.py --models gpt-5.5 claude-opus-4.7 gemini-3.1-pro --benchmarks our_test mis_easy mis_hard
```

### MIRage 对照重训练（Tier A 同架构）

```bash
# 必须用 mis_train.json 重训练；禁止使用 Tuwhy/* 公开 checkpoint 作对照
python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml

python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml

python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml
```

---

## 单独推理 / 评测

```bash
# 仅推理
python scripts/run_inference_only.py --config main/main_dreams_internvl3_5.yaml --model-path /path/to/checkpoint --benchmarks our_test mis_easy mis_hard --limit 200

# 仅评测（自动按 benchmark 的 evaluator_type 路由）
python scripts/run_eval_only.py --responses results/main/main_dreams_internvl3_5/YYYYMMDD_HHMMSS/responses/

# 强制使用 GPT judge（仅建议用于安全 benchmark）
python scripts/run_eval_only.py --responses results/main/.../responses/ --benchmarks our_test --evaluator-type gpt4o

# 规则评测（AdvBench 关键词匹配）
python scripts/run_eval_only.py --responses results/main/.../responses/ --benchmarks advbench --evaluator-type rule

# 准确率评测（MMStar 多选题）
python scripts/run_eval_only.py --responses results/main/.../responses/ --benchmarks mmstar --evaluator-type accuracy
```

---

## 生成论文表格

```bash
# 所有当前主实验表格（E1–E4；E5 归档保留，不默认生成）
python scripts/generate_report.py --group main --format latex markdown

# 单个实验
python scripts/generate_report.py --group main --experiment-set e1 --format markdown
```

---

## 实验结果格式

```
results/main/{experiment_name}/{YYYYMMDD_HHMMSS}/
├── config_snapshot.yaml       # 完整配置快照
├── config_hash.txt            # 去重用 hash
├── gpu_plan.json              # GPU 分配方案
├── responses/
│   ├── our_test.jsonl         # {id, question, response, image_path1, image_path2,
│   │                          #  category, harm_type, img_source_type}
│   └── mis_easy.jsonl
├── eval_results/
│   ├── our_test.jsonl         # {…, label_str, label_perception}
│   └── mis_easy.jsonl
└── metrics.json               # 单 benchmark 时同时保留根级 overall / per_harm_type /
                               # per_img_source_type；并始终保留 benchmarks 子块
                               # 例如：
                               # {
                               #   "overall": {"ASR": ..., "RSR": ..., "RR": ..., "HR": ...},
                               #   "per_harm_type": {"explicit": {...}, "implicit": {...}},
                               #   "per_img_source_type": {"synth": {...}, "real": {...}, "mix": {...}},
                               #   "benchmarks": {"our_test": {"ASR": ..., "RSR": ..., "RR": ..., "HR": ...}},
                               #   "PD": ..., "PC": ..., "VS": ..., "vs_threshold": 0.3  # E5 归档字段，当前主流程不要求
                               # }
```

---

## 关键约束

| 约束 | 说明 |
|------|------|
| GPU 分配 | 不硬编码；每次通过 `GPUAllocator` 动态检测 |
| Conda 环境 | 训练/推理/评测全部用 `mis_safety` |
| MIRage 对比 | 必须用同架构 + mis_train.json 重训练；禁用 `Tuwhy/*` 公开 checkpoint |
| API Key | 环境变量注入，不得硬编码 |
| E4 V2/V4 | 配置已就绪：V2=`500` 条 M4-Instruct，V4=`11%` final ratio；运行前仅需确保 `/mnt/hdd/xuran/mis_dataset_builder/general_data/m4_instruct` 存在 |
| E5 | 已暂时取消；`test_cf.json` / `build_cf_pairs.py` 仅归档保留，当前主流程不生成、不运行 |
| Tier C | `--models` 必须显式指定，无默认值 |
| MIS 目录 | 禁止修改 `/mnt/hdd/xuran/MIS/` 下任何文件 |

---

## 快速参考：数据路径

| 数据 | 路径 |
|------|------|
| DREAMS train | `/mnt/hdd/xuran/mis_dataset_builder/dataset/train.json` |
| DREAMS test | `/mnt/hdd/xuran/mis_dataset_builder/dataset/test.json` |
| E5 CF pairs（归档保留，当前不使用） | `/mnt/hdd/xuran/mis_dataset_builder/dataset/test_cf.json` |
| MIS test sets | `/mnt/hdd/xuran/MIS/mis_test/{mis_easy,mis_hard,mis_real}.json` |
| MIS train | `/mnt/hdd/xuran/MIS/mis_train/mis_train.json` |
| LlamaFactory | `/mnt/hdd/xuran/LlamaFactory` |
| GPT-4o 评测参考 | `/mnt/hdd/xuran/MIS/evaluation/gpt_eval.py` |
