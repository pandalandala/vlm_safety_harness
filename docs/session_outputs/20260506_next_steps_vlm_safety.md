# VLM Safety Research — Next Steps

**Date**: 2026-05-06  
**Project**: `/mnt/hdd/xuran/vlm_safety_harness/`

---

## 当前状态速览

| 资源 | 状态 |
|------|------|
| Harness 框架 | ✅ 完成（Phase 1-4） |
| DREAMS scored.json | ✅ 17,022 samples |
| DREAMS train.json | ✅ 15,319 samples |
| DREAMS test.json | ✅ 1,703 samples |
| DREAMS CoT 标签 | ❌ 0 / 17,022（全部空） |
| MIS easy/hard/real | ✅ 1675 / 510 / 100 samples |
| MIRage checkpoint | ❌ 未找到（需确认路径） |
| MSSBench | ❌ 未下载 |

---

## Phase A：A 实验前置准备（先做）

### A0. 确认 MIRage checkpoint
- **动作**：告诉我 MIRage checkpoint 路径，或提供 HuggingFace model ID
- A1/A2/A3/A4 均需用 MIRage 做对比。若无 MIRage，只能跑 base model 端。
- **预计工作量**：用户提供路径即可，0分钟

### A1. 运行 A3（最简，零额外准备）
```bash
python scripts/run_prelim.py --experiment A3 --model-path [base_model_path] --dry-run
python scripts/run_prelim.py --experiment A3 --model-path OpenGVLab/InternVL2_5-8B
```
- 直接用 MIS easy/hard/real 三子集
- 需要 base model + MIRage 各跑一次
- **预计 GPU 时间**：~2h（单卡 A6000，3个subset × 2个模型）

### A2. 下载 MSSBench（A4 依赖）
```bash
# MSSBench HF: JailbreakBench/MSSBench or similar — 待确认实际 repo
huggingface-cli download --repo-type dataset [mssbench_repo] --local-dir data_links/mssbench
```
- 需确认 MSSBench 的 HuggingFace Dataset ID

### A3. 构建 A1/A2 探针数据集
```bash
python scripts/run_prelim.py --build-probes
```
- A1 自动构建（黑图替换），立即可运行
- A2 需要手工构造 `extra_relation_probe.jsonl`（~150条，图片来自COCO/OpenImages）

---

## Phase B：DREAMS 训练前置（A 实验完成后）

### B1. 生成 CoT 标签（关键路径）
```bash
# 若有72B VLM可用：
python scripts/generate_cot_labels.py \
    --input data_links/our_dataset/train.json \
    --backend vllm --model Qwen/Qwen2-VL-72B-Instruct

# 若无大模型，用GPT-4o：
python scripts/generate_cot_labels.py \
    --input data_links/our_dataset/train.json \
    --backend openai --model gpt-4o
```
- 15,319 samples × ~$0.01/sample (GPT-4o) = ~$150
- vLLM 72B on 4× A6000: ~8-12h
- **这是训练的前置条件**

### B2. 更新 dataset config 路径
configs 中 `train_path` 已指向 `data_links/our_dataset/train.json` ✅  
CoT 生成完成后需将 `train_path` 指向 `*_cot.json` 输出文件。

### B3. 训练（B1完成后）
```bash
python scripts/run_experiment.py main/main_dreams_internvl.yaml
python scripts/run_experiment.py main/main_dreams_qwen2vl.yaml
```

---

## Phase C：评测与论文

### C1. 运行消融实验
```bash
for cfg in abl_no_cot abl_synthetic_only abl_no_diverse_relations abl_no_cf_pairs abl_data_scale; do
    python scripts/run_experiment.py ablation/${cfg}.yaml --skip-train
done
```

### C2. 生成论文表格
```bash
python scripts/generate_report.py --group main --format latex --output paper_tables/main.tex
python scripts/generate_report.py --group prelim
python scripts/generate_report.py --group ablation --format latex
```

---

## 立即可做的事（无需额外信息）

1. `python scripts/run_prelim.py --build-probes` — 构建A1/A2探针（2分钟）
2. 用 base InternVL2.5-8B 跑 A3（不需 MIRage）
3. 确认 MSSBench 的 HuggingFace repo 名称

---

## 需要用户确认的信息

| 问题 | 用途 |
|------|------|
| MIRage checkpoint 路径/HF ID？ | A1-A4 baseline |
| MSSBench HuggingFace repo ID？ | A4 实验 |
| CoT 生成用 GPT-4o 还是本地72B？ | 预算 vs 速度 |
| A2 relation probe 图片从哪里获取（COCO/OpenImages 账户）？ | A2 手工探针 |

---

## 系统管理说明

- **命令日志**：每次 Bash 命令自动记录到 `/mnt/hdd/xuran/claude_logs/commands_YYYYMMDD.log`
- **回答存档**：每次回话结束自动保存到 `/mnt/hdd/xuran/claude_outputs/`
- **调研任务**：文献检索、代码库探索、数据格式调查 → 交给 subagent（Explore/general-purpose）
- **主 agent**：规划、综合、写代码、做决策
