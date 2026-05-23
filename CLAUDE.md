# DREAMS VLM Safety Harness — Project Rules

## 项目背景
基于 MIS (ICLR 2026) 缺陷，用 DREAMS 数据集改进多图 VLM 安全微调。
MIS 缺陷详见: `.claude/docs/MIS_shortcomes_final.md`

## 关键路径（绝对路径）

| 资源 | 路径 |
|------|------|
| 我们的数据集（上游源） | `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/` |
| MIS 基准测试集（上游源） | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test/` |
| MIS 训练数据（上游源） | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_train/` |
| **实验读取路径（快照）** | `/mnt/hdd/xuran/vlm_safety_harness/data_links/{our_dataset, mis_test, mis_train}/` |
| GPT-4o 评测参考 | `/mnt/hdd/xuran/MIS/evaluation/gpt_eval.py` |
| vLLM 推理参考 | `/mnt/hdd/xuran/MIS/evaluation/inference_vllm.py` |
| LF 训练模板 | `/mnt/hdd/xuran/LLaMA-Factory/examples/train_full/qwen2_5vl_full_sft.yaml` |
| LF 数据集注册 | `/mnt/hdd/xuran/LLaMA-Factory/data/dataset_info.json` |
| 项目实验结果 | `/mnt/hdd/xuran/vlm_safety_harness/results/` |
| 项目 Harness 代码 | `/mnt/hdd/xuran/vlm_safety_harness/harness/` |

## GPU 使用规则（重要）

- **不要硬编码 GPU 数量**，每次会话由 `GPUAllocator` 动态检测可用资源
- 可用 GPU 上限：8x RTX A6000 48GB，但每次不一定都是 8 块
- 训练时：所有可用 GPU 跑 LLaMA-Factory + DeepSpeed ZeRO-3
- 推理时：≤8B 模型用 1 卡，>8B 按 `GPUAllocator.plan_inference()` 分配
- CoT 生成：用剩余 GPU 跑最大可用模型（优先 Qwen2-VL-72B 或 InternVL2.5-78B）

## 数据格式要点

- 数据集主文件：`scored.json`（17,022 条），字段：`id/category/conversations/image/img_source/vlm_score`
- 训练格式：LLaMA-Factory sharegpt（`conversations→messages, image→images`）
- 推理输出：JSONL `{id, question, response, image_path1, image_path2, category}`
- 评测输出：JSONL `{id, ..., label_perception, label_str}`（与 `MIS gpt_eval.py` 格式一致）

## 实验命名规范（model_tag 布局，2026-05 重构）

- 主实验：`results/main/E{1-4}/{cfg.name}/{model_tag}/`
- A 实验：`results/prelim/A{1-4}/{cfg.name}/{model_tag}/`
- 消融：`results/ablation/abl_{变量}/{cfg.name}/{model_tag}/`
- **leaf 是 model_tag 不是时间戳**：同一 config 的 base / SFT / 不同 checkpoint 落在不同
  model_tag 目录，互不混淆、可溯源；同 (config, model) 重跑**原地覆盖**。
- `model_tag` 派生（`harness.config.registry.model_tag_from`）：model_path/HF id 的 basename
  （sanitized）；训练 run 用 `cfg.model.name`。例：`--model-path .../models/dreams_internvl3_5`
  → `dreams_internvl3_5`；`OpenGVLab/InternVL3_5-8B-HF` → `InternVL3_5-8B-HF`。
- 每个 run 写 `run_meta.json`（记录真实 `model_path` + timestamp），产物可溯源到具体模型。
- `--experiment-id` 控制路径中的实验号（E1/E2/E3/E4/A1 等）；同 config 跑不同实验自动分目录。
- 评分：`run_eval_only.py --experiment-id E1 --config {cfg} --model-tag {tag}` 自动定位
  `results/{group}/{eid}/{cfg}/{tag}/responses/`，无需 `ls -td` 猜测。
- 设计文档：`docs/run_traceability_plan.md`；迁移脚本：`scripts/migrate_run_layout.py`。

## A 实验约束

A 实验（prelim/）必须在构建 DREAMS 和训练自己模型之前完成：
- 只能用 MIS 已有测试集和公开基准
- 不可使用 `our_dataset/` 中的任何数据训练或评测

## Conda 环境

- 训练/推理/评测：`mis_safety`
- 数据构建：`mis`

## 禁止事项

- 不硬编码 `OPENAI_API_KEY`（从环境变量读取）
- `data_links/` 存放实际数据快照（无外部 symlink），由 `scripts/materialize_data_links.py` 生成；禁止再用 symlink 指向项目外路径
- 不在 `.claude/` 目录存放实验结果
- 不修改 `/mnt/hdd/xuran/MIS/` 下任何文件
