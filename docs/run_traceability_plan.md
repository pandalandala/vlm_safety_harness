# Run 可溯源重构 plan

## Context（问题）

E1 多个模型共用同一 config（`main_dreams_*` 同时被 base 推理和 DREAMS SFT 推理使用，仅靠 `--model-path` 区分），
旧路径用时间戳作 leaf：`results/{group}/{eid}/{cfg}/{YYYYMMDD_HHMMSS}/`。
后果：

- `--model-path` 未持久化 → run 产物无法溯源到具体模型
- base 与 SFT 混在同一 config 目录，只能靠时间戳，`ls -td|head -1` 会取错（实测 InternVL/LLaVA 的 latest 竟是 base）
- 评分定位靠 `ls -td|head -1`，不认模型身份

## 目标

1. 每个 run 可溯源（产物记录真实 model_path）
2. base / SFT 不再混到一起
3. 最新 run 覆盖之前的（同模型同 config → 同目录，原地覆盖）
4. 评分自动从对应目录取结果

## 新路径方案

时间戳 leaf → **model_tag** leaf：

```
results/{group}/{experiment_id}/{cfg.name}/{model_tag}/
  ├── run_meta.json        ← {model_path, model_tag, timestamp, experiment_id, group, config_name}
  ├── config_snapshot.yaml
  ├── config_hash.txt
  ├── experiment_id.txt
  ├── responses/*.jsonl
  ├── eval_results/*.jsonl
  └── metrics.json
```

### model_tag 派生（`registry.model_tag_from`）

取 model 标识的 basename，sanitize（非 `[A-Za-z0-9._-]` → `_`）：

| 来源 | model_tag |
|------|-----------|
| `--model-path .../models/dreams_internvl3_5` | `dreams_internvl3_5` |
| `--model-path OpenGVLab/InternVL3_5-8B-HF`（base） | `InternVL3_5-8B-HF` |
| `--model-path .../models/mirage_data_qwen3_5` | `mirage_data_qwen3_5` |
| 训练 run（无 --model-path） | `cfg.model.name`（如 `internvl3_5_8b`） |

→ base 与 SFT tag 不同 → 不同目录 → 不混。同模型重跑 → 同目录 → 覆盖。

## 代码改动

| 文件 | 改动 |
|------|------|
| `harness/config/registry.py` | 新 `model_tag_from(s)`；`make_run_dir(cfg, experiment_id, model_tag, model_path)` 用 model_tag 作 leaf + 写 `run_meta.json`；`is_completed(cfg, experiment_id, model_tag)` 查 model_tag 目录；`find_runs`/`get_best_run` 从 run_meta 读 timestamp+model |
| `scripts/run_experiment.py` | 解析 model_tag（model_path 优先，训练 run 用 cfg.model.name），传入 make_run_dir/is_completed；run_meta 记录真实 model_path |
| `scripts/run_eval_only.py` | 新增 `--experiment-id/--config/--model-tag` 自动拼 responses 路径（仍兼容 `--responses`） |

## 已有结果迁移

根据 `logs/main/*.log` 的 "loading weights file" / "name_or_path" 行还原每个 run 真实模型，
把带 responses 的 run 移到新布局 `{cfg}/{model_tag}/`；同 (cfg, model_tag) 多个 → 保留最新；补写 run_meta.json。
迁移脚本：`scripts/migrate_run_layout.py`（一次性，dry-run 优先）。

已确认的 E1 latest 真身：
- `main_dreams_internvl3_5/20260519_223152` = **BASE**（InternVL3_5-8B-HF）
- `main_dreams_internvl3_5/20260519_220031` = DREAMS SFT
- `main_dreams_qwen3_5/20260521_003825` = DREAMS SFT
- `main_dreams_llava_ov/20260520_143036` = **BASE**（LLaVA-OneVision-1.5-8B-Instruct）
- `main_dreams_llava_ov/20260519_222931` = DREAMS SFT

## 文档改动

- `docs/paper_guide.md`：所有 `$(ls -td .../{cfg}/*/ | head -1)responses/` → 确定性 `.../{cfg}/{model_tag}/responses/`
- `models/MODELS.md`：link_latest 改为读新布局
- `CLAUDE.md`：实验命名规范更新为 model_tag leaf

## 验证

1. dry-run migrate，核对每个 run 的 model_tag 判定
2. 跑一次 `run_experiment --skip-train --model-path` 确认落到 `{cfg}/{model_tag}/` + run_meta 正确
3. `run_eval_only --experiment-id E1 --config main_dreams_internvl3_5 --model-tag dreams_internvl3_5` 能定位
