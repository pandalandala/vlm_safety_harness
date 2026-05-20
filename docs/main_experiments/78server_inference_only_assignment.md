# 78server 推理/测试任务分配

> 来源: `docs/paper_guide.md`
> 范围: 只分配不需要训练的任务，包括推理、评测、切片、probe 构建、闭源 API 调用和报告生成。
> 工作目录: `/mnt/hdd/xuran/vlm_safety_harness`
> 运行环境: `...`
> E5 状态: counterfactual consistency 暂时取消；78server 当前不接收 E5 / `our_test_cf` / PD-PC-VS 任务。

## 分配规则

- 可以给 78server: `run_prelim.py`、`run_eval_only.py`、`generate_report.py`、`run_closed_source.py`、带 `--skip-train` 的 `run_experiment.py`。
- 条件可给 78server: DREAMS/MIRage-data checkpoint 已经由训练机产出并同步后的 `--skip-train` 推理。
- 不给 78server: 任何 SFT 训练、E4 的训练变体、Section 6 消融训练、没有 `--skip-train` 且可能触发训练的 `run_experiment.py`/`run_main.py`。

## 立即可跑

### A1-A4 Preliminary Analysis

A1-A4 全部属于推理/评测/标注/probe 构建任务，不需要训练，可直接分配给 78server:

- A1: 构建黑帧 probe，跑 full/text_only 条件，评测 `results/prelim/A1/`。
- A2: GPT-4o 标注 MIS-hard 关系类型，构建 extra relation probe，跑 4 个模型。
- A3: 跑 `mis_easy`、`mis_hard`、`mis_real` 三个 split。
- A4: 跑 MSSBench FPR 分析。

对应命令直接使用 `paper_guide.md` 的 Section 5.1-5.4。

### E1 Base / Tier B / Tier C

可以给 78server:

- Tier A base 三行，必须带 `--skip-train --model-path ...`。
- Tier B 8 个开源 baseline，全部带 `--skip-train`。
- Tier C 闭源模型，前提是 78server 上有对应 API key。

不要给 78server 跑 Section 6.1.1 的 DREAMS/MIRage-data SFT 训练命令。

### E2 Metrics Re-Slicing

E2 只复用 E1 输出，按 `img_source_type` 重算 metrics，不需要重新推理或训练。E1 输出存在后可直接分配。

### E4 V0 Capability

只分配 V0:

```bash
python scripts/run_capability.py --variants V0 --baseline internvl3_5 --benchmarks mmstar mmmu muirbench blink mmt
```

V1/V2/V3/V4 默认是训练变体，不分给 78server。

## Checkpoint 同步后可跑

这些任务依赖训练机先完成 DREAMS/MIRage-data checkpoint。同步后在 78server 上只跑 `--skip-train` 推理和评测。

### E1 SFT Checkpoint Inference

```bash
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-train
python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-train
python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-train

python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-train
python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-train
python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-train
```

### E3 Cross-Benchmark Safety

Run with `--skip-train` only:

```bash
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss
python scripts/run_experiment.py main/main_dreams_qwen3_5.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss
python scripts/run_experiment.py main/main_dreams_llava_ov.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss

python scripts/run_experiment.py main/main_baseline_mirage_data_internvl3_5.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss
python scripts/run_experiment.py main/main_baseline_mirage_data_qwen3_5.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss
python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml --skip-train --benchmarks advbench safebench figstep mm_safety jailbreakv siuo mss
```

### E5 Counterfactual Consistency（已暂时取消，归档保留）

> 当前不分配给 78server。以下前置条件和分配内容仅作为未来恢复 E5 时的归档参考。

归档前置条件:

- `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/test_cf.json` 已存在。
- DREAMS/MIRage-data checkpoints 已同步。
- 所有 `run_experiment.py` 都带 `--skip-train`。

归档分配内容:

- 6 个 checkpoint 跑 `our_test`。
- 6 个 checkpoint 跑 `our_test_cf`。
- 再聚合 PD/PC/VS；如果使用 `run_main.py`，也必须加 `--skip-train`。

## 报告生成

已有 metrics/responses 后可分配:

```bash
python scripts/generate_report.py --group main --experiment-set e1_per_category --format markdown

python scripts/generate_report.py --group main --experiment-set e3_full --format markdown

# [归档保留 / 当前不运行] E5 VS 阈值敏感性分析
# python scripts/run_eval_only.py --responses results/main/main_dreams_internvl3_5/YYYYMMDD/responses/ --vs-thresholds 0.1 0.2 0.3 0.5
```

## 建议队列

立刻启动:

1. A1-A4
2. E1 Tier A base
3. E1 Tier B
4. E4 V0
5. 已有输出上的 E2/report

等待 checkpoint 同步后:

1. E1 DREAMS/MIRage-data `--skip-train`
2. E3 `--skip-train`
3. Appendix A/B
4. E5 `--skip-train`（归档保留，当前不运行）
