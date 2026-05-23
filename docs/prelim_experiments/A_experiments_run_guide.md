# A 实验运行指南

生成时间：2026-05-06  
适用目录：`/mnt/hdd/xuran/vlm_safety_harness/`  
Conda env：`mis_safety`  
所有命令在 `vlm_safety_harness/` 根目录下运行。

---

## 模型清单

### Base VLMs（无 safety SFT）

| Short name | HF ID | Arch |
|-----------|-------|------|
| IVL | `OpenGVLab/InternVL3_5-8B` | `internvl` |
| Qwen | `Qwen/Qwen3.5-9B` | `qwen2vl` |
| LLaVA | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | `llava` |

### MIRage（safety-SFT 对比基线）

| Short name | HF ID | Arch |
|-----------|-------|------|
| MIRage-IVL | `Tuwhy/InternVL2.5-8B-MIRage` | `internvl` |
| MIRage-Qwen | `Tuwhy/Qwen2-VL-7B-MIRage` | `qwen2vl` |

---

## 运行前检查

```bash
# 确认 OPENAI_API_KEY 已设置
echo $OPENAI_API_KEY

# GPU 状态（需至少 1 块空闲 ≥20GB）
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# Env 可用
python -c "import vllm; print(vllm.__version__)"
```

---

## 执行顺序

```
Step 0: 建 probes（A1/A2 依赖）
Step 1: A3（无 probe 依赖，立刻可跑）
Step 2: A4（需先下载 MSSBench）
Step 3: A1（Step 0 完成后）
Step 4: A2（需手工 probe JSONL，工作量最大）
```

每步先跑 3 个 base VLMs，再跑 2 个 MIRage 模型。

---

## Step 0：构建 Probes（A1/A2 依赖）

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/run_prelim.py --build-probes
```

输出：
- `results/prelim/probes/probe_text_only.json`         (A1: easy 黑框)
- `results/prelim/probes/probe_text_only_hard.json`    (A1: hard 黑框)
- `results/prelim/probes/mis_hard_relation_annotated.json` (A2 占位)
- `results/prelim/probes/relation_type_probe.json`     (A2 合并占位)

### Step 0b：构建 extra_relation_probe（A2 DREAMS 关系分类）

**首次运行**（实时 API，扫前 5000）：
```bash
export OPENAI_API_KEY="sk-..."
conda activate mis_safety
python scripts/build_a2_dreams_probe.py \
  --min-per-type 50 \
  --max-candidates 5000 \
  --model gpt-4o
```

**续扫剩余 12022 条**（已扫前 5000，需从 rank 5001 继续）：
```bash
export OPENAI_API_KEY="sk-..."
conda activate mis_safety

# 实时模式（快，pay per token）
python scripts/build_a2_dreams_probe.py \
  --resume \
  --skip-n 5000 \
  --max-candidates 12022 \
  --min-per-type 50 \
  --model gpt-4o

# Batch API 模式（慢，最长 24h，省 50% 费用）
python scripts/build_a2_dreams_probe.py \
  --resume \
  --skip-n 5000 \
  --max-candidates 12022 \
  --min-per-type 50 \
  --model gpt-4o \
  --use-batch \
  --batch-poll-interval 120
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--resume` | 读取现有输出文件，跳过已有 ID，append 新结果 |
| `--skip-n N` | 跳过排序后前 N 条候选（避免重扫已处理样本） |
| `--max-candidates N` | 从 skip-n 之后最多扫 N 条 |
| `--use-batch` | 启用 OpenAI Batch API（异步，50% 折扣） |
| `--batch-poll-interval N` | Batch 轮询间隔秒数（默认 60） |
| `--model MODEL` | OpenAI 模型，默认 `gpt-4o` |
| `--concurrency N` | 实时模式并发数，默认 20 |

输出：`results/prelim/probes/extra_relation_probe.jsonl`

---

## Step 1：A3 Synthetic-Real Gap

```bash
cd /mnt/hdd/xuran/vlm_safety_harness

# --- Base VLMs ---
python scripts/run_prelim.py --experiment A3 --models OpenGVLab/InternVL3_5-8B

python scripts/run_prelim.py --experiment A3 --models Qwen/Qwen3.5-9B

python scripts/run_prelim.py --experiment A3 --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct

# --- MIRage ---
python scripts/run_prelim.py --experiment A3 --models Tuwhy/InternVL2.5-8B-MIRage

python scripts/run_prelim.py --experiment A3 --models Tuwhy/Qwen2-VL-7B-MIRage
```

**每次跑 3 个 benchmark**：`mis_easy` / `mis_hard` / `mis_real`  
**关键指标**：`Gap = ASR(mis_real) − ASR(mis_easy)`  
**结果目录**：`results/prelim/A3_synthetic_real_gap/{timestamp}/metrics.json`

---

## Step 2：A4 Counterfactual

### 2a. 下载 MSSBench

```bash
huggingface-cli download kzhou35/mssbench --repo-type dataset --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mssbench
```

### 2b. 确认 test_path（如文件名不是 `mssbench.json`）

```bash
ls /mnt/hdd/xuran/vlm_safety_harness/data_links/mssbench/
```

若文件名不同，编辑 `configs/experiments/prelim/A4_counterfactual.yaml`：
```yaml
dataset:
  test_path: /mnt/hdd/xuran/vlm_safety_harness/data_links/mssbench/<实际文件名>
```

### 2c. 运行 A4

```bash
cd /mnt/hdd/xuran/vlm_safety_harness

# --- Base VLMs ---
python scripts/run_prelim.py --experiment A4 --models OpenGVLab/InternVL3_5-8B

python scripts/run_prelim.py --experiment A4 --models Qwen/Qwen3.5-9B

python scripts/run_prelim.py --experiment A4 --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct

# --- MIRage ---
python scripts/run_prelim.py --experiment A4 --models Tuwhy/InternVL2.5-8B-MIRage

python scripts/run_prelim.py --experiment A4 --models Tuwhy/Qwen2-VL-7B-MIRage
```

**每次跑 3 个 benchmark**：`mssbench_safe` / `mssbench_unsafe` / `mis_hard`  
**关键指标**：MSS-Unsafe ASR↓, MSS-Safe FPR↓, Pair Consistency↑  
**结果目录**：`results/prelim/A4_counterfactual/{timestamp}/metrics.json`

---

## Step 3：A1 Textual Shortcut（Step 0 完成后）

```bash
cd /mnt/hdd/xuran/vlm_safety_harness

# --- Base VLMs ---
python scripts/run_prelim.py --experiment A1 --models OpenGVLab/InternVL3_5-8B

python scripts/run_prelim.py --experiment A1 --models Qwen/Qwen3.5-9B

python scripts/run_prelim.py --experiment A1 --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct

# --- MIRage ---
python scripts/run_prelim.py --experiment A1 --models Tuwhy/InternVL2.5-8B-MIRage

python scripts/run_prelim.py --experiment A1 --models Tuwhy/Qwen2-VL-7B-MIRage
```

**每次跑 4 个 benchmark**：`mis_easy` / `mis_hard` / `probe_text_only` / `probe_text_only_hard`  
**关键指标**：`ΔASR = ASR(mis_easy) − ASR(probe_text_only)`  
**结果目录**：`results/prelim/A1_textual_shortcut/{timestamp}/metrics.json`

---

## Step 4：A2 Relation Pattern Coverage（工作量最大）

### 4a. GPT-4o 标注 MIS-hard 关系类型

占位文件已在 Step 0 生成：`results/prelim/probes/mis_hard_relation_annotated.json`  
需用 GPT-4o 填充每条记录的 `relation_type` 字段：
```
"tool_target" | "before_after" | "identity_linking" | "context_shift"
```
（此步骤目前无独立脚本，需手动调用 GPT-4o evaluator 或单独实现）

### 4b. 手工构建非标准关系 probe

创建文件：`results/prelim/probes/extra_relation_probe.jsonl`  
每行格式：
```json
{"id": "rel_ba_001", "image_path1": "/abs/path/img1.jpg", "image_path2": "/abs/path/img2.jpg", "question": "...", "category": "Violence", "relation_type": "before_after"}
```
图片来源：COCO（用 `fiftyone`）/ OpenImages（`/mnt/hdd/xuran/anaconda3/lib/python3.13/site-packages/openimages/`）

### 4c. 重新生成合并 probe（包含手工样本）

```bash
python scripts/run_prelim.py --build-probes --build-probes-experiment A2
```

### 4d. 运行 A2

```bash
cd /mnt/hdd/xuran/vlm_safety_harness

# --- Base VLMs ---
python scripts/run_prelim.py --experiment A2 --models OpenGVLab/InternVL3_5-8B

python scripts/run_prelim.py --experiment A2 --models Qwen/Qwen3.5-9B

python scripts/run_prelim.py --experiment A2 --models lmms-lab/LLaVA-OneVision-1.5-8B-Instruct

# --- MIRage ---
python scripts/run_prelim.py --experiment A2 --models Tuwhy/InternVL2.5-8B-MIRage

python scripts/run_prelim.py --experiment A2 --models Tuwhy/Qwen2-VL-7B-MIRage
```

---

## 结果查看

```bash
# 所有 metrics.json 位置
find /mnt/hdd/xuran/vlm_safety_harness/results/prelim/ -name metrics.json | sort

# 某次结果
cat results/prelim/A3_synthetic_real_gap/<timestamp>/metrics.json

# config snapshot（确认是哪个模型）
cat results/prelim/A3_synthetic_real_gap/<timestamp>/config_snapshot.yaml | grep hf_path
```

填写到：`docs/result_tables.md`

---

## 调试选项

```bash
# 只跑 10 条测试流程
python scripts/run_prelim.py --experiment A3 --models OpenGVLab/InternVL3_5-8B --limit 10

# 跳过 GPT-4o eval（省费用，只看推理输出）
python scripts/run_prelim.py --experiment A3 --models OpenGVLab/InternVL3_5-8B --skip-eval

# 打印命令不执行
python scripts/run_prelim.py --experiment A3 --dry-run
```
