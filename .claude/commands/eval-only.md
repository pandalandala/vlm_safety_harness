# /eval-only — GPT-4o Evaluation Only

Run GPT-4o evaluation on existing model response files (skip training and inference).

## Usage

```
/eval-only [responses_dir] [--resume] [--judge llama_guard]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `responses_dir` | Directory containing `{benchmark}.jsonl` response files |
| `--resume` | Continue from partially completed evaluation |
| `--judge` | `gpt4o` (default) or `llama_guard` |

## Examples

```bash
# Evaluate responses from a previous inference run
/eval-only results/main/main_dreams_internvl/20260505_143022/responses/

# Resume interrupted evaluation
/eval-only results/main/main_dreams_internvl/20260505_143022/responses/ --resume

# Use LlamaGuard instead (for A5 judge reliability experiment)
/eval-only results/prelim/A5_judge_reliability/responses/ --judge llama_guard
```

## Invocation

```python
python /mnt/hdd/xuran/vlm_safety_harness/scripts/run_eval_only.py \
  --responses-dir results/main/main_dreams_internvl/20260505_143022/responses/ \
  [options]
```
