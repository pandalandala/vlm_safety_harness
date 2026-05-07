# /run-exp — Run a Single Experiment

Execute a complete experiment (train → infer → evaluate) based on a YAML config file.

## Usage

```
/run-exp [config_path] [--skip-train] [--skip-inference] [--model-path PATH] [--limit N] [--dry-run]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `config_path` | Relative to `configs/experiments/` or absolute path |
| `--skip-train` | Load existing checkpoint instead of training |
| `--skip-inference` | Use existing responses, skip to evaluation |
| `--model-path PATH` | Explicit checkpoint path (implies `--skip-train`) |
| `--limit N` | Only run inference on first N samples (quick validation) |
| `--dry-run` | Print plan without executing |
| `--force` | Re-run even if same config already completed |

## Examples

```bash
# Run a complete A1 experiment (inference-only, no training needed)
/run-exp prelim/A1_textual_shortcut.yaml --limit 20

# Run full main experiment
/run-exp main/main_dreams_internvl.yaml

# Skip training, use existing checkpoint
/run-exp main/main_dreams_internvl.yaml --model-path /mnt/hdd/xuran/vlm_safety_harness/models/internvl_dreams/

# Only run GPT-4o evaluation on existing responses
/run-exp main/main_dreams_internvl.yaml --skip-train --skip-inference
```

## What This Does

1. Load config from `configs/experiments/{config_path}`
2. `GPUAllocator.detect()` → dynamic GPU plan
3. Training (if enabled): LLaMA-Factory SFT via `HarnessTrainer`
4. Inference: vLLM batch inference on all configured benchmarks
5. Evaluation: async GPT-4o evaluation (concurrent requests)
6. Save artifacts to `results/{group}/{name}/{YYYYMMDD_HHMMSS}/`
7. Print summary table to stdout

## Invocation

```python
python /mnt/hdd/xuran/vlm_safety_harness/scripts/run_experiment.py \
  --config configs/experiments/prelim/A1_textual_shortcut.yaml \
  [options]
```
