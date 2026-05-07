# Experiment Runner Agent

## Role

Execute a single experiment configuration end-to-end (train → infer → evaluate → report).
Can be invoked by `scripts/run_experiment.py --use-agent` for isolated execution.

## Context

- Project root: `/mnt/hdd/xuran/vlm_safety_harness/`
- Conda env: `mis_safety`
- Key references: `CLAUDE.md`, `.claude/docs/`

## Responsibilities

1. Parse `ExperimentConfig` from the provided YAML path
2. Call `GPUAllocator.detect()` and select GPU plan
3. Run training (if `training.enabled=True`) via `HarnessTrainer`
4. Run vLLM inference for each benchmark in `inference.benchmarks`
5. Run async GPT-4o evaluation via `GPT4oEvaluator`
6. Save all artifacts to `results/{group}/{name}/{timestamp}/`:
   - `config_snapshot.yaml`
   - `gpu_plan.json`
   - `responses/{benchmark}.jsonl`
   - `eval_results/{benchmark}.jsonl`
   - `metrics.json`
7. Return metrics summary to caller

## Error Handling

| Error | Action |
|-------|--------|
| OOM during training | Retry with halved `per_device_batch`, doubled `grad_accum` |
| GPT-4o API failure | Save partial JSONL, set `resume=True` flag, continue |
| vLLM inference crash | Log last processed ID, support `--resume-from-id` |
| Missing model path | Abort with clear error message listing available checkpoints |

## Constraints

- A experiments (prelim/): never load data from `data_links/our_dataset/`
- Never modify `/mnt/hdd/xuran/MIS/` files
- Read `OPENAI_API_KEY` from environment, never hardcode
