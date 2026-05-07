# /gpu-status — Check GPU Status and Plan

Show current GPU utilization and compute dynamic allocation plans for training/inference.

## Usage

```
/gpu-status [--plan-for MODEL_SIZE_B]
```

## Examples

```bash
# Show current GPU status
/gpu-status

# Show GPU status + compute training plan for 8B model
/gpu-status --plan-for 8
```

## Invocation

```python
python /mnt/hdd/xuran/vlm_safety_harness/harness/gpu/allocator.py --status [--plan-for 8]
```

## Output Example

```
[GPU Status]
  GPU0: RTX A6000 | 1200MB/49152MB | util=5%   ← available
  GPU1: RTX A6000 | 35000MB/49152MB | util=98%  ← busy
  GPU2: RTX A6000 | 800MB/49152MB | util=2%    ← available
  ...

Available GPUs: [0, 2, 4, 5, 6, 7] (6 GPUs)

[Training Plan for 8B model]
  gpu_ids: [0, 2, 4, 5]  (4 GPUs)
  deepspeed: zero3
  per_device_batch: 1
  grad_accum: 4
  effective_batch: 16

[Inference Plan for 8B model]
  gpu_ids: [0]  (1 GPU, tensor_parallel=1)
```
