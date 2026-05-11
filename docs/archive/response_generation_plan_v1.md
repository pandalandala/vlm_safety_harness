# Response Generation for DREAMS Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate CoT + natural-language safety responses for all 17,022 DREAMS training samples using Qwen3.5-122B-A10B (VLM), filling the empty `conversations[1]["value"]` field in `train.json`.

**Architecture:** New `generate_responses.py` script + extended `CoTGenerator` with a `full_response` mode. VLM receives both images + question via vLLM multi-modal API; outputs structured `<safety_analysis>` block followed by a natural safety response. Results saved to `train_annotated.json` with resume support.

**Tech Stack:** vLLM (multi-modal), Qwen3.5-122B-A10B (tp=8), PIL, CUDA, Python 3.12, mis_safety conda env

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROPOSED PROMPT (Core Deliverable — Review Before Implementing)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### System Prompt (`RESPONSE_GENERATION_SYSTEM`)

```
You are an expert AI safety researcher specializing in multimodal threat analysis. Your task is to generate high-quality safety training data for vision-language models.

You will be shown two images alongside a harmful question. These images work TOGETHER to provide visual context for the harmful request:
- Image 1 typically depicts an object, substance, or tool
- Image 2 typically depicts an environment, target location, or contextual scenario

The harmful intent often emerges from the COMBINATION of both images, not from either image alone.

Generate your response in two parts:

PART 1 — Structured Analysis (always include, wrap in XML tags):
<safety_analysis>
Step 1 [Object Recognition]: Identify the key objects, substances, or entities shown in Image 1 and Image 2 respectively.
Step 2 [Relationship Analysis]: Explain specifically how the combination of Image 1 and Image 2 relates to the harmful question. What makes this particular pairing dangerous? Would either image alone be concerning without the other?
Step 3 [Harm Assessment]: (a) What specific harm could result from fulfilling this request? (b) Is this harm grounded in these specific images or would any similar images suffice? (c) Counterfactual check — would the request be substantially safer if Image 1 depicted a benign object instead?
Conclusion: [Unsafe] — {one concise sentence summarizing the core safety concern}
</safety_analysis>

PART 2 — Safety Response (natural prose, no headers, immediately after closing tag):
Write 3–5 sentences in natural conversational style that:
- Opens by acknowledging what you observe across both images (ground your response visually)
- Explains why this specific visual combination raises safety concerns
- Declines to fulfill the harmful request with clear reasoning
- Redirects toward safe alternatives or educational context where appropriate
- Does NOT provide step-by-step instructions, specific technical procedures, chemical/biological formulas, or any operationally useful harmful guidance
```

### User Message Template

```
{query}
```
(Images are prepended as multi-modal tokens via vLLM; no `<image>` placeholder needed in text)

### Why This Prompt Is Better Than MIS Paper Prompt

| Aspect | MIS Paper | Our Prompt |
|--------|-----------|------------|
| Image structure | Generic "two images" | Explicit Image1=object, Image2=context |
| Combined effect | Not mentioned | Explicitly requires analyzing combination |
| Counterfactual reasoning | Not present | Step 3(c) checks if swapping Image1 makes it safe (aligned with A4 experiment) |
| Output structure | Free-form prose | CoT tags + prose (matches `use_cot_labels=true`) |
| Visual grounding | Implicit | Explicitly required in Part 2 |
| Generic refusal prevention | "not too short" | "must reference specific visual content" |

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `docs/Models_List.md` | Modify | Add Qwen3.5-122B-A10B to annotator section |
| `harness/inference/model_configs.py` | Modify | Add `qwen3_5_vl` architecture config (tp=8) |
| `harness/training/cot_generator.py` | Modify | Add `RESPONSE_GENERATION_SYSTEM` prompt + `full_response` mode |
| `scripts/generate_responses.py` | Create | CLI entry point for response generation |

---

## Task 1: Add Model to Docs & Config

**Files:**
- Modify: `docs/Models_List.md`
- Modify: `harness/inference/model_configs.py`

- [ ] **Step 1: Add to Models_List.md**

In `docs/Models_List.md`, add a new section for annotator models (after existing content):

```markdown
### Annotator / Response Generator Models

https://huggingface.co/Qwen/Qwen3.5-122B-A10B
```

- [ ] **Step 2: Add vLLM architecture config**

In `harness/inference/model_configs.py`, add to the `ARCH_CONFIGS` dict:

```python
"qwen3_5_vl": VLLMModelConfig(
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    mm_processor_kwargs={
        "min_pixels": 200704,   # 448*448
        "max_pixels": 1003520,  # same as Qwen2.5-VL
    },
),
```

- [ ] **Step 3: Commit**

```bash
git add docs/Models_List.md harness/inference/model_configs.py
git commit -m "feat: add Qwen3.5-122B-A10B to model list and vLLM arch configs"
```

---

## Task 2: Extend CoTGenerator with Full Response Mode

**Files:**
- Modify: `harness/training/cot_generator.py`

- [ ] **Step 1: Add the new system prompt constant**

After the existing `COT_SYSTEM_PROMPT` constant (line ~30), add:

```python
RESPONSE_GENERATION_SYSTEM = """You are an expert AI safety researcher specializing in multimodal threat analysis. Your task is to generate high-quality safety training data for vision-language models.

You will be shown two images alongside a harmful question. These images work TOGETHER to provide visual context for the harmful request:
- Image 1 typically depicts an object, substance, or tool
- Image 2 typically depicts an environment, target location, or contextual scenario

The harmful intent often emerges from the COMBINATION of both images, not from either image alone.

Generate your response in two parts:

PART 1 — Structured Analysis (always include, wrap in XML tags):
<safety_analysis>
Step 1 [Object Recognition]: Identify the key objects, substances, or entities shown in Image 1 and Image 2 respectively.
Step 2 [Relationship Analysis]: Explain specifically how the combination of Image 1 and Image 2 relates to the harmful question. What makes this particular pairing dangerous? Would either image alone be concerning without the other?
Step 3 [Harm Assessment]: (a) What specific harm could result from fulfilling this request? (b) Is this harm grounded in these specific images or would any similar images suffice? (c) Counterfactual check — would the request be substantially safer if Image 1 depicted a benign object instead?
Conclusion: [Unsafe] — {one concise sentence summarizing the core safety concern}
</safety_analysis>

PART 2 — Safety Response (natural prose, no headers, immediately after closing tag):
Write 3–5 sentences in natural conversational style that:
- Opens by acknowledging what you observe across both images (ground your response visually)
- Explains why this specific visual combination raises safety concerns
- Declines to fulfill the harmful request with clear reasoning
- Redirects toward safe alternatives or educational context where appropriate
- Does NOT provide step-by-step instructions, specific technical procedures, chemical/biological formulas, or any operationally useful harmful guidance"""
```

- [ ] **Step 2: Add `mode` parameter and routing to `__init__` and `generate_batch`**

In `CoTGenerator.__init__`, add:
```python
self.mode = mode  # "cot_only" (default) or "full_response"
```

Add `mode: str = "cot_only"` parameter after `temperature` in `__init__` signature.

In `generate_batch`, the existing routing is already correct — it dispatches to `_generate_vllm` or `_generate_openai`. No change needed there.

- [ ] **Step 3: Update `_build_vllm_prompt` to use the right system prompt**

In `_build_vllm_prompt`, change the system message to be conditional:

```python
def _build_vllm_prompt(self, r: dict) -> dict:
    from PIL import Image
    imgs = []
    for key in ("image_path1", "image_path2"):
        p = r.get(key, "")
        if p:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))

    system = (
        RESPONSE_GENERATION_SYSTEM
        if self.mode == "full_response"
        else COT_SYSTEM_PROMPT
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "placeholder1"}},
                {"type": "image_url", "image_url": {"url": "placeholder2"}},
                {"type": "text", "text": r["question"]},
            ],
        },
    ]
    return {"prompt": messages, "multi_modal_data": {"image": imgs}}
```

- [ ] **Step 4: Update `_generate_vllm` output field for full_response mode**

In `_generate_vllm`, change the result building to:

```python
results = []
for r, out in zip(records, outputs):
    text = out.outputs[0].text.strip()
    if self.mode == "full_response":
        results.append({**r, "gpt_response": text})
    else:
        results.append({**r, "cot_response": text})
return results
```

- [ ] **Step 5: Commit**

```bash
git add harness/training/cot_generator.py
git commit -m "feat: add full_response mode to CoTGenerator with DREAMS-optimized prompt"
```

---

## Task 3: Create `scripts/generate_responses.py`

**Files:**
- Create: `scripts/generate_responses.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Generate CoT + safety responses for DREAMS training data using Qwen3.5-122B-A10B.

Usage:
    python scripts/generate_responses.py \
        --dataset /mnt/hdd/xuran/mis_dataset_builder/dataset \
        --model Qwen/Qwen3.5-122B-A10B \
        --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json \
        --batch-size 4 \
        --resume
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.gpu.allocator import GPUAllocator
from harness.training.cot_generator import CoTGenerator


DATASET_ROOT = Path("/mnt/hdd/xuran/mis_dataset_builder/dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DATASET_ROOT), help="Dataset root dir")
    p.add_argument("--input", default="train.json", help="Input filename under --dataset")
    p.add_argument("--output", default=None, help="Output JSON path (default: train_annotated.json)")
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_train_records(input_path: Path, dataset_root: Path) -> list[dict]:
    """Convert train.json sharegpt format to flat records with absolute image paths."""
    with open(input_path) as f:
        raw = json.load(f)

    records = []
    for r in raw:
        human_val = r["conversations"][0]["value"]
        # Strip <image> tokens
        question = human_val.replace("<image>\n", "").strip()

        img_paths = r.get("image", [])
        record = {
            "id": r["id"],
            "category": r.get("category", ""),
            "question": question,
            "image_path1": str(dataset_root / img_paths[0]) if len(img_paths) > 0 else "",
            "image_path2": str(dataset_root / img_paths[1]) if len(img_paths) > 1 else "",
            "_original": r,  # keep for output reconstruction
        }
        records.append(record)
    return records


def merge_responses(original_records: list[dict], annotated: list[dict]) -> list[dict]:
    """Merge gpt_response back into original train.json format."""
    response_map = {r["id"]: r["gpt_response"] for r in annotated if r.get("gpt_response")}
    result = []
    for r in original_records:
        orig = r["_original"]
        resp = response_map.get(r["id"], "")
        # Fill the gpt turn
        orig["conversations"][1]["value"] = resp
        result.append(orig)
    return result


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    input_path = dataset_root / args.input
    output_path = Path(args.output) if args.output else dataset_root / "train_annotated.json"

    # Detect GPUs — 122B MoE needs all available
    allocator = GPUAllocator()
    print(allocator.status_report())
    available = allocator.get_available()
    gpu_ids = [g.index for g in available]
    tp = len(gpu_ids)
    print(f"[gen] GPUs: {gpu_ids} (tp={tp})")
    print(f"[gen] Model: {args.model}")

    # Load + flatten records
    all_records = load_train_records(input_path, dataset_root)
    if args.limit:
        all_records = all_records[: args.limit]

    # Resume: skip already done
    done_ids: set = set()
    annotated_so_far: list[dict] = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        done_ids = {
            r["id"]
            for r in existing
            if r.get("conversations", [{}] * 2)[1].get("value", "")
        }
        annotated_so_far = existing
        print(f"[gen] Resuming: {len(done_ids)} done, {len(all_records) - len(done_ids)} remaining")

    pending = [r for r in all_records if r["id"] not in done_ids]
    if not pending:
        print("[gen] All records already annotated.")
        return

    # Init generator in full_response mode
    generator = CoTGenerator(
        model_path=args.model,
        backend="vllm",
        gpu_ids=gpu_ids,
        tensor_parallel_size=tp,
        max_tokens=args.max_tokens,
        temperature=0.1,
        mode="full_response",
    )

    # Process in batches with checkpoint
    bs = args.batch_size
    all_annotated_flat = list(annotated_so_far)  # existing full train.json records
    done_map = {r["id"]: r for r in annotated_so_far}

    for i in range(0, len(pending), bs):
        batch = pending[i : i + bs]
        results = generator.generate_batch(batch)

        # Reconstruct train.json format for this batch
        for r in results:
            orig = r["_original"].copy()
            orig["conversations"][1]["value"] = r.get("gpt_response", "")
            done_map[orig["id"]] = orig

        # Checkpoint every batch
        checkpoint_records = list(done_map.values())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(checkpoint_records, f, ensure_ascii=False, indent=2)

        n_done = i + len(batch)
        print(f"[gen] {n_done}/{len(pending)} processed — saved to {output_path}")

    print(f"[done] {len(done_map)} records written to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /mnt/hdd/xuran/vlm_safety_harness/scripts/generate_responses.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_responses.py
git commit -m "feat: add generate_responses.py for DREAMS training data annotation"
```

---

## Task 4: Verification (Dry Run on 5 Samples)

Before running on 17K samples, verify the pipeline end-to-end.

- [ ] **Step 1: Run on 5 samples with a small model first (optional sanity check)**

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
conda run -n mis_safety python scripts/generate_responses.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --limit 5 \
    --output /tmp/test_responses.json \
    --no-resume
```

Expected: 5 records in `/tmp/test_responses.json` with non-empty `conversations[1]["value"]` containing `<safety_analysis>` tags.

- [ ] **Step 2: Inspect output quality**

```bash
python3 -c "
import json
d = json.load(open('/tmp/test_responses.json'))
for r in d[:3]:
    print('ID:', r['id'])
    print('Category:', r['category'])
    print('Response:', r['conversations'][1]['value'][:300])
    print('---')
"
```

Expected: Response starts with `<safety_analysis>`, contains all 3 steps + Conclusion, followed by natural prose after `</safety_analysis>`.

- [ ] **Step 3: Run full dataset with Qwen3.5-122B-A10B**

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
conda run -n mis_safety python scripts/generate_responses.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --batch-size 4 \
    --max-tokens 768 \
    --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json \
    --resume
```

Expected runtime: ~15,319 samples ÷ 4 batch × ~10s/batch ≈ 6–10 hours (depending on GPU speed).

- [ ] **Step 4: Validate output completeness**

```bash
python3 -c "
import json
d = json.load(open('/mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json'))
total = len(d)
filled = sum(1 for r in d if r['conversations'][1]['value'].strip())
has_cot = sum(1 for r in d if '<safety_analysis>' in r['conversations'][1]['value'])
print(f'Total: {total}, Filled: {filled}, With CoT: {has_cot}')
"
```

Expected: `Total: 15319, Filled: 15319, With CoT: 15319`

---

## Notes

- **`_original` field**: Used internally during processing; NOT written to final JSON output (it's a temp key overwritten when merging back).
- **Resume safety**: Checkpoint written after every batch; safe to kill and restart.
- **Model path**: Once downloaded, user may need to pass local path: `--model /mnt/hdd/xuran/models/Qwen3.5-122B-A10B`
- **GPU requirement**: 122B MoE @ BF16 ≈ 244GB VRAM. Needs all 8× A6000 (384GB total). Confirm GPUs are free before running.
- **`_original` key leak**: The `merge_responses` function in the script uses `r["_original"]` which contains the `_original` key itself. The final JSON uses `orig = r["_original"].copy()` and modifies it — the `_original` key isn't in the train.json records themselves, so the final output is clean.
