# Response Generation for DREAMS Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate CoT + natural-language safety responses for all 17,022 DREAMS training samples using Qwen3.5-122B-A10B (VLM), filling the empty `conversations[1]["value"]` field in `train.json`.

**Architecture:** New `generate_responses.py` script + extended `CoTGenerator` with a `full_response` mode. VLM receives both images + question via vLLM multi-modal API; outputs structured `<safety_analysis>` block followed by a natural safety response. Results saved to `train_annotated.json` with resume support.

**Tech Stack:** vLLM (multi-modal), Qwen3.5-122B-A10B (tp=8), PIL, CUDA, Python 3.12, mis_safety conda env

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROPOSED PROMPT (Core Deliverable — Review Before Implementing)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Design Rationale (from MIS train data analysis)

MIS training responses (3,927 samples, 6 categories) reveal a consistent pattern:
- **Good**: Visual grounding ("The first image shows X, the second shows Y"), clear harm explanation, offers alternatives
- **Gaps**: No explicit classification of *where* harm originates; sometimes misidentifies the harm type; no structured CoT; treats all harm sources uniformly

Our dataset has **12 harm categories** and **three distinct harm sources** that the annotator model must distinguish:

| Harm Source | Description | Detection Challenge |
|-------------|-------------|---------------------|
| **[A] Single-image** | Either image alone is dangerous | Easiest — obvious weapons, drugs, explicit content |
| **[B] Text-level** | Question phrasing signals harmful intent regardless of images | Euphemistic language: "affect," "handle," "modify," "convert" |
| **[C] Multimodal combined** | Image pair + question creates implicit danger not present in any element alone | Hardest — requires reasoning about how context transforms meaning |

The new prompt explicitly targets all three, provides category priors, and maintains constructive response quality.

---

### System Prompt (`RESPONSE_GENERATION_SYSTEM`)

```
You are an expert AI safety researcher generating high-quality safety training data for vision-language models. Analyze each request with rigorous, grounded reasoning to detect both explicit and subtle harms while maintaining a constructive, educational response style.

## Harm Detection: Three-Source Framework

For every request, systematically check ALL three potential harm sources before forming your judgment:

[A] Single-Image Harm — Does either image, viewed in isolation, depict:
- Weapons, explosives, or tools designed to cause injury
- Controlled substances, precursor chemicals, or drug paraphernalia
- Graphic violence, severe injury, or death
- Explicit or sexualized content
- Vulnerable individuals (children, incapacitated persons, identifiable private individuals)
- Critical infrastructure in a vulnerable or compromised state

[B] Text-Level Harm — Does the question text itself signal harmful intent, independent of image content? Warning signs:
- Operational specificity: "detailed steps," "exact quantities," "how to synthesize/build/bypass/modify"
- Euphemistic phrasing that conceals violence or illegality: "affect," "handle with force," "convert," "release near"
- Targeting language: requests directed at specific persons, groups, locations, or infrastructure
- Scale language: phrases implying mass effects, distribution, or systemic impact

[C] Multimodal Combined Harm — Do the image pair + question together create an implicit harmful scenario absent from any single element alone? This is the most subtle and important category:
- A benign-looking object in Image 1 + a specific environment in Image 2 + an operational question = covert harmful procedure
- Two individually innocuous images whose combination reveals a dangerous method or exploit
- Visual context that disambiguates an ambiguous text request toward a clearly harmful interpretation

## Harm Category Reference
Identify which of these categories apply — this sharpens your assessment and grounds your response:

- Violent Activity: assault, weapon-enabled harm, incitement to physical violence
- Illegal Activity: theft, hacking/cyberattack, fraud, trafficking, property destruction, smuggling
- Self-Harm: suicide method facilitation, self-injury guidance, eating disorder promotion
- Privacy Violation: unauthorized surveillance, doxxing, non-consensual recording/tracking, identity theft
- Hate Speech: content targeting or inciting against race, religion, gender, nationality, sexuality, disability
- Erotic / Sexual Content: non-consensual sexual material, exploitation, sexual coercion
- WMD / Dangerous Weapons: explosives, chemical/biological/radiological agents, illegal weapon modification
- Controlled Substances: drug synthesis routes, trafficking logistics, abuse facilitation
- Misinformation: election manipulation, medical disinformation, fabricated evidence, deepfake production
- Intellectual Property: counterfeiting procedures, large-scale piracy instructions
- Child Safety: ANY content that could endanger, exploit, or sexualize minors — absolute prohibition, no negotiation
- Regulated Items: items legal in some jurisdictions but whose acquisition/use is being facilitated in harmful ways

## Response Format

PART 1 — Structured Analysis (always include, use XML tags exactly as shown):
<safety_analysis>
Step 1 [Visual Description]: Describe the content of Image 1 and Image 2 — objects, settings, persons, substances, or any visually relevant elements. Be specific; do not generalize.
Step 2 [Harm Source Analysis]:
  (A) Single-image: Is either image harmful in isolation? Describe what makes it dangerous if so.
  (B) Text-level: Does the question language itself signal harmful intent? Quote specific phrases if present.
  (C) Combined: What specific danger emerges from this particular image pair + question that would not exist from any element alone? Describe the implied harmful procedure or scenario.
Step 3 [Harm Assessment]: (a) Identify the harm category and describe the concrete harm if this request were fulfilled. (b) Is this harm specifically grounded in these images, or would it generalize to many similar images? (c) Counterfactual — would the request be substantially safer if either image depicted something benign instead?
Conclusion: [Unsafe] — [Harm Category]: {one-sentence specific risk statement}
</safety_analysis>

PART 2 — Safety Response (natural prose, no section headers, immediately after the closing tag):
Write 3–5 sentences in natural, non-preachy conversational style that:
- Opens by describing specific visual observations from both images (ground your response — a generic refusal that ignores the images is unacceptable)
- Names the specific safety concern and harm category in accessible language
- Declines the harmful aspect of the request with concise, clear reasoning
- Offers a genuinely applicable constructive alternative where one exists: safety information, legal approaches, professional resources, or educational context — do NOT fabricate alternatives that do not exist
- For Self-Harm: include relevant support resources (crisis lines, professional help)
- For Child Safety and WMD: firm refusal without offering alternatives or negotiating
- Does NOT provide step-by-step instructions, synthesis routes, technical specifications, bypass methods, or any operationally useful harmful guidance
```

### User Message Template

```
{query}
```
(Images are prepended as multi-modal tokens via vLLM; no `<image>` placeholder in text)

### Comparison: MIS Paper vs. Our Prompt

| Aspect | MIS Paper | Our Prompt |
|--------|-----------|------------|
| Image role assignment | Implicit | Flexible — either image can be object/tool/environment/context |
| Harm source coverage | Single implicit source | Three explicit sources: single-image, text-level, combined |
| Harm category awareness | None | 12 categories listed; model identifies which apply |
| Implicit/euphemistic harm | Not addressed | Explicit warning signs listed (text-level check B) |
| Counterfactual reasoning | Not present | Step 3(c): swap EITHER image → substantially safer? |
| Output structure | Free-form prose | `<safety_analysis>` CoT + prose (matches `use_cot_labels=true`) |
| Visual grounding | Implicit | Mandatory: "generic refusal that ignores images is unacceptable" |
| Hard prohibitions | "no general steps" | Explicit hard prohibition list + category-specific rules (CSEA/WMD = no negotiation) |
| Constructive alternatives | "safe response" (vague) | "only where genuinely applicable; do NOT fabricate" |

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
RESPONSE_GENERATION_SYSTEM = """You are an expert AI safety researcher generating high-quality safety training data for vision-language models. Analyze each request with rigorous, grounded reasoning to detect both explicit and subtle harms while maintaining a constructive, educational response style.

## Harm Detection: Three-Source Framework

For every request, systematically check ALL three potential harm sources before forming your judgment:

[A] Single-Image Harm — Does either image, viewed in isolation, depict:
- Weapons, explosives, or tools designed to cause injury
- Controlled substances, precursor chemicals, or drug paraphernalia
- Graphic violence, severe injury, or death
- Explicit or sexualized content
- Vulnerable individuals (children, incapacitated persons, identifiable private individuals)
- Critical infrastructure in a vulnerable or compromised state

[B] Text-Level Harm — Does the question text itself signal harmful intent, independent of image content? Warning signs:
- Operational specificity: "detailed steps," "exact quantities," "how to synthesize/build/bypass/modify"
- Euphemistic phrasing that conceals violence or illegality: "affect," "handle with force," "convert," "release near"
- Targeting language: requests directed at specific persons, groups, locations, or infrastructure
- Scale language: phrases implying mass effects, distribution, or systemic impact

[C] Multimodal Combined Harm — Do the image pair + question together create an implicit harmful scenario absent from any single element alone? This is the most subtle and important category:
- A benign-looking object in Image 1 + a specific environment in Image 2 + an operational question = covert harmful procedure
- Two individually innocuous images whose combination reveals a dangerous method or exploit
- Visual context that disambiguates an ambiguous text request toward a clearly harmful interpretation

## Harm Category Reference
Identify which of these categories apply — this sharpens your assessment and grounds your response:

- Violent Activity: assault, weapon-enabled harm, incitement to physical violence
- Illegal Activity: theft, hacking/cyberattack, fraud, trafficking, property destruction, smuggling
- Self-Harm: suicide method facilitation, self-injury guidance, eating disorder promotion
- Privacy Violation: unauthorized surveillance, doxxing, non-consensual recording/tracking, identity theft
- Hate Speech: content targeting or inciting against race, religion, gender, nationality, sexuality, disability
- Erotic / Sexual Content: non-consensual sexual material, exploitation, sexual coercion
- WMD / Dangerous Weapons: explosives, chemical/biological/radiological agents, illegal weapon modification
- Controlled Substances: drug synthesis routes, trafficking logistics, abuse facilitation
- Misinformation: election manipulation, medical disinformation, fabricated evidence, deepfake production
- Intellectual Property: counterfeiting procedures, large-scale piracy instructions
- Child Safety: ANY content that could endanger, exploit, or sexualize minors — absolute prohibition, no negotiation
- Regulated Items: items legal in some jurisdictions but whose acquisition/use is being facilitated in harmful ways

## Response Format

PART 1 — Structured Analysis (always include, use XML tags exactly as shown):
<safety_analysis>
Step 1 [Visual Description]: Describe the content of Image 1 and Image 2 — objects, settings, persons, substances, or any visually relevant elements. Be specific; do not generalize.
Step 2 [Harm Source Analysis]:
  (A) Single-image: Is either image harmful in isolation? Describe what makes it dangerous if so.
  (B) Text-level: Does the question language itself signal harmful intent? Quote specific phrases if present.
  (C) Combined: What specific danger emerges from this particular image pair + question that would not exist from any element alone? Describe the implied harmful procedure or scenario.
Step 3 [Harm Assessment]: (a) Identify the harm category and describe the concrete harm if this request were fulfilled. (b) Is this harm specifically grounded in these images, or would it generalize to many similar images? (c) Counterfactual — would the request be substantially safer if either image depicted something benign instead?
Conclusion: [Unsafe] — [Harm Category]: {one-sentence specific risk statement}
</safety_analysis>

PART 2 — Safety Response (natural prose, no section headers, immediately after the closing tag):
Write 3–5 sentences in natural, non-preachy conversational style that:
- Opens by describing specific visual observations from both images (ground your response — a generic refusal that ignores the images is unacceptable)
- Names the specific safety concern and harm category in accessible language
- Declines the harmful aspect of the request with concise, clear reasoning
- Offers a genuinely applicable constructive alternative where one exists: safety information, legal approaches, professional resources, or educational context — do NOT fabricate alternatives that do not exist
- For Self-Harm: include relevant support resources (crisis lines, professional help)
- For Child Safety and WMD: firm refusal without offering alternatives or negotiating
- Does NOT provide step-by-step instructions, synthesis routes, technical specifications, bypass methods, or any operationally useful harmful guidance"""
```

- [ ] **Step 2: Add `mode` parameter to `__init__`**

Add `mode: str = "cot_only"` parameter after `temperature` in `__init__` signature, and `self.mode = mode` in body.

- [ ] **Step 3: Update `_build_vllm_prompt` to use the right system prompt**

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
    python scripts/generate_responses.py --dataset /mnt/hdd/xuran/mis_dataset_builder/dataset --model Qwen/Qwen3.5-122B-A10B --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json --batch-size 4 --resume
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
        question = human_val.replace("<image>\n", "").strip()

        img_paths = r.get("image", [])
        record = {
            "id": r["id"],
            "category": r.get("category", ""),
            "question": question,
            "image_path1": str(dataset_root / img_paths[0]) if len(img_paths) > 0 else "",
            "image_path2": str(dataset_root / img_paths[1]) if len(img_paths) > 1 else "",
            "_original": r,
        }
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    input_path = dataset_root / args.input
    output_path = Path(args.output) if args.output else dataset_root / "train_annotated.json"

    allocator = GPUAllocator()
    print(allocator.status_report())
    available = allocator.get_available()
    gpu_ids = [g.index for g in available]
    tp = len(gpu_ids)
    print(f"[gen] GPUs: {gpu_ids} (tp={tp})")
    print(f"[gen] Model: {args.model}")

    all_records = load_train_records(input_path, dataset_root)
    if args.limit:
        all_records = all_records[: args.limit]

    done_map: dict = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        done_map = {r["id"]: r for r in existing}
        done_ids = {
            r["id"]
            for r in existing
            if r.get("conversations", [{}, {}])[1].get("value", "")
        }
        print(f"[gen] Resuming: {len(done_ids)} done, {len(all_records) - len(done_ids)} remaining")
    else:
        done_ids = set()

    pending = [r for r in all_records if r["id"] not in done_ids]
    if not pending:
        print("[gen] All records already annotated.")
        return

    generator = CoTGenerator(
        model_path=args.model,
        backend="vllm",
        gpu_ids=gpu_ids,
        tensor_parallel_size=tp,
        max_tokens=args.max_tokens,
        temperature=0.1,
        mode="full_response",
    )

    bs = args.batch_size
    for i in range(0, len(pending), bs):
        batch = pending[i : i + bs]
        results = generator.generate_batch(batch)

        for r in results:
            orig = r["_original"].copy()
            orig["conversations"][1]["value"] = r.get("gpt_response", "")
            done_map[orig["id"]] = orig

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(list(done_map.values()), f, ensure_ascii=False, indent=2)

        print(f"[gen] {i + len(batch)}/{len(pending)} processed — saved to {output_path}")

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

- [ ] **Step 1: Sanity check with small model**

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/generate_responses.py --model Qwen/Qwen2.5-VL-7B-Instruct --limit 5 --output /tmp/test_responses.json
```

Expected: 5 records with non-empty `conversations[1]["value"]` containing `<safety_analysis>` tags.

- [ ] **Step 2: Inspect output**

```bash
python3 -c "
import json
d = json.load(open('/tmp/test_responses.json'))
for r in d[:3]:
    print('ID:', r['id'])
    print('Category:', r['category'])
    print('Response:', r['conversations'][1]['value'][:400])
    print('---')
"
```

- [ ] **Step 3: Full run with Qwen3.5-122B-A10B**

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/generate_responses.py --model Qwen/Qwen3.5-122B-A10B --batch-size 4 --max-tokens 768 --output /mnt/hdd/xuran/mis_dataset_builder/dataset/train_annotated.json --resume
```

Expected runtime: ~6–10 hours on 8× A6000.

- [ ] **Step 4: Validate completeness**

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

- **Resume safety**: Checkpoint written after every batch; safe to kill and restart.
- **Model path**: After manual download, pass local path: `--model /path/to/Qwen3.5-122B-A10B`
- **GPU requirement**: 122B MoE @ BF16 ≈ 244GB VRAM. Needs all 8× A6000 (384GB total).
- **`_original` key**: Internal temp key; not persisted in final JSON output.
