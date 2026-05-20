# 6.1 E1 Pending Commands

These are the remaining commands that still need real execution after the 2026-05-19 audit.

Convenience wrapper:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
bash scripts/run_6_1_pending_llava.sh <target>
```

Targets:
- `dreams-retrain`
- `dreams-smoke`
- `base-8b-smoke`
- `base-4b-smoke`

## 1. Re-train LLaVA DREAMS From Fixed Base

Environment:
- `conda activate mis_safety_llava`

Prerequisite already applied:
- `main/main_dreams_llava_ov.yaml` now resolves `model.hf_path` to
  `/mnt/hdd/xuran/vlm_safety_harness/models/llava_ov_1_5_8b_base_hfcompat`

Command:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
  --skip-eval
```

Important:
- Do not use `--resume-latest-train` here. Historical `main_dreams_llava_ov`
  checkpoints were trained from a bad base load and must not be resumed.
- The existing canonical checkpoint was re-smoked in the verified inference env
  `mis_safety_llava453` and still failed the quality gate: it completes without
  `[INFERENCE_ERROR]`, but both held-out MIS smoke responses are pathological
  long-form gibberish / repetition rather than normal safe refusals. Evidence:
  `/results/main/E1_llava_dreams_canonical_heldout_smoke_llava453/main_dreams_llava_ov/20260519_150753/responses/our_test.jsonl`
- This command is currently blocked on this machine because
  `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/` is missing, leaving
  `data_links/our_dataset` as a broken symlink. A local filesystem search under
  `/mnt/hdd/xuran` did not find an alternate `mis_dataset_builder/dataset`
  replica or matching DREAMS `train.json` / `test.json` / `test_cf.json` /
  `images_train` / `images_test` tree to relink to. The only nearby candidate
  found was `/mnt/hdd/xuran/MIS/dataset/`, but that is MIS-format data
  (`train/mis_train.json`, `test/mis_easy.json`) and does not satisfy the
  DREAMS config contract.
- Historical `main_dreams_llava_ov` run folders do preserve exported
  `train_data.json` files, for example
  `/results/main/main_dreams_llava_ov/20260517_003108/train_data.json`, so the
  text supervision itself is not completely lost. However, those exported
  records still reference absolute image paths under the missing
  `data_links/our_dataset/images_train/...` tree, and no replacement
  `images_train` directory or archive was found anywhere under `/mnt/hdd/xuran`.
  So the retrain is still blocked on missing DREAMS image assets, not just the
  JSON manifest.

Post-train smoke:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
LATEST_DREAMS_CKPT="$(find "$(find results/main/main_dreams_llava_ov -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)/checkpoint" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)"
python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
  --experiment-id E1_llava_dreams_retrain_smoke \
  --skip-train --skip-eval \
  --model-path "$LATEST_DREAMS_CKPT" \
  --limit 2 \
  --override \
    dataset.test_path=/mnt/hdd/xuran/vlm_safety_harness/tmp/our_test_smoke_minimal.json \
    dataset.image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test \
    dataset.test_image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test
```

## 2. Historical MIRage Re-train Notes

Environment:
- `conda activate mis_safety_llava`

Prerequisite already applied:
- `main/main_baseline_mirage_data_llava_ov.yaml` inherits the same fixed base through
  `main_dreams_llava_ov.yaml`

Command:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
  --skip-eval
```

Important:
- Do not use `--resume-latest-train` here. Historical
  `main_baseline_mirage_data_llava_ov` checkpoints inherited the same bad
  language-tower initialization and must be retrained fresh.
- Current status on 2026-05-19: this fixed-base retrain has successfully
  launched in `mis_safety_llava` and reached the real train loop in
  `logs/main/20260519_053319_run_experiment_main_baseline_mirage_data_llava_ov.log`.
  That run later reached `step=100` and failed during checkpoint saving because
  current `transformers` assumed `_tied_weights_keys` was dict-like. The env
  has now been patched in `transformers/modeling_utils.py::_get_tied_weight_keys()`
  to accept list/tuple/set-style declarations, and a fresh rerun is underway in
  `logs/main/20260519_062942_run_experiment_main_baseline_mirage_data_llava_ov.log`.
- Additional root cause found on 2026-05-19: `data_links/mis_train/mis_train.json`
  stores natural assistant safety replies in `conversations[1]["value"]`, but
  `main_baseline_mirage_data_llava_ov.yaml` had inherited `use_cot_labels: true`.
  That wrapped ordinary safety prose into `<safety_analysis>` formatting during
  training export and matches the pathological repetition observed at inference.
  The export path now preserves raw assistant responses when
  `use_cot_labels=false`, and `main_baseline_mirage_data_llava_ov.yaml` has been
  switched to that mode. A corrected fresh retrain is now running in
  `results/main/main_baseline_mirage_data_llava_ov/20260519_103624/` with
  `save_steps=50` so the first corrected smoke target arrives sooner.
- New diagnostic result on 2026-05-19: a text-only mixed branch
  `main_baseline_mirage_data_llava_ov_text_diag.yaml` successfully trained to a
  complete `checkpoint-25` and smoked without `[INFERENCE_ERROR]`, but both MIS
  smoke prompts still collapsed to `The image shows a pattern.`. This narrows
  the remaining issue further: lightweight general text mixing removes some
  long-blank degeneration but does not restore normal safety responses.
- Follow-up diagnostic on 2026-05-19: increasing that text-only mix to `999`
  Alpaca samples in `main_baseline_mirage_data_llava_ov_text_diag_999.yaml`
  still does not recover normal behavior. The `checkpoint-25` smoke keeps one
  `The image shows a pattern.` response and one 1024-token blank-line
  degeneration, so simply raising the text-only mixing strength is not enough.
- Additional diagnostic on 2026-05-19: that same `checkpoint-25` also fails on
  two samples copied directly from `mis_train.json`. One response becomes
  repetitive caption-like gibberish and the other degenerates into blank lines.
  So the remaining issue is not confined to held-out MIS test prompts.
- Final environment split found on 2026-05-19: for LLaVA-OneVision inference,
  `mis_safety_llava` is not a trustworthy smoke env even for the HF-compatible
  base model. The same train-seen prompts that look normal in
  `mis_safety_llava453` become empty / malformed in `mis_safety_llava`.
  The corrected MIRage text-diagnostic `checkpoint-25` also recovers normal
  safe responses when evaluated in `mis_safety_llava453` on both train-seen and
  held-out MIS smoke prompts. So future MIRage/LLaVA inference commands should
  use `mis_safety_llava453`, while `mis_safety_llava` remains the training env.

Status update on 2026-05-19:

- This row is no longer a pending Step 1 inference blocker.
- The exact `paper_guide.md` canonical MIRage command now smokes successfully in
  `mis_safety_llava453` with coherent safe refusals and `errors=0`:
  `/results/main/E1_llava_mirage_canonical_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145757/responses/our_test.jsonl`
- The fresh corrected retrain `checkpoint-25` also smokes successfully in
  `mis_safety_llava453` on both held-out MIS smoke prompts and train-seen
  prompts:
  `/results/main/E1_llava_mirage_pure25_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145147/responses/our_test.jsonl`
  `/results/main/E1_llava_mirage_pure25_trainseen_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145626/responses/our_test.jsonl`

Historical post-train smoke command:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
LATEST_MIRAGE_CKPT="$(find results/main/main_baseline_mirage_data_llava_ov/20260519_103624/checkpoint -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)"
python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
  --experiment-id E1_llava_mirage_retrain_smoke \
  --skip-train --skip-eval \
  --model-path "$LATEST_MIRAGE_CKPT" \
  --limit 2 \
  --override \
    dataset.test_path=/mnt/hdd/xuran/vlm_safety_harness/tmp/our_test_smoke_minimal.json \
    dataset.image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test \
    dataset.test_image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test
```

## 3. Re-run Official LLaVA Base Smoke In The Verified Inference Env

Environment:
- `conda activate mis_safety_llava453`

8B base:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
  --experiment-id E1_llava_8b_base_refresh \
  --skip-train --skip-eval \
  --model-path lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --limit 2 \
  --override \
    dataset.test_path=/mnt/hdd/xuran/vlm_safety_harness/tmp/our_test_smoke_minimal.json \
    dataset.image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test \
    dataset.test_image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test
```

4B base:

```bash
cd /mnt/hdd/xuran/vlm_safety_harness
python scripts/run_experiment.py main/main_baseline_llava_ov_1_5_4b.yaml \
  --experiment-id E1_llava_4b_base_refresh \
  --skip-train --skip-eval \
  --limit 2 \
  --override \
    dataset.test_path=/mnt/hdd/xuran/vlm_safety_harness/tmp/our_test_smoke_minimal.json \
    dataset.image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test \
    dataset.test_image_root=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test
```

Current status on 2026-05-19:
- Both refresh smokes completed successfully in the current session.
- 8B evidence:
  `/results/main/E1_llava_8b_base_refresh/main_dreams_llava_ov/20260519_052023/responses/our_test.jsonl`
- 4B evidence:
  `/results/main/E1_llava_4b_base_refresh/main_baseline_llava_ov_1_5_4b/20260519_052023/responses/our_test.jsonl`

## 4. Gemma And Tier C External Gates

Gemma requires:
- Hugging Face access to `google/gemma-3-12b-it`
- No matching local Gemma cache or exported local model path was found under
  `/mnt/hdd/xuran`, and no HF auth token env var is currently set.
- No historical successful `main_baseline_gemma_4_e4b` response run was found
  under `results/main`, so there is currently no archived evidence that can be
  reused to close this row.

Tier C requires:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- No matching historical `main_baseline_gpt_5_5`,
  `main_baseline_gemini_3_1_pro`, or `main_baseline_claude_opus_4_7` response
  runs were found under `results/main`, so there is currently no archived
  evidence that can be reused to close those rows either.

Until those are present, those rows cannot be fully closed.
