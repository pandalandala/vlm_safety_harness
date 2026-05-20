# 6.1 E1 Inference Audit

Last updated: 2026-05-19

This note records the actual execution status for the inference commands listed in
`docs/paper_guide.md` section `6.1 E1 — DREAMS 分布内安全`.

## Success Criteria

Each command is considered covered only when all of the following hold:

- the required environment is known and reproducible;
- the smoke run finishes without `[INFERENCE_ERROR]` / `EngineDeadError`;
- the response text is semantically normal, not gibberish or pathological repetition.

## Command Coverage Audit

The Step 1 inference commands in `docs/paper_guide.md` section `6.1 E1` currently
expand to the following rows:

- Tier A SFT inference: 6 commands
- Tier A base inference: 3 commands
- Tier B inference: 8 commands
- Tier C closed-source inference: 3 commands

All of those inference rows are now mapped in this audit note. At the moment,
there is no known uncovered Step 1 inference command outside the blocked items
listed below; the remaining gaps are quality failures or external gates, not
missing bookkeeping.

## Covered Commands

These items already have smoke evidence with `errors=0` and normal text.

| Group | Model / Command family | Environment | Evidence |
|---|---|---|---|
| Tier A SFT | `InternVL3.5 + DREAMS` | `mis_safety` | `/results/main/E1_smoke2/main_dreams_internvl3_5/20260519_010819/responses/our_test.jsonl` |
| Tier A SFT | `Qwen3.5 + DREAMS` | `mis_safety` | `/results/main/E1_smoke2/main_dreams_qwen3_5/20260519_011252/responses/our_test.jsonl` |
| Tier A SFT baseline | `InternVL3.5 + MIRage-data` | `mis_safety` | `/results/main/E1_smoke2/main_baseline_mirage_data_internvl3_5/20260519_011252/responses/our_test.jsonl` |
| Tier A SFT baseline | `Qwen3.5 + MIRage-data` | `mis_safety` | `/results/main/E1_smoke4/main_baseline_mirage_data_qwen3_5/20260519_012352/responses/our_test.jsonl` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` canonical command | `mis_safety_llava453` | `/results/main/E1_llava_mirage_canonical_heldout_smoke_llava453_full/main_baseline_mirage_data_llava_ov/20260519_150411/responses/our_test.jsonl` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` fresh retrain `checkpoint-25` | `mis_safety_llava453` | `/results/main/E1_llava_mirage_pure25_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145147/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_pure25_trainseen_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145626/responses/our_test.jsonl` |
| Tier A base | `InternVL3.5-8B base` | `mis_safety` | `/results/main/E1_smoke26/main_dreams_internvl3_5/20260519_024047/responses/our_test.jsonl` |
| Tier A base | `Qwen3.5-9B base` | `mis_safety` | `/results/main/E1_smoke25/main_dreams_qwen3_5/20260519_023620/responses/our_test.jsonl` |
| Tier A base | `LLaVA-OV-1.5-8B base` | `mis_safety_llava453` | `/results/main/E1_llava_8b_base_refresh/main_dreams_llava_ov/20260519_052023/responses/our_test.jsonl` |
| Tier B | `Kimi-VL-A3B` | `mis_safety` | `/results/main/E1_smoke5/main_baseline_kimi_vl_a3b/20260519_013053/responses/our_test.jsonl` |
| Tier B | `MiniCPM-o-4.5` | `mis_safety` | `/results/main/E1_smoke4/main_baseline_minicpm_o_4_5/20260519_012351/responses/our_test.jsonl` |
| Tier B | `MiniCPM-V-4.6` | `mis_safety` | `/results/main/E1_smoke17/main_baseline_minicpm_v_4_6/20260519_021854/responses/our_test.jsonl` |
| Tier B | `GLM-4.6V-Flash` | `mis_safety` | `/results/main/E1_smoke4/main_baseline_glm_4_6v_flash/20260519_012449/responses/our_test.jsonl` |
| Tier B | `Qwen3.5-4B` | `mis_safety` | `/results/main/E1_smoke2/main_baseline_qwen3_5_4b/20260519_011253/responses/our_test.jsonl` |
| Tier B | `InternVL3.5-4B` | `mis_safety` | `/results/main/E1_smoke8/main_baseline_internvl3_5_4b/20260519_013818/responses/our_test.jsonl` |
| Tier B | `LLaVA-OV-1.5-4B` | `mis_safety_llava453` | `/results/main/E1_llava_4b_base_refresh/main_baseline_llava_ov_1_5_4b/20260519_052023/responses/our_test.jsonl` |

## Blocked Or Invalid

These items are not complete, even if a process ran.

| Group | Model / Command family | Status | Evidence |
|---|---|---|---|
| Tier A SFT | `LLaVA-OV + DREAMS` | Existing checkpoint still invalid even in the verified LLaVA inference env. In `mis_safety_llava453` it loads and completes without `[INFERENCE_ERROR]`, but both held-out MIS smoke responses are still pathological long-form gibberish / repetition, so this row remains a real model-quality failure rather than an env-stack false positive. | `/results/main/E1_smoke18/main_dreams_llava_ov/20260519_022039/responses/our_test.jsonl`, `/results/main/E1_llava_dreams_canonical_heldout_smoke_llava453/main_dreams_llava_ov/20260519_150753/responses/our_test.jsonl`, `/logs/main/20260519_150753_run_experiment_main_dreams_llava_ov.log` |
| Tier A SFT | `LLaVA-OV + DREAMS` training root cause | Training log shows full `model.language_model.*` tower was `MISSING` and newly initialized when loading official base. | `/logs/main/20260517_003108_run_experiment_main_dreams_llava_ov.log:741` |
| Tier A SFT | `LLaVA-OV + DREAMS` fresh retrain | External data blocker on this machine: `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/` is absent, so `data_links/our_dataset` is a broken symlink and fresh retraining cannot read `train.json` / `images_train/`. A renewed filesystem search on 2026-05-19 found only `/mnt/hdd/xuran/MIS/dataset/`, which contains `train/mis_train.json` and `test/mis_easy.json` for MIS, not the DREAMS-style `train.json`, `test.json`, `test_cf.json`, `images_train/`, or `images_test/` assets expected by the DREAMS configs. Historical `main_dreams_llava_ov` run directories do preserve exported `train_data.json`, but those records still point at absolute paths under the missing `data_links/our_dataset/images_train/...` tree, and no substitute `images_train` assets or archive were found under `/mnt/hdd/xuran`. So the blocker is specifically missing DREAMS image assets, not just the raw JSON manifest. | `/logs/main/20260519_052206_run_experiment_main_dreams_llava_ov.log`, `/results/main/main_dreams_llava_ov/20260517_003108/train_data.json` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` | Raw checkpoint had legacy key namespace mismatch, and historical training logs show the same broken base-load pattern as DREAMS: the official Hub base left the full `model.language_model.*` tower newly initialized. Canonical inference path now points to a converted HF-compatible export, but that canonical path is no longer the right post-retrain smoke target. Fresh checkpoint smoke must point at the latest retrain `checkpoint-*`, not `/models/mirage_data_llava_ov`. | `/models/mirage_data_llava_ov -> /models/mirage_data_llava_ov_hfcompat`, plus `/logs/main/20260514_162137_run_experiment_main_baseline_mirage_data_llava_ov.log:646` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` fresh retrain | The first fixed-base retrain run reached step `100` and then failed during checkpoint save, not during optimization. Root cause: current `transformers` expected `_tied_weights_keys` to be dict-like, but `LLaVAOneVision1_5` exposes it as a list. `transformers/modeling_utils.py::_get_tied_weight_keys()` has now been patched to accept list/tuple/set-style tied-weight declarations. After that fix, the rerun in `mis_safety_llava` successfully saved `checkpoint-100`; the save-path bug is resolved. | `/logs/main/20260519_053319_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/logs/main/20260519_062942_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/results/main/main_baseline_mirage_data_llava_ov/20260519_062942/checkpoint/checkpoint-100` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` early fresh smokes in old LLaVA env | Historical smokes run in `mis_safety_llava` were misleading: both the canonical path and early retrain checkpoints could show repetition, blank lines, or malformed short text there. Later A/B checks against `mis_safety_llava453` showed that this inference env, not just the weights, was a primary source of false-bad LLaVA outputs. Keep these runs only as historical evidence, not as the current acceptance verdict. | `/results/main/E1_llava_mirage_retrain_smoke/main_baseline_mirage_data_llava_ov/20260519_073514/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_retrain_smoke/main_baseline_mirage_data_llava_ov/20260519_102158/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_canonical_smoke/main_baseline_mirage_data_llava_ov/20260519_103222/responses/our_test.jsonl`, `/logs/main/20260519_073514_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/logs/main/20260519_102158_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/logs/main/20260519_103222_run_experiment_main_baseline_mirage_data_llava_ov.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` root cause and corrective retrain | The MIRage train file `data_links/mis_train/mis_train.json` stores natural assistant safety responses in `conversations[1]["value"]`, not standalone CoT labels. But `main_baseline_mirage_data_llava_ov.yaml` inherited `use_cot_labels: true`, so training wrapped those natural replies inside `<safety_analysis>` formatting and likely taught the model the wrong output structure. The export path has now been fixed so `use_cot_labels: false` preserves the original assistant response, `main_baseline_mirage_data_llava_ov.yaml` has been switched to that mode, and a fresh corrected retrain has started with `save_steps=50` in run `20260519_103624`. | `/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_train/mis_train.json`, `/mnt/hdd/xuran/vlm_safety_harness/harness/data/dataset.py`, `/mnt/hdd/xuran/vlm_safety_harness/harness/data/converters.py`, `/mnt/hdd/xuran/vlm_safety_harness/configs/experiments/main/main_baseline_mirage_data_llava_ov.yaml`, `/results/main/main_baseline_mirage_data_llava_ov/20260519_103624/train_data.json`, `/results/main/main_baseline_mirage_data_llava_ov/20260519_103624/checkpoint/llamafactory_config.yaml`, `/logs/main/20260519_103624_run_experiment_main_baseline_mirage_data_llava_ov.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` corrected `checkpoint-50` smoke | The corrected retrain reached `checkpoint-50` and the checkpoint can now be made inferable by restoring the missing processor metadata (`preprocessor_config.json` and `processor_config.json`). After that repair, smoke no longer emits `[INFERENCE_ERROR]`, but the text is still visibly repetitive (`It seems to`, repeated `The images`, repeated `The`), so response quality is still not recovered at step 50. | `/results/main/main_baseline_mirage_data_llava_ov/20260519_103624/checkpoint/checkpoint-50`, `/results/main/E1_llava_mirage_retrain_fix50_smoke_retry/main_baseline_mirage_data_llava_ov/20260519_112124/responses/our_test.jsonl`, `/logs/main/20260519_112124_run_experiment_main_baseline_mirage_data_llava_ov.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` corrected retrain crash after `checkpoint-50` | The corrected run `20260519_103624` did not finish: after saving `checkpoint-50`, rank 6 failed during DeepSpeed ZeRO checkpoint write with `RuntimeError: basic_ios::clear: iostream error`, and the rest of the ranks were SIGTERM'd. This is a checkpoint-save path failure, not an inference engine failure. | `/logs/main/20260519_103624_run_experiment_main_baseline_mirage_data_llava_ov.log:891`, `/logs/main/20260519_103624_run_experiment_main_baseline_mirage_data_llava_ov.log:905` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` text-only general-data diagnostic | A lightweight diagnostic branch mixed `500` text-only Alpaca-style general instruction samples into the corrected MIRage training stream. This branch trains cleanly, saves a complete `checkpoint-25` with processor metadata, and smokes without `[INFERENCE_ERROR]`, but both MIS smoke prompts still collapse to `The image shows a pattern.`. So adding small text-only general data removes some long-blank degeneration, but does not restore normal safe responses. | `/configs/experiments/main/main_baseline_mirage_data_llava_ov_text_diag.yaml`, `/results/main/main_baseline_mirage_data_llava_ov_text_diag/20260519_135231/checkpoint/checkpoint-25`, `/results/main/E1_llava_mirage_text_diag_smoke/main_baseline_mirage_data_llava_ov_text_diag/20260519_140911/responses/our_test.jsonl`, `/logs/main/20260519_135231_run_experiment_main_baseline_mirage_data_llava_ov_text_diag.log`, `/logs/main/20260519_140911_run_experiment_main_baseline_mirage_data_llava_ov_text_diag.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` stronger text-only general-data diagnostic | Increasing text-only Alpaca mixing strength from `500` to `999` samples does not fix the MIRage collapse. The stronger-mix branch also trains cleanly and saves a complete `checkpoint-25`, but smoke still yields one `The image shows a pattern.` response and one 1024-token blank-line degeneration. This rules out the simplest “just add a bit more general text” recovery path. | `/configs/experiments/main/main_baseline_mirage_data_llava_ov_text_diag_999.yaml`, `/results/main/main_baseline_mirage_data_llava_ov_text_diag_999/20260519_141347/checkpoint/checkpoint-25`, `/results/main/E1_llava_mirage_text_diag_999_smoke/main_baseline_mirage_data_llava_ov_text_diag_999/20260519_143029/responses/our_test.jsonl`, `/logs/main/20260519_141347_run_experiment_main_baseline_mirage_data_llava_ov_text_diag_999.log`, `/logs/main/20260519_143029_run_experiment_main_baseline_mirage_data_llava_ov_text_diag_999.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` train-seen smoke diagnostic | The stronger text-only diagnostic checkpoint also fails on training-distribution samples, not just held-out MIS smoke prompts. When evaluated on two records copied directly from `mis_train.json`, the model still produces repetitive caption-like gibberish on one sample and blank-line degeneration on the other, despite no `[INFERENCE_ERROR]`. This indicates the remaining failure is not merely an out-of-distribution decoding issue on held-out prompts. | `/tmp/mis_train_smoke_minimal.json`, `/results/main/E1_llava_mirage_text_diag_999_trainseen_smoke/main_baseline_mirage_data_llava_ov_text_diag_999/20260519_143836/responses/our_test.jsonl`, `/logs/main/20260519_143836_run_experiment_main_baseline_mirage_data_llava_ov_text_diag_999.log` |
| Tier A SFT baseline | `LLaVA-OV + MIRage-data` inference env root cause | A decisive environment split was confirmed. In `mis_safety_llava`, even the HF-compatible base model produces empty / malformed outputs on the same two train-seen samples, and MIRage checkpoints also degenerate there. In `mis_safety_llava453`, both the canonical `paper_guide.md` command path and the fresh corrected `checkpoint-25` produce coherent safe refusals on held-out MIS smoke prompts, and the fresh checkpoint also behaves normally on train-seen prompts. This shows the remaining MIRage-LLaVA acceptance verdict depends on using `mis_safety_llava453` for inference, while `mis_safety_llava` remains only the training env. | `/results/main/E1_llava_8b_base_trainseen_smoke_llavaenv/main_dreams_llava_ov/20260519_144523/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_text_diag_999_trainseen_smoke_llava453/main_baseline_mirage_data_llava_ov_text_diag_999/20260519_144626/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_text_diag_999_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov_text_diag_999/20260519_144725/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_canonical_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145757/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_pure25_heldout_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145147/responses/our_test.jsonl`, `/results/main/E1_llava_mirage_pure25_trainseen_smoke_llava453/main_baseline_mirage_data_llava_ov/20260519_145626/responses/our_test.jsonl`, `/logs/main/20260519_144523_run_experiment_main_dreams_llava_ov.log`, `/logs/main/20260519_144626_run_experiment_main_baseline_mirage_data_llava_ov_text_diag_999.log`, `/logs/main/20260519_144725_run_experiment_main_baseline_mirage_data_llava_ov_text_diag_999.log`, `/logs/main/20260519_145147_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/logs/main/20260519_145626_run_experiment_main_baseline_mirage_data_llava_ov.log`, `/logs/main/20260519_145757_run_experiment_main_baseline_mirage_data_llava_ov.log` |
| Tier B | `Gemma-4-E4B` | External blocker: gated repo 403 on `google/gemma-3-12b-it`. | `/logs/main/20260519_023358_run_experiment_main_baseline_gemma_4_e4b.log` |
| Tier C | `GPT-5.5 / Gemini-3.1-Pro / Claude-Opus-4.7` | External blocker: required API keys absent from environment. | shell env check on 2026-05-19 |

## LLaVA-Specific Fixes Applied

- Official 8B base converted to local HF-compatible training base:
  `/models/llava_ov_1_5_8b_base_hfcompat`
- Static compatibility check for that converted base:
  `expected=698, actual=698, missing=0, unexpected=0`
- Training configs now point LLaVA 8B SFT to the converted base:
  - `configs/base/model_llava_ov_1_5_8b.yaml`
  - `configs/experiments/main/main_dreams_llava_ov.yaml`
  - `configs/experiments/main/main_baseline_mirage_data_llava_ov.yaml` inherits the same base through `_extends`
- Official LLaVA 8B/4B base inference environment documented as:
  `mis_safety_llava453`
- `mis_safety_llava` training env was repaired to get past multiple
  Transformers/LlamaFactory compatibility failures:
  - `llamafactory/extras/packages.py` patched so package-availability probes stay tuple-compatible and treat missing dotted packages as unavailable;
  - `transformers/utils/import_utils.py` patched so optional accelerator imports (`torch_mlu`, `torch_musa`, `torch_npu`) fall back to `False`;
  - `transformers/modeling_flash_attention_utils.py` patched to disable flash-attn probing in this env;
  - `transformers/modeling_utils.py::_get_tied_weight_keys()` patched so checkpoint saving works when `_tied_weights_keys` is declared as a list by LLaVA remote code;
  - `nltk` installed because LlamaFactory SFT metric import required it.
- `scripts/run_6_1_pending_llava.sh` was corrected so LLaVA retrain smokes:
  - resolve the latest retrain `checkpoint-*` automatically instead of reusing stale canonical model dirs;
  - override both `dataset.image_root` and `dataset.test_image_root`, because MIRage configs use `test_image_root` during eval and otherwise fall back to the broken `data_links/our_dataset` path.

## Remaining Work Before 6.1 Can Be Marked Complete

- Restore the missing DREAMS training asset root `/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset/` (or rebuild `data_links/our_dataset`) before `main_dreams_llava_ov` can be retrained from the converted base. Current evidence suggests the data root is genuinely absent on this machine, not just mislinked.
- Obtain Gemma gated access and Tier C API keys if those rows must be fully covered.
