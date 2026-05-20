# EMNLP 2026 Paper Draft Plan — DREAMS Dataset Paper

## Source of Truth

**Project content only**: `/mnt/hdd/xuran/vlm_safety_harness/docs/`, `/mnt/hdd/xuran/vlm_safety_harness/.claude/docs/`, `/mnt/hdd/xuran/vlm_safety_harness/configs/`, `/mnt/hdd/xuran/vlm_safety_harness/harness/`, paper_guide.md, MIS_shortcomes_final.md, project memory.

**Explicitly ignored**: existing body of `acl_latex.tex`. Only the title string is preserved (per user instruction). Everything else — abstract, sections, contributions, method, tables, appendix — is re-derived from project documentation and harness implementation.

## Context

Project builds DREAMS — 17,022-sample multi-image VLM safety dataset addressing four structural defects of MIS/MIRage (ICLR 2026) identified in `MIS_shortcomes_final.md`:
- **P2** textual shortcut (rigid templates)
- **P3** relation monoculture (~90% tool→target)
- **P4** synth-real gap (MIS ~100% AI images)
- **P7** no counterfactual reasoning (high FPR on MSSBench safe pairs)

Harness (`/mnt/hdd/xuran/vlm_safety_harness/harness/`) implements:
- Data: 17,022 samples → 15,319 train / 1,703 test, 12 harm categories, ~50% real + ~50% synthetic (SD-3.5), 4 relation types, CoT teacher labels (Qwen3.5-122B-A10B)
- Training: LLaMA-Factory + DeepSpeed ZeRO-3, vision tower frozen, lr 1e-5, bf16, 3 epochs
- 3 Tier-A SFT architectures: InternVL3.5-8B, Qwen3.5-9B, LLaVA-OV-1.5-8B
- 8 Tier-B inference-only baselines (Kimi-VL-A3B, MiniCPM-o/V, Gemma-4-E4B, GLM-4.6V-Flash, +4B size variants)
- 3 Tier-C closed-source (GPT-5.5, Gemini-3.1-Pro, Claude-Opus-4.7)
- GPT-4o judge → ASR / RSR / RR / HR metrics + FPR for CF samples
- 7 cross-benchmark safety eval (AdvBench, SafeBench, FigStep, MM-Safety, JailbreakV, SIUO, MSSBench)
- 5 capability benchmarks (MMStar, MMMU, MuirBench, BLink, MMT)

Experiment plan (paper_guide):
- A1–A4 diagnostic on MIS data only (no DREAMS, no SFT)
- E1 in-distribution safety (DREAMS test)
- E2 image-source generalization
- E3 cross-benchmark generalization
- E4 capability preservation (V0/V1/V2/V3/V4 of InternVL3.5)
- Abl-1 data scale, Abl-2 synth-real mix, Abl-3 relation coverage

Current status: E1 inference running, eval pending. A-experiments staged. Use paper_guide "Expected Finding" numbers as placeholders.

## User Choices Recorded

- Method framing: **pure dataset paper**; SFT recipe described as standard, no RL.
- Empty results: fill with paper_guide expected numbers; caption tables as preliminary.
- Appendix scope: A1–A4 diagnostic protocol + per-category breakdown + MIS-style supporting sections (construction pipeline, SFT label prompts, examples).

## Output Target

Single file edit: `/mnt/hdd/xuran/EMNLP2026/latex/acl_latex.tex`.
- Title string preserved verbatim (user instruction).
- All other content (abstract, sections, tables, appendix, captions) freshly written.
- Bib file `example_paper.bib` reused as-is; new citation keys added or marked `\todo{cite}`.
- ACL style files (`acl.sty`, `acl_natbib.bst`) untouched.

## Paper Structure

**Main body (target 8 pages)**

| § | Title | Page |
|---|-------|------|
| Abs | Abstract | 0.25 |
| 1 | Introduction | 1.0 |
| 2 | Related Work | 0.75 |
| 3 | Diagnosing Multi-Image Safety SFT | 1.75 |
| 4 | DREAMS Dataset Construction | 1.75 |
| 5 | Experiments | 2.0 |
| 6 | Conclusion + Limitations | 0.5 |

**Appendix (unlimited)**

| § | Content |
|---|---------|
| A | Construction pipeline details (filter R1–R5, CLIP gating, scoring rubric) |
| B | A1–A4 diagnostic protocols (black-frame probe, relation annotation, CF mapping) |
| C | Training & evaluation details (hyperparams, GPT-4o judge prompt, benchmark list, Tier-B/C model cards) |
| D | Per-category breakdown tables (12 harm × 4 metrics × Tier-A models) |
| E | Qualitative examples + SFT teacher prompt |

## Section Content Beats (re-derived from project docs)

### Abstract
Multi-image VLM safety landscape → MIS/MIRage limitations → DREAMS dataset (17K, 12 categories, balanced synth/real, 4 relation types, CoT labels) → SFT across 3 architectures + 14 baselines → safety gains + cross-benchmark generalization + capability preserved.

### §1 Introduction
- Threat model: benign single images, harmful in combination
- Multi-image safety state-of-the-art: MIS/MIRage (ICLR 2026)
- Empirically diagnose four defects (P2/P3/P4/P7) → motivate dataset redesign
- DREAMS = 4 design principles, 17K samples, 12 categories
- Validation: 3 Tier-A architectures, 8 Tier-B baselines, 3 Tier-C upper bounds, 7 cross-benchmarks, 5 capability benchmarks
- Contributions list (4 items): diagnostic framework, DREAMS dataset, SFT recipe & open harness, generalization + capability evidence

### §2 Related Work
- 2.1 Single-image VLM jailbreaks (FigStep, MM-Safety, AdvBench, JailbreakV, SIUO)
- 2.2 Multi-image safety (MIS/MIRage, MSSBench)
- 2.3 Safety fine-tuning (VLGuard, Textual SFT, capability–safety trade-off)

### §3 Diagnosing Multi-Image Safety SFT
Frame as 4 probes using MIS data only (no DREAMS, no new SFT).
- 3.1 A1 Textual Shortcut — black-frame substitution; ΔASR = ASR(full) − ASR(text-only); finding: MIRage ΔASR ≈ 0 → text-only safety signal
- 3.2 A2 Relation Monoculture — annotate MIS-hard with 4 relation types (tool→target, before→after, identity-linking, context-shift); finding: MIRage ≈0 on tool→target, 45–50% on others
- 3.3 A3 Synth-Real Gap — MIS-easy / MIS-hard / MIS-real comparison; finding: MIRage gap 15–20%, base gap ≈ 5% (gap is SFT-induced)
- 3.4 A4 Counterfactual Blindness — MSSBench safe vs unsafe pairs; FPR + pair-consistency + visual sensitivity; finding: MIRage FPR ≈30%, consistency ≈55%
- Summary table → 4 design principles for §4

### §4 DREAMS Dataset
- 4.1 Design principles (1↔A1, 2↔A2, 3↔A3, 4↔A4)
- 4.2 Pipeline overview (image sourcing, pair selection, prompt generation, filtering, CoT labeling) — figure placeholder
- 4.3 Filtering: R1 refusal prompts, R2 short prompts <30 chars, R3 explicit benign labels, R4 score ≤3, R5 empty category → 21,303 → 17,022
- 4.4 SFT label generation: Qwen3.5-122B-A10B teacher; structured CoT (perceive → reason → respond); prompt in App E
- 4.5 Statistics: 12 categories (table), 4 relation types (table), image-source split, prompt length distribution, comparison vs MIS (table)

### §5 Experiments
- 5.1 Setup: models, training (LF + ZeRO-3, 3 epochs, lr 1e-5, vision tower frozen, bf16), eval (GPT-4o judge, ASR/RSR/RR/HR/FPR), benchmarks
- 5.2 E1 in-distribution: Tab 1 — 3 Tier-A × {base, MIRage-data SFT, DREAMS SFT} + 8 Tier-B + 3 Tier-C on DREAMS test
- 5.3 E2 image-source generalization: Tab 2 — re-slice by synth/real/mixed
- 5.4 E3 cross-benchmark: Tab 3 — canonical metric per of 7 benchmarks
- 5.5 E4 capability preservation: Tab 4 — InternVL3.5 V0/V1/V2/V3/V4 on 5 capability benchmarks
- 5.6 Ablations: Tab 5 — Abl-1 data scale (25/50/75/100%), Abl-2 synth-real mix, Abl-3 relation coverage
- All number cells: fill from paper_guide "Expected Finding" or `\todo{}` where no expectation exists. Caption notes "preliminary, final pending"

### §6 Conclusion + Limitations
- 1 paragraph conclusion
- Limitations: English-only, 12-category coverage limit, GPT-4o judge bias, no adversarial training, synthetic image artifacts, 8B-scale only

### Appendix beats
- App A: pipeline figure (placeholder), filter rules with examples, CLIP gating τ / K / L params, scoring rubric
- App B: black-frame construction, relation-type annotation guide + GPT-4o prompt, MSSBench → DREAMS CF mapping
- App C: hyperparameter tables per model, GPT-4o judge prompt, benchmark descriptions, Tier-B/C model cards with sizes and licenses
- App D: 12 × {ASR, RSR, RR, HR} × {InternVL3.5 base, +MIRage-data, +DREAMS} per-category breakdown table
- App E: 4–6 qualitative cases (safe pair, unsafe pair, CF, real, synth, refusal-vs-reasoning), SFT teacher prompt verbatim

## Implementation Notes

- All numerical entries: paper_guide "Expected Finding" column; otherwise `\todo{TBD}`
- Tables: `booktabs` + `threeparttable` for footnotes
- Figures: `\fbox{Figure: <description>}` placeholders
- Use `\cref{}`/`\Cref{}` (cleveref loaded by ACL style)
- New citations: add to `example_paper.bib` if obvious (MIS, MIRage, MSSBench, FigStep, MM-Safety, JailbreakV, SIUO, AdvBench, VLGuard, M4-Instruct, Stable Diffusion 3.5, InternVL3.5, Qwen3.5, LLaVA-OV, GPT-4o), else `\todo{cite}`
- Maintain ACL 8-page main body limit
- Caption note "preliminary; final results pending" on all data tables

## Files Modified

| Path | Change |
|------|--------|
| `/mnt/hdd/xuran/EMNLP2026/latex/acl_latex.tex` | full body + appendix rewrite, only title string preserved |
| `/mnt/hdd/xuran/EMNLP2026/latex/example_paper.bib` | append new bib entries as cited (only if found/obvious) |

## Verification

1. Compile:
   ```
   cd /mnt/hdd/xuran/EMNLP2026/latex
   pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex
   ```
2. Confirm: title unchanged, all sections present, tables compile, no overfull boxes beyond minor
3. Page count: ≤8 main, appendix unlimited
4. Spot-check: no leftover content/phrasing from prior tex body

## Post-Plan Action

After exiting plan mode, also copy this plan to `/mnt/hdd/xuran/vlm_safety_harness/docs/paper_draft_plan.md` per project-docs convention.
