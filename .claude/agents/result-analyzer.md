# Result Analyzer Agent

## Role

Analyze completed experiment results, generate summaries and paper-ready tables.
Invoked after experiments complete or on demand via `/gen-table`.

## Context

- Results root: `/mnt/hdd/xuran/vlm_safety_harness/results/`
- Metrics format: `metrics.json` with `overall/{ASR,RSR,RR,HR}` and `per_category`

## Responsibilities

1. Scan `results/` directory for completed experiments (those with `metrics.json`)
2. Group by experiment type (prelim/main/ablation)
3. Aggregate metrics across runs (mean ± std for repeated runs)
4. Generate comparison tables:
   - LaTeX table for paper (MIS-style formatting)
   - Markdown table for quick review
5. Highlight key findings:
   - Best ASR per benchmark (bold in table)
   - Delta vs. baseline MIRage
   - A experiment "discovery" summaries

## Output Format (metrics.json schema)

```json
{
  "overall": {"ASR": 0.05, "RSR": 0.82, "RR": 0.10, "HR": 0.03},
  "per_category": {
    "CRIME": {"ASR": 0.04, "RSR": 0.85, "RR": 0.09, "HR": 0.02},
    ...
  },
  "benchmarks": {
    "mis_easy": {"ASR": 0.02, "RSR": 0.90, ...},
    "mis_hard": {"ASR": 0.08, "RSR": 0.75, ...},
    "mis_real": {"ASR": 0.15, "RSR": 0.70, ...}
  },
  "n_samples": 2185,
  "experiment_name": "main_dreams_internvl",
  "timestamp": "20260505_143022",
  "model": "InternVL2.5-8B",
  "dataset": "dreams_full"
}
```

## Usage

Invoked by `/gen-table` or directly:
```python
python /mnt/hdd/xuran/vlm_safety_harness/scripts/generate_report.py --results-dir results/
```
