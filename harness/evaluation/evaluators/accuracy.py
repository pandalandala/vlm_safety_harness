"""
Accuracy evaluator — multiple-choice exact match for capability benchmarks
(MMStar, MMMU, MuirBench, BLINK, MMT).

Each input record must have a `gold_answer` field (single letter A/B/C/D or
short string). The model response is parsed for the first answer choice
appearing; if it matches gold_answer, the record is correct.
"""
from __future__ import annotations

import re
from typing import Optional

from harness.evaluation.benchmark_evaluator import BenchmarkEvaluator
from harness.evaluation.metrics import MetricsDict


# Match a leading "A.", "(B)", "Answer: C", "Option D)" etc.
_CHOICE_PATTERN = re.compile(
    r"\b(?:answer\s*[:\-]?\s*)?\(?([A-D])\)?\b",
    re.IGNORECASE,
)


def _extract_choice(response: str) -> Optional[str]:
    if not response:
        return None
    m = _CHOICE_PATTERN.search(response)
    return m.group(1).upper() if m else None


class AccuracyEvaluator(BenchmarkEvaluator):
    metric_name = "Accuracy"
    metric_direction = "↑"

    def __init__(self, gold_field: str = "gold_answer"):
        self.gold_field = gold_field

    def evaluate(self, records: list[dict]) -> tuple[list[dict], MetricsDict]:
        n_correct = 0
        per_cat: dict[str, list[bool]] = {}
        annotated: list[dict] = []

        for r in records:
            gold = (r.get(self.gold_field) or "").strip().upper()
            pred = _extract_choice(r.get("response", "")) or ""
            ok = bool(gold) and (pred == gold)
            n_correct += int(ok)

            cat = r.get("category", "all")
            per_cat.setdefault(cat, []).append(ok)

            annotated.append({
                **r,
                "predicted_choice": pred,
                "gold_choice": gold,
                "correct": ok,
                # Map into MIS-style fields for downstream code that expects them.
                "label_str": "Safe" if ok else "Unsafe",
                "label_perception": "Perception Level1" if ok else "Perception Level2",
            })

        n = len(records) or 1
        overall = {
            "Accuracy": n_correct / n,
            "ASR": 1 - n_correct / n,  # complement, for tools that always read ASR
            "RSR": n_correct / n,
            "RR": 0.0,
            "HR": 0.0,
        }
        per_category = {
            cat: {
                "Accuracy": sum(v) / len(v),
                "ASR": 1 - sum(v) / len(v),
                "RSR": sum(v) / len(v),
                "RR": 0.0,
                "HR": 0.0,
            }
            for cat, v in per_cat.items()
        }
        n_by_category = {cat: len(v) for cat, v in per_cat.items()}

        metrics = MetricsDict(
            overall=overall,
            per_category=per_category,
            n_samples=n,
            n_by_category=n_by_category,
        )
        return annotated, metrics
