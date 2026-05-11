"""
BenchmarkEvaluator ABC — pluggable scoring interface for E3 / E4.

Each Benchmark loader declares an `evaluator_type` ∈ {gpt4o, rule, harmbench, accuracy}.
`run_eval_only.py` reads that attribute and dispatches to the matching subclass.

Subclass contract:
    metric_name: str          — column header for the E3/E4 result table
    metric_direction: ↑|↓     — for ↑/↓ in table headers
    evaluate(records) → (records_with_labels, MetricsDict)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from harness.evaluation.metrics import MetricsDict


class BenchmarkEvaluator(ABC):
    metric_name: str = "ASR"
    metric_direction: Literal["↑", "↓"] = "↓"

    @abstractmethod
    def evaluate(self, records: list[dict]) -> tuple[list[dict], MetricsDict]:
        """
        Score a list of inference-output records.

        Args:
            records: list of dicts, each with at least {id, question, response}.
                     Some evaluators also need image paths (gpt4o), gold answers
                     (accuracy), or category metadata.

        Returns:
            (annotated_records, MetricsDict)
        """
        ...


def get_evaluator(
    evaluator_type: Literal["gpt4o", "rule", "harmbench", "accuracy"],
    **kwargs,
) -> BenchmarkEvaluator:
    """Factory: instantiate the right BenchmarkEvaluator subclass."""
    if evaluator_type == "gpt4o":
        from harness.evaluation.evaluators.gpt4o import GPT4oBenchmarkEvaluator
        return GPT4oBenchmarkEvaluator(**kwargs)
    if evaluator_type == "rule":
        from harness.evaluation.evaluators.rule_based import RuleBasedEvaluator
        return RuleBasedEvaluator(**kwargs)
    if evaluator_type == "harmbench":
        from harness.evaluation.evaluators.harmbench import HarmBenchEvaluator
        return HarmBenchEvaluator(**kwargs)
    if evaluator_type == "accuracy":
        from harness.evaluation.evaluators.accuracy import AccuracyEvaluator
        return AccuracyEvaluator(**kwargs)
    raise ValueError(f"Unknown evaluator_type: {evaluator_type}")
