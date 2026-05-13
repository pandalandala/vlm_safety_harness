"""
BenchmarkEvaluator ABC — pluggable scoring interface for E3 / E4.

Each Benchmark loader declares an `evaluator_type` ∈ {gpt4o, rule, harmbench, accuracy}.
`run_eval_only.py` reads that attribute and dispatches to the matching subclass.

Subclass contract:
    metric_name: str          — column header for the E3/E4 result table
    metric_direction: ↑|↓     — for ↑/↓ in table headers
    evaluate(records) → (records_with_labels, MetricsDict)
    evaluate_file(inference_jsonl, output_jsonl) → same, with file I/O helper
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
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

    def evaluate_file(
        self,
        inference_jsonl: Path,
        output_jsonl: Path,
        resume: bool = True,
    ) -> tuple[list[dict], MetricsDict]:
        records = []
        with open(inference_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        annotated, metrics = self.evaluate(records)
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w") as f:
            for r in annotated:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return annotated, metrics


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
