"""GPT-4o evaluator adapter — wraps existing GPT4oEvaluator into BenchmarkEvaluator interface."""
from __future__ import annotations

from typing import Optional

from harness.evaluation.benchmark_evaluator import BenchmarkEvaluator
from harness.evaluation.gpt4o_evaluator import GPT4oEvaluator
from harness.evaluation.metrics import MetricsDict


class GPT4oBenchmarkEvaluator(BenchmarkEvaluator):
    metric_name = "ASR"
    metric_direction = "↓"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        max_concurrent: int = 20,
        max_tokens: int = 128,
        temperature: float = 0.0,
        compute_per_category: bool = True,
        output_jsonl: Optional[str] = None,
    ):
        self.output_jsonl = output_jsonl
        self._inner = GPT4oEvaluator(
            model=model,
            api_key_env=api_key_env,
            max_concurrent=max_concurrent,
            max_tokens=max_tokens,
            temperature=temperature,
            compute_per_category=compute_per_category,
        )

    def evaluate(self, records: list[dict]) -> tuple[list[dict], MetricsDict]:
        return self._inner.evaluate_records(records, output_jsonl=self.output_jsonl)
