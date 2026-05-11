"""
HarmBench evaluator (STUB) — calls HarmBench classifier.

Pull HarmBench separately:
    git clone https://github.com/centerforaisafety/HarmBench /mnt/hdd/xuran/HarmBench

The classifier takes (behavior, response) → bool (jailbreak success). Wire up
when JailbreakV / similar benchmarks are run; until then, raise a clear error
so we don't silently fall through.
"""
from __future__ import annotations

from harness.evaluation.benchmark_evaluator import BenchmarkEvaluator
from harness.evaluation.metrics import MetricsDict


class HarmBenchEvaluator(BenchmarkEvaluator):
    metric_name = "Jailbreak Success Rate"
    metric_direction = "↓"

    def __init__(self, classifier_path: str = "/mnt/hdd/xuran/HarmBench"):
        self.classifier_path = classifier_path

    def evaluate(self, records: list[dict]) -> tuple[list[dict], MetricsDict]:
        raise NotImplementedError(
            "HarmBenchEvaluator is a stub. Clone HarmBench to "
            f"{self.classifier_path} and wire its classifier here, or override "
            "--evaluator-type to gpt4o for an LLM-judge fallback."
        )
