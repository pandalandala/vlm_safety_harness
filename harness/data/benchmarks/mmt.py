"""
MMT-Bench loader (STUB).
HF source: OpenGVLab/MMT-Bench

Pull command:
    huggingface-cli download OpenGVLab/MMT-Bench --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mmt

Multi-task multimodal benchmark — accuracy.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class MMTBenchmark(Benchmark):
    name = "mmt"
    metric_name = "Accuracy"
    metric_direction = "↑"
    evaluator_type = "accuracy"

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "MMTBenchmark.load is a stub. Pull HF repo OpenGVLab/MMT-Bench "
            "to data_links/mmt/, then implement load()."
        )
