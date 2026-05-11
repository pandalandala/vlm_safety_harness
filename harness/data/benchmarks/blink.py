"""
BLINK loader (STUB).
HF source: BLINK-Benchmark/BLINK

Pull command:
    huggingface-cli download BLINK-Benchmark/BLINK --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/blink

Multi-image perception eval — accuracy.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class BLINKBenchmark(Benchmark):
    name = "blink"
    metric_name = "Accuracy"
    metric_direction = "↑"
    evaluator_type = "accuracy"

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "BLINKBenchmark.load is a stub. Pull HF repo BLINK-Benchmark/BLINK "
            "to data_links/blink/, then implement load()."
        )
