"""
SIUO (Safe Inputs Unsafe Outputs) loader (STUB).
HF source: sinwang/SIUO

Pull command:
    huggingface-cli download sinwang/SIUO --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/siuo
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class SIUOBenchmark(Benchmark):
    name = "siuo"
    metric_name = "ASR"
    metric_direction = "↓"
    evaluator_type = "gpt4o"

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "SIUOBenchmark.load is a stub. Pull HF repo sinwang/SIUO "
            "to data_links/siuo/, then implement load()."
        )
