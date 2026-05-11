"""
SafeBench loader (STUB).
HF source: huggingface.co/datasets/Zonghao2025/safebench

Pull command:
    huggingface-cli download Zonghao2025/safebench --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/safebench

Multi-image safety benchmark. Evaluator type TBD — read README on download.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class SafeBenchBenchmark(Benchmark):
    name = "safebench"
    metric_name = "ASR"
    metric_direction = "↓"
    evaluator_type = "gpt4o"  # provisional; confirm vs README

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "SafeBenchBenchmark.load is a stub. Pull HF repo "
            "Zonghao2025/safebench to data_links/safebench/, inspect its "
            "README, then implement load() returning the unified record schema."
        )
