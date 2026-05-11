"""
MuirBench loader (STUB).
HF source: MUIRBENCH/MUIRBENCH

Pull command:
    huggingface-cli download MUIRBENCH/MUIRBENCH --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/muirbench

Multi-image understanding benchmark — most relevant to DREAMS multi-image setting.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class MuirBenchBenchmark(Benchmark):
    name = "muirbench"
    metric_name = "Accuracy"
    metric_direction = "↑"
    evaluator_type = "accuracy"

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "MuirBenchBenchmark.load is a stub. Pull HF repo MUIRBENCH/MUIRBENCH "
            "to data_links/muirbench/, then implement load()."
        )
