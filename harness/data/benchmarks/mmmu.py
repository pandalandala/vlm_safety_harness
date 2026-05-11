"""
MMMU loader (STUB).
HF source: MMMU/MMMU

Pull command:
    huggingface-cli download MMMU/MMMU --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/mmmu

College-level multi-discipline multimodal eval. Multiple-choice → accuracy.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class MMMUBenchmark(Benchmark):
    name = "mmmu"
    metric_name = "Accuracy"
    metric_direction = "↑"
    evaluator_type = "accuracy"

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "MMMUBenchmark.load is a stub. Pull HF repo MMMU/MMMU to "
            "data_links/mmmu/, then implement load() with gold_answer field."
        )
