"""
JailbreakV-28K loader (STUB).
HF source: JailbreakV-28K/JailBreakV-28k

Pull command:
    huggingface-cli download JailbreakV-28K/JailBreakV-28k --repo-type dataset \
        --local-dir /mnt/hdd/xuran/vlm_safety_harness/data_links/jailbreakv

Adversarial multimodal jailbreak set. Evaluator typically `harmbench` or `gpt4o`.
"""
from __future__ import annotations

from pathlib import Path
from harness.data.benchmarks.base import Benchmark


class JailbreakVBenchmark(Benchmark):
    name = "jailbreakv"
    metric_name = "Jailbreak Success Rate"
    metric_direction = "↓"
    evaluator_type = "harmbench"  # provisional

    def __init__(self, data_path: Path, **kwargs):
        self.data_path = Path(data_path)

    def get_root(self) -> Path:
        return self.data_path

    def load(self) -> list[dict]:
        raise NotImplementedError(
            "JailbreakVBenchmark.load is a stub. Pull HF repo "
            "JailbreakV-28K/JailBreakV-28k to data_links/jailbreakv/, "
            "inspect its README + scoring script, then implement load()."
        )
