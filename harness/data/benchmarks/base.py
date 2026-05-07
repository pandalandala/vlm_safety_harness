"""Abstract base class for all benchmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Benchmark(ABC):
    name: str

    @abstractmethod
    def load(self) -> list[dict]:
        """
        Return list of unified records:
        {
          "id": int | str,
          "question": str,
          "image_path1": str,
          "image_path2": str,   # empty string if benchmark is single-image
          "category": str,
          "sub_category": str,
          "img_source": str,
          # benchmark-specific extra fields allowed
        }
        """
        ...

    @abstractmethod
    def get_root(self) -> Path:
        """Return path to benchmark data root."""
        ...

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
