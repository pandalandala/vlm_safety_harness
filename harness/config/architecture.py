"""Canonical architecture routing for experiment configs."""
from __future__ import annotations


ARCH_ALIASES: dict[str, str] = {}


def canonical_architecture(architecture: str) -> str:
    return ARCH_ALIASES.get(architecture, architecture)


def normalize_model_architecture(data: dict) -> dict:
    model = data.get("model")
    if isinstance(model, dict) and "architecture" in model:
        model["architecture"] = canonical_architecture(model["architecture"])
    return data
