"""
YAML config loader with _extends inheritance and --override support.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .architecture import normalize_model_architecture
from .schema import ExperimentConfig

CONFIGS_ROOT = Path(__file__).parent.parent.parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_extends(data: dict, config_path: Path) -> dict:
    """
    Resolve _extends key: load the base config and merge current on top.
    _extends path is relative to configs/ root or absolute.
    """
    base_ref = data.pop("_extends", None)
    if base_ref is None:
        return data

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = CONFIGS_ROOT / base_ref
    if not base_path.exists():
        raise FileNotFoundError(f"_extends target not found: {base_path}")

    with open(base_path) as f:
        base_data = yaml.safe_load(f) or {}
    base_data = _resolve_extends(base_data, base_path)

    return _deep_merge(base_data, data)


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation overrides like "training.learning_rate=1e-4".
    Supports nested keys and basic type coercion (bool, int, float, None, str).
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")

        # Type coercion
        value: Any
        if raw_value.lower() in ("true", "false"):
            value = raw_value.lower() == "true"
        elif raw_value.lower() in ("null", "none", "~"):
            value = None
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        # Set nested key
        d = data
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    return data


class ConfigLoader:
    @staticmethod
    def load(
        config_path: str | Path,
        overrides: list[str] | None = None,
    ) -> ExperimentConfig:
        """
        Load and validate an experiment config YAML.

        Args:
            config_path: Path to YAML file (relative to configs/experiments/ or absolute)
            overrides: List of "key.path=value" strings to override after loading

        Returns:
            Validated ExperimentConfig
        """
        path = Path(config_path)
        if not path.is_absolute():
            # Try relative to configs/experiments/ first, then configs/
            candidates = [
                CONFIGS_ROOT / "experiments" / path,
                CONFIGS_ROOT / path,
                Path.cwd() / path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Config not found: {config_path}\nSearched: {candidates}"
                )

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        data = _resolve_extends(data, path)

        if overrides:
            data = _apply_overrides(data, overrides)

        data = normalize_model_architecture(data)

        return ExperimentConfig(**data)
