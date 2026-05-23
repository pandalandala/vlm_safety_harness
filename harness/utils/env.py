from __future__ import annotations

import os
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
EVAL_MODEL_ENV = "VLM_SAFETY_EVAL_MODEL"


def load_project_env(env_file: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from the project .env into os.environ if unset."""
    path = env_file or ENV_FILE
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        try:
            parsed = shlex.split(value, posix=True)
            if len(parsed) == 0:
                value = ""
            elif len(parsed) == 1:
                value = parsed[0]
        except ValueError:
            value = value.strip("\"'")

        os.environ.setdefault(key, value)
        loaded[key] = os.environ[key]
    return loaded


def resolve_eval_model(default: str = "gpt-4o", *, use_env_override: bool = False) -> str:
    """Resolve the evaluation model, optionally letting the project env override it."""
    model = os.environ.get(EVAL_MODEL_ENV, "").strip()
    if use_env_override and model:
        return model
    return default
