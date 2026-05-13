"""
Structured command logging for vlm_safety_harness.

Every script imports get_logger(name, category) and calls log.info/log.error.
Writes to:
  logs/<category>/YYYYMMDD_HHMMSS_<name>.log   — per-run detail
  logs/commands_YYYY-MM-DD.log                  — daily rolling summary

Categories:
  data_prep   — generate_responses, build_cf_pairs, dataset prep
  prelim      — A1 / A2 / A3 / A4 experiments
  main        — E1–E5 main experiments
  training    — SFT / LLaMA-Factory runs
  capability  — E4 capability benchmarks
  eval        — GPT-4o evaluation / metric aggregation
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

HARNESS_ROOT = Path(__file__).parent.parent.parent
LOGS_ROOT = HARNESS_ROOT / "logs"

VALID_CATEGORIES = frozenset(
    ["data_prep", "prelim", "main", "training", "capability", "eval"]
)


def get_logger(name: str, category: str = "main") -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name:     short script name, e.g. "generate_responses" or "run_prelim_A1"
        category: one of VALID_CATEGORIES
    """
    if category not in VALID_CATEGORIES:
        category = "main"

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    cat_dir = LOGS_ROOT / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    log_file = cat_dir / f"{ts_str}_{name}.log"
    daily_log = LOGS_ROOT / f"commands_{date_str}.log"

    logger_id = f"harness.{category}.{name}.{ts_str}"
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    dh = logging.FileHandler(daily_log, encoding="utf-8")
    dh.setLevel(logging.INFO)
    dh.setFormatter(fmt)
    logger.addHandler(dh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    logger.info("=== run start: %s  log → %s ===", name, log_file)
    return logger
