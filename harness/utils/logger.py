"""
Session logging for vlm_safety_harness.

Two public APIs:

1. get_logger(name, category) → logging.Logger
   Structured per-logger messages (info/error) written to file + terminal.
   Unchanged from original — backward-compatible.

2. init_session(script_name, tag, category) → Path
   Tees ALL stdout+stderr (print(), tracebacks, subprocess output) to a
   timestamped log file.  Call once at the top of main().  Log path is
   printed at session end automatically via atexit.

Categories:
  data_prep   — generate_responses, build_cf_pairs, dataset prep
  prelim      — A1 / A2 / A3 / A4 experiments
  main        — E1–E5 main experiments
  training    — SFT / LLaMA-Factory runs
  capability  — E4 capability benchmarks
  eval        — GPT-4o evaluation / metric aggregation
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

HARNESS_ROOT = Path(__file__).parent.parent.parent
LOGS_ROOT = HARNESS_ROOT / "logs"

VALID_CATEGORIES = frozenset(
    ["data_prep", "prelim", "main", "training", "capability", "eval"]
)

# ── Module-level singleton so init_session() is idempotent ────────────────
_active_session: Optional["TeeSession"] = None


# ─────────────────────────────────────────────────────────────────────────────
# TeeSession
# ─────────────────────────────────────────────────────────────────────────────

class TeeSession:
    """
    Redirect stdout (fd 1) and stderr (fd 2) to both the terminal and a log
    file for the lifetime of a script run.

    Uses os.dup2 at the file-descriptor level so that subprocess output
    (e.g. torchrun, llamafactory) is captured without any extra plumbing.

    Usage:
        session = TeeSession()
        log_path = session.start("run_experiment", tag="main_dreams_internvl3_5", category="main")
        ...
        session.stop()   # called automatically via atexit
    """

    def __init__(self) -> None:
        self._log_path: Optional[Path] = None
        self._log_fd: Optional[int] = None
        self._saved_stdout_fd: Optional[int] = None
        self._saved_stderr_fd: Optional[int] = None
        self._tee_thread: Optional[threading.Thread] = None
        self._pipe_r: Optional[int] = None
        self._pipe_w: Optional[int] = None
        self._started = False

    # ------------------------------------------------------------------
    def start(self, script_name: str, tag: str = "", category: str = "main") -> Path:
        if self._started:
            return self._log_path  # type: ignore[return-value]

        if category not in VALID_CATEGORIES:
            category = "main"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = _sanitize(tag)[:48]
        filename = f"{ts}_{script_name}" + (f"_{safe_tag}" if safe_tag else "") + ".log"
        cat_dir = LOGS_ROOT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = cat_dir / filename

        # Open log file fd
        self._log_fd = os.open(
            str(self._log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )

        # Save originals
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Create pipe: writes to fd 1/2 go to pipe_w; tee thread reads pipe_r
        self._pipe_r, self._pipe_w = os.pipe()

        # Redirect fd 1 and fd 2 to the write end of the pipe
        os.dup2(self._pipe_w, 1)
        os.dup2(self._pipe_w, 2)
        os.close(self._pipe_w)  # close our extra reference
        self._pipe_w = None

        # Python-level objects also need updating
        # (TextIOWrapper wraps the underlying fd, so flushing them is enough)
        sys.stdout = open(1, "w", buffering=1, closefd=False, encoding="utf-8")
        sys.stderr = open(2, "w", buffering=1, closefd=False, encoding="utf-8")

        # Start tee thread: pipe_r → terminal (saved_stdout_fd) + log file
        self._tee_thread = threading.Thread(
            target=self._tee_loop,
            args=(self._pipe_r, self._saved_stdout_fd, self._log_fd),
            daemon=True,
            name="session-tee",
        )
        self._tee_thread.start()

        self._started = True

        # Print session header to the new (teed) stdout
        print(f"[session] log → {self._log_path}", flush=True)
        return self._log_path

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self._started:
            return
        self._started = False  # prevent double-stop

        # Flush Python-level stdout/stderr
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # Closing fd 1 / fd 2 signals EOF to the tee thread
        try:
            os.close(1)
        except OSError:
            pass
        try:
            os.close(2)
        except OSError:
            pass

        # Wait for tee thread to drain
        if self._tee_thread is not None:
            self._tee_thread.join(timeout=5)

        # Restore originals
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)

        # Close log fd
        if self._log_fd is not None:
            try:
                os.close(self._log_fd)
            except OSError:
                pass

        # Restore Python objects
        sys.stdout = open(1, "w", buffering=1, closefd=False, encoding="utf-8")
        sys.stderr = open(2, "w", buffering=1, closefd=False, encoding="utf-8")

        # Final banner on the restored terminal
        if self._log_path:
            banner = "═" * 63
            print(f"\n{banner}", flush=True)
            print(f"  Session log saved → {self._log_path}", flush=True)
            print(f"{banner}\n", flush=True)

    # ------------------------------------------------------------------
    def __enter__(self) -> "TeeSession":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    @staticmethod
    def _tee_loop(pipe_r: int, term_fd: int, log_fd: int) -> None:
        """Read from pipe, write to both terminal and log file."""
        buf_size = 4096
        while True:
            try:
                chunk = os.read(pipe_r, buf_size)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(term_fd, chunk)
            except OSError:
                pass
            try:
                os.write(log_fd, chunk)
            except OSError:
                pass
        try:
            os.close(pipe_r)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_session(
    script_name: str,
    tag: str = "",
    category: str = "main",
) -> Path:
    """
    Start a tee session for the current script.  Idempotent — calling more
    than once returns the existing log path without creating a new session.

    Args:
        script_name: short name for the log file, e.g. "run_experiment"
        tag:         experiment identifier appended to the filename,
                     e.g. args.config or args.experiment_id
        category:    log sub-directory; one of VALID_CATEGORIES

    Returns:
        Path to the log file.
    """
    global _active_session
    if _active_session is not None:
        return _active_session._log_path  # type: ignore[return-value]

    session = TeeSession()
    log_path = session.start(script_name, tag=tag, category=category)
    _active_session = session

    # Register cleanup on normal exit
    atexit.register(session.stop)

    # Register cleanup on SIGTERM
    def _sigterm_handler(signum, frame):  # noqa: ANN001
        session.stop()
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (OSError, ValueError):
        pass  # may fail in non-main threads

    return log_path


def get_logger(name: str, category: str = "main") -> logging.Logger:
    """
    Return a configured structured logger (backward-compatible).

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(s: str) -> str:
    """Make a string safe for use in a filename."""
    s = os.path.basename(s)           # strip directory parts (e.g. config paths)
    s = re.sub(r"\.ya?ml$", "", s)    # drop .yaml / .yml suffix
    s = re.sub(r"[^\w\-]", "_", s)   # replace non-word chars with _
    s = re.sub(r"_+", "_", s)        # collapse repeated _
    return s.strip("_")
