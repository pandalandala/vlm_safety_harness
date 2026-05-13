#!/bin/bash
# run_logged.sh — run any command and log output to a categorized log file.
#
# Usage:
#   bash scripts/run_logged.sh <category> <label> <command...>
#
# Arguments:
#   category  : data_prep | prelim | main | training | capability | eval
#   label     : short name for this run, e.g. "A1_internvl" or "E3_advbench"
#   command...: the actual command to run (e.g. python ...)
#
# Examples:
#   bash scripts/run_logged.sh prelim A1_internvl python scripts/run_prelim.py --experiment A1 ...
#
#   bash scripts/run_logged.sh main E1_tier_b python scripts/run_experiment.py main/main_baseline_kimi_vl_a3b.yaml --skip-train
#
# Logs written to:
#   logs/<category>/YYYYMMDD_HHMMSS_<label>.log   (full stdout+stderr)
#   logs/commands_YYYY-MM-DD.log                   (summary: command + exit code)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_ROOT="$HARNESS_ROOT/logs"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <category> <label> <command...>" >&2
    echo "Categories: data_prep prelim main training capability eval" >&2
    exit 1
fi

CATEGORY="$1"
LABEL="$2"
shift 2

VALID_CATEGORIES="data_prep prelim main training capability eval"
if ! echo "$VALID_CATEGORIES" | grep -qw "$CATEGORY"; then
    echo "[warn] Unknown category '$CATEGORY'; defaulting to 'main'" >&2
    CATEGORY="main"
fi

TS="$(date +%Y%m%d_%H%M%S)"
DATE="$(date +%Y-%m-%d)"
CAT_DIR="$LOGS_ROOT/$CATEGORY"
mkdir -p "$CAT_DIR"

LOG_FILE="$CAT_DIR/${TS}_${LABEL}.log"
DAILY_LOG="$LOGS_ROOT/commands_${DATE}.log"

# Header
{
    echo "=== run_logged.sh ==="
    echo "Timestamp : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Category  : $CATEGORY"
    echo "Label     : $LABEL"
    echo "Command   : $*"
    echo "Log file  : $LOG_FILE"
    echo "============================================================"
} | tee -a "$LOG_FILE" | tee -a "$DAILY_LOG"

echo ""

# Run the command, tee to log file and console
set +e
"$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

# Footer
{
    echo ""
    echo "============================================================"
    echo "Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Exit code : $EXIT_CODE"
    echo "Log saved : $LOG_FILE"
    echo "============================================================"
} | tee -a "$LOG_FILE" | tee -a "$DAILY_LOG"

exit "$EXIT_CODE"
