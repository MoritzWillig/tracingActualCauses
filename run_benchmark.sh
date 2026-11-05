#!/usr/bin/env bash
# run_benchmark.sh  –  Convenience launcher for the TrAC benchmark.
#
# Usage:
#   bash run_benchmark.sh [WORKERS [TIMEOUT [OUTPUT [MIN_NODES [MAX_NODES]]]]]
#   bash run_benchmark.sh --plot-only [OUTPUT]
#   bash run_benchmark.sh --update [WORKERS ...]
#
# Flags (must come before positional arguments):
#   --plot-only    Skip the benchmark; regenerate the plot from existing
#                  per-cell CSV files in results/data/.
#   --update       Re-run all tasks even if results already exist in the
#                  per-cell CSVs.  Timings from the previous run are kept
#                  for strategies not requested in this run, so you can
#                  add a single strategy (e.g. BRUTE_FORCE) without losing
#                  previously-recorded results.
#
# Positional arguments (benchmark mode only):
#   WORKERS     parallel worker processes          (default: 8)
#   TIMEOUT     per-task timeout in seconds        (default: 120)
#   OUTPUT      CSV output path                    (default: results/benchmark_results.csv)
#   MIN_NODES   smallest node count                (default: 5)
#   MAX_NODES   largest node count (inclusive)     (default: 20)
#
# Environment variables override positional arguments:
#   WORKERS, TIMEOUT, OUTPUT, SEED, MIN_NODES, MAX_NODES

set -euo pipefail

PLOT_ONLY=false
UPDATE_FLAG=()
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --plot-only) PLOT_ONLY=true ;;
        --update)    UPDATE_FLAG=(--update) ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

# Defaults
WORKERS=${WORKERS:-8}
TIMEOUT=${TIMEOUT:-120}
OUTPUT=${OUTPUT:-results/trac.csv}
SEED=${SEED:-123}
MIN_NODES=${MIN_NODES:-5}
MAX_NODES=${MAX_NODES:-20}
REPS=${REPS:-20}
HP2SAT_STRATEGIES=${HP2SAT_STRATEGIES:-SAT,MAX_SAT}

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        echo "Error: no Python interpreter found (set PYTHON= to override)" >&2
        exit 1
    fi
fi

if [ ${#POSITIONAL[@]} -ge 1 ]; then WORKERS=${POSITIONAL[0]};   fi
if [ ${#POSITIONAL[@]} -ge 2 ]; then TIMEOUT=${POSITIONAL[1]};   fi
if [ ${#POSITIONAL[@]} -ge 3 ]; then OUTPUT=${POSITIONAL[2]};    fi
if [ ${#POSITIONAL[@]} -ge 4 ]; then MIN_NODES=${POSITIONAL[3]}; fi
if [ ${#POSITIONAL[@]} -ge 5 ]; then MAX_NODES=${POSITIONAL[4]}; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$(dirname "$OUTPUT")"
CELLS_DIR="${RESULTS_DIR}/data"
PLOT_OUTPUT="${OUTPUT%.csv}.pdf"

echo "╔══════════════════════════════════════════════╗"
echo "║  TrAC Benchmark Launcher                     ║"
echo "╚══════════════════════════════════════════════╝"

if $PLOT_ONLY; then
    echo "  mode        : plot only"
    echo "  cells dir   : $CELLS_DIR"
    echo "  plot output : $PLOT_OUTPUT"
    echo "  python      : $PYTHON"
    echo
    "$PYTHON" "$SCRIPT_DIR/trac/plot_results.py" \
        --results-dir "$CELLS_DIR" \
        --output      "$PLOT_OUTPUT"
else
    echo "  workers    : $WORKERS"
    echo "  timeout    : ${TIMEOUT} s/task"
    echo "  nodes      : ${MIN_NODES} – ${MAX_NODES}"
    echo "  reps       : $REPS"
    echo "  hp2sat     : $HP2SAT_STRATEGIES"
    echo "  output     : $OUTPUT"
    echo "  cells dir  : $CELLS_DIR"
    echo "  seed       : $SEED"
    echo "  python     : $PYTHON"
    echo
    NODE_COUNTS_ARG=()
    if [ -n "${NODE_COUNTS:-}" ]; then
        NODE_COUNTS_ARG=(--node-counts "$NODE_COUNTS")
    else
        NODE_COUNTS_ARG=(--min-nodes "$MIN_NODES" --max-nodes "$MAX_NODES")
    fi

    LOG_FILE="${OUTPUT%.csv}.stderr.log"
    echo "  stderr log : $LOG_FILE"
    echo

    "$PYTHON" "$SCRIPT_DIR/trac/benchmark_trac.py" \
        --workers            "$WORKERS"            \
        --timeout            "$TIMEOUT"            \
        --output             "$OUTPUT"             \
        --seed               "$SEED"               \
        --repetitions        "$REPS"               \
        "${NODE_COUNTS_ARG[@]}"                    \
        --hp2sat-strategies  "$HP2SAT_STRATEGIES"  \
        "${UPDATE_FLAG[@]}"                        \
        2>"$LOG_FILE"
fi
