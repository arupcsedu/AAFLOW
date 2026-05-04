#!/bin/bash
#SBATCH --job-name=stateful-plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

SAA_ENV_FILE="${SAA_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
else
  echo "Missing Stateful Agentic Algebra env file: $SAA_ENV_FILE" >&2
  exit 1
fi

BASELINE="${BASELINE:-AAFLOW+}"
WORKLOAD="${WORKLOAD:-linear_handoff}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
NUM_AGENTS="${NUM_AGENTS:-2}"
BRANCH_FACTOR="${BRANCH_FACTOR:-2}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/slurm_plot/${SLURM_JOB_ID:-manual}}"
RESULTS_CSV="${RESULTS_CSV:-$OUTPUT_DIR/results.csv}"
FIGURE_DIR="${FIGURE_DIR:-$OUTPUT_DIR/figures}"

mkdir -p "$OUTPUT_DIR" "$FIGURE_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

LAUNCHER=()
if [[ -n "${SLURM_JOB_ID:-}" && "${SAA_USE_SRUN:-1}" == "1" && -n "$(command -v srun || true)" ]]; then
  LAUNCHER=(srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}")
fi

echo "Launcher: ${LAUNCHER[*]:-direct}"

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "No results CSV found at $RESULTS_CSV; running a small mock experiment first."
  /usr/bin/time -v \
    "${LAUNCHER[@]}" \
    "$PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner \
      --baseline "$BASELINE" \
      --workload "$WORKLOAD" \
      --context-tokens "$CONTEXT_TOKENS" \
      --output-tokens "$OUTPUT_TOKENS" \
      --num-agents "$NUM_AGENTS" \
      --branch-factor "$BRANCH_FACTOR" \
      --num-requests "$NUM_REQUESTS" \
      --output-dir "$OUTPUT_DIR" \
    >"$OUTPUT_DIR/experiment.out" 2>"$OUTPUT_DIR/experiment.time.txt"
fi

echo "Plotting: $RESULTS_CSV"
echo "Figures: $FIGURE_DIR"

/usr/bin/time -v \
  "${LAUNCHER[@]}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESULTS_CSV" \
    --output-dir "$FIGURE_DIR" \
  >"$OUTPUT_DIR/plot.out" 2>"$OUTPUT_DIR/plot.time.txt"

echo "Done. Figures written to:"
echo "  $FIGURE_DIR"
