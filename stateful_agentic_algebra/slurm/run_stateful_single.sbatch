#!/bin/bash
#SBATCH --job-name=stateful-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

SAA_ENV_FILE="${SAA_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${PRJ_PATH:-/project/bi_dsc_community/drc_rag}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE="${BASELINE:-AAFLOW+}"
WORKLOAD="${WORKLOAD:-linear_handoff}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
NUM_AGENTS="${NUM_AGENTS:-2}"
BRANCH_FACTOR="${BRANCH_FACTOR:-2}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/slurm_single/${SLURM_JOB_ID:-manual}}"

mkdir -p "$OUTPUT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Baseline: $BASELINE"
echo "Workload: $WORKLOAD"
echo "Context tokens: $CONTEXT_TOKENS"
echo "Output tokens: $OUTPUT_TOKENS"
echo "Agents: $NUM_AGENTS"
echo "Branch factor: $BRANCH_FACTOR"
echo "Requests: $NUM_REQUESTS"
echo "Output: $OUTPUT_DIR"

cd "$PROJECT_ROOT"

/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner \
    --baseline "$BASELINE" \
    --workload "$WORKLOAD" \
    --context-tokens "$CONTEXT_TOKENS" \
    --output-tokens "$OUTPUT_TOKENS" \
    --num-agents "$NUM_AGENTS" \
    --branch-factor "$BRANCH_FACTOR" \
    --num-requests "$NUM_REQUESTS" \
    --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/experiment.out" 2>"$OUTPUT_DIR/time.txt"

echo "Done. Outputs:"
echo "  $OUTPUT_DIR/results.json"
echo "  $OUTPUT_DIR/results.csv"
echo "  $OUTPUT_DIR/config.json"
echo "  $OUTPUT_DIR/skipped_baselines.json"
