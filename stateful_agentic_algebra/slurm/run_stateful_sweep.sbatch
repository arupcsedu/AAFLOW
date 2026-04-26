#!/bin/bash
#SBATCH --job-name=stateful-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/stateful_agentic_algebra/slurm/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/stateful_agentic_algebra/slurm/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

PROJECT_ROOT="${PROJECT_ROOT:-/project/bi_dsc_community/drc_rag}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE="${BASELINE:-all}"
WORKLOAD="${WORKLOAD:-all}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-1024,4096,8192}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
NUM_AGENTS="${NUM_AGENTS:-2,4,8}"
BRANCH_FACTOR="${BRANCH_FACTOR:-2,4}"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/slurm_sweep/${SLURM_JOB_ID:-manual}}"

NVIDIA_LIB_PATHS="$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages()]
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass
paths = []
for root in roots:
    for path in root.glob("nvidia/*/lib"):
        if path.is_dir():
            paths.append(str(path))
print(":".join(paths))
PY
)"
if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Baseline: $BASELINE"
echo "Workload: $WORKLOAD"
echo "Context grid: $CONTEXT_TOKENS"
echo "Output tokens: $OUTPUT_TOKENS"
echo "Agent grid: $NUM_AGENTS"
echo "Branch grid: $BRANCH_FACTOR"
echo "Requests: $NUM_REQUESTS"
echo "Output: $OUTPUT_DIR"

cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner
)

if [[ "$BASELINE" == "all" ]]; then
  cmd+=(--all-baselines)
else
  cmd+=(--baseline "$BASELINE")
fi

if [[ "$WORKLOAD" == "all" ]]; then
  cmd+=(--all-workloads)
else
  cmd+=(--workload "$WORKLOAD")
fi

cmd+=(
  --context-grid "$CONTEXT_TOKENS"
  --agent-grid "$NUM_AGENTS"
  --branch-grid "$BRANCH_FACTOR"
  --output-tokens "$OUTPUT_TOKENS"
  --num-requests "$NUM_REQUESTS"
  --output-dir "$OUTPUT_DIR"
)

/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  "${cmd[@]}" \
  >"$OUTPUT_DIR/experiment.out" 2>"$OUTPUT_DIR/time.txt"

echo "Done. Outputs:"
echo "  $OUTPUT_DIR/results.json"
echo "  $OUTPUT_DIR/results.csv"
echo "  $OUTPUT_DIR/config.json"
echo "  $OUTPUT_DIR/skipped_baselines.json"
