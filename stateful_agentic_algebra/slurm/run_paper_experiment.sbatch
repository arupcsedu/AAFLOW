#!/bin/bash
#SBATCH --job-name=saa-paper-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAA_ENV_FILE="${SAA_ENV_FILE:-$SCRIPT_DIR/../env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${PRJ_PATH:-/project/bi_dsc_community/drc_rag}}"
ENV_PATH="${ENV_PATH:-/scratch/${USER}/env}"
DATA_PATH="${DATA_PATH:-/scratch/${USER}/stateful_aaflow}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/saa_vllm_env/bin/python}"
SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-$ENV_PATH/drc_rag_bench_env/bin/python}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must point to a paper experiment YAML file}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PLOT_PYTHON_BIN="${PLOT_PYTHON_BIN:-$PYTHON_BIN}"

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.hf_token"
fi

export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export SGLANG_PYTHON_BIN

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_ROOT"

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Config: $CONFIG_PATH"
echo "Output override: ${OUTPUT_DIR:-from config}"
echo "Project: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "SGLang Python: $SGLANG_PYTHON_BIN"
echo "HF_HOME: $HF_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi -L
else
  echo "nvidia-smi not found"
fi

RUN_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR/logs"
  RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

RESOLVED_OUTPUT_DIR="$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "${OUTPUT_DIR:-}"
import sys
from pathlib import Path
from stateful_agentic_algebra.config_utils import load_config_file

config = load_config_file(sys.argv[1])
output = sys.argv[2] or str(config.get("output_dir", "runs/stateful/paper_experiment"))
print(str(Path(output)))
PY
)"

/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.paper_experiment_runner "${RUN_ARGS[@]}"

if [[ -s "$RESOLVED_OUTPUT_DIR/results.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/results.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm || true
elif [[ -s "$RESOLVED_OUTPUT_DIR/crossover.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/crossover.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm || true
elif [[ -s "$RESOLVED_OUTPUT_DIR/consistency.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/consistency.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm || true
fi

echo "Done. Outputs:"
echo "  $RESOLVED_OUTPUT_DIR/benchmark.out"
echo "  $RESOLVED_OUTPUT_DIR/figures"
