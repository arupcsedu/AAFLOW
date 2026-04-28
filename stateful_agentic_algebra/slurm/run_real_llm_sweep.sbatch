#!/bin/bash
#SBATCH --job-name=stateful-llm-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

# Override the generic GPU request at submit time for a specific accelerator.
# On this cluster, A100 nodes are available through the gpu partition:
#   sbatch -p gpu --gres=gpu:a100:1 ...

SAA_ENV_FILE="${SAA_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${PRJ_PATH:-/project/bi_dsc_community/drc_rag}}"
ENV_PATH="${ENV_PATH:-/scratch/${USER}/env}"
DATA_PATH="${DATA_PATH:-/scratch/${USER}/stateful_aaflow}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/saa_vllm_env/bin/python}"
SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-$ENV_PATH/drc_rag_bench_env/bin/python}"

MODEL_ID="${MODEL_ID:-gpt2,mistralai/Mistral-7B-Instruct-v0.3,meta-llama/Meta-Llama-3-8B-Instruct}"
BACKEND="${BACKEND:-hf,vllm,sglang}"
CONTEXT_GRID="${CONTEXT_GRID:-1024,4096,8192}"
OUTPUT_GRID="${OUTPUT_GRID:-64,128}"
AGENT_GRID="${AGENT_GRID:-2,4,8}"
BRANCH_GRID="${BRANCH_GRID:-2,4}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  JOB_PORT_OFFSET=$((SLURM_JOB_ID % 20000))
else
  JOB_PORT_OFFSET=0
fi
VLLM_PORT="${VLLM_PORT:-$((20000 + JOB_PORT_OFFSET))}"
SGLANG_PORT="${SGLANG_PORT:-$((30000 + JOB_PORT_OFFSET))}"
VLLM_SERVER_TIMEOUT_SEC="${VLLM_SERVER_TIMEOUT_SEC:-900}"
SGLANG_SERVER_TIMEOUT_SEC="${SGLANG_SERVER_TIMEOUT_SEC:-900}"
SGLANG_SERVER_EXTRA_ARGS="${SGLANG_SERVER_EXTRA_ARGS:---disable-overlap-schedule --disable-cuda-graph --skip-server-warmup}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-}"
SGLANG_GCC_MODULE="${SGLANG_GCC_MODULE:-gcc/12.4.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/real_llm_sweep/${SLURM_JOB_ID:-manual}}"
PLOT_PYTHON_BIN="${PLOT_PYTHON_BIN:-$SGLANG_PYTHON_BIN}"
GENERATE_PLOTS="${GENERATE_PLOTS:-1}"
CONFIG_PATH="${CONFIG_PATH:-}"

# Load a locally stored Hugging Face token when the submit environment did not
# already provide one. The file may contain shell exports such as
# `export HF_TOKEN=...`; the token value itself is never printed below.
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

if [[ -n "$SGLANG_CONTEXT_LENGTH" ]]; then
  SGLANG_SERVER_EXTRA_ARGS="$SGLANG_SERVER_EXTRA_ARGS --context-length $SGLANG_CONTEXT_LENGTH"
fi
if [[ -n "$SGLANG_MAX_PREFILL_TOKENS" ]]; then
  SGLANG_SERVER_EXTRA_ARGS="$SGLANG_SERVER_EXTRA_ARGS --max-prefill-tokens $SGLANG_MAX_PREFILL_TOKENS"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load "$SGLANG_GCC_MODULE" || true
  module load "$CUDA_MODULE" || true
fi
export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
NVIDIA_LIB_PATHS="$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages()]
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass
print(":".join(str(p) for root in roots for p in root.glob("nvidia/*/lib") if p.is_dir()))
PY
)"
if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_DIR/logs" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Job: ${SLURM_JOB_ID:-manual}" | tee "$OUTPUT_DIR/logs/job_info.log"
echo "Node list: ${SLURM_NODELIST:-unknown}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Backends: $BACKEND" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Models: $MODEL_ID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Python: $PYTHON_BIN" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "SGLang Python: $SGLANG_PYTHON_BIN" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "SGLang server extra args: $SGLANG_SERVER_EXTRA_ARGS" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "SGLang GCC module: $SGLANG_GCC_MODULE" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "CUDA module: $CUDA_MODULE" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "CC: ${CC:-unset}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "CXX: ${CXX:-unset}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Context grid: $CONTEXT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Output grid: $OUTPUT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Agent grid: $AGENT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Branch grid: $BRANCH_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "TP size: $TENSOR_PARALLEL_SIZE" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Output: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Plot Python: $PLOT_PYTHON_BIN" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Generate plots: $GENERATE_PLOTS" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Config path: ${CONFIG_PATH:-none}" | tee -a "$OUTPUT_DIR/logs/job_info.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUTPUT_DIR/logs/nvidia_smi_start.log"
  nvidia-smi -L | tee "$OUTPUT_DIR/logs/nvidia_smi_devices.log"
else
  echo "nvidia-smi not found" | tee "$OUTPUT_DIR/logs/nvidia_smi_start.log"
fi

echo "Running tiny dry orchestration smoke test..."
/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
    --models gpt2 \
    --backend hf,vllm,sglang \
    --context-grid 16 \
    --output-grid 4 \
    --agent-grid 1 \
    --branch-grid 1 \
    --num-prompts 1 \
    --sglang-python-bin "$SGLANG_PYTHON_BIN" \
    --output-dir "$OUTPUT_DIR/smoke_orchestration" \
    --dry-run \
  >"$OUTPUT_DIR/logs/smoke_orchestration.out" 2>"$OUTPUT_DIR/logs/smoke_orchestration.err"

echo "Running real multi-LLM sweep..."
RUNNER_ARGS=()
if [[ -n "$CONFIG_PATH" ]]; then
  RUNNER_ARGS+=(--config "$CONFIG_PATH")
else
  RUNNER_ARGS+=(
    --models "$MODEL_ID"
    --backend "$BACKEND"
    --context-grid "$CONTEXT_GRID"
    --output-grid "$OUTPUT_GRID"
    --agent-grid "$AGENT_GRID"
    --branch-grid "$BRANCH_GRID"
    --num-prompts "$NUM_PROMPTS"
  )
fi
RUNNER_ARGS+=(
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --vllm-port "$VLLM_PORT"
  --vllm-server-timeout-sec "$VLLM_SERVER_TIMEOUT_SEC"
  --sglang-port "$SGLANG_PORT"
  --sglang-server-timeout-sec "$SGLANG_SERVER_TIMEOUT_SEC"
  --sglang-python-bin "$SGLANG_PYTHON_BIN"
  --sglang-server-extra-args "$SGLANG_SERVER_EXTRA_ARGS"
  --output-dir "$OUTPUT_DIR"
)
/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
    "${RUNNER_ARGS[@]}" \
  >"$OUTPUT_DIR/logs/sweep.out" 2>"$OUTPUT_DIR/logs/sweep.err"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUTPUT_DIR/logs/nvidia_smi_end.log"
fi

if [[ "$GENERATE_PLOTS" == "1" && -s "$OUTPUT_DIR/results.csv" ]]; then
  echo "Generating real-LLM plots..."
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$OUTPUT_DIR/results.csv" \
    --output-dir "$OUTPUT_DIR/figures" \
    --real-llm \
    >"$OUTPUT_DIR/logs/plots.out" 2>"$OUTPUT_DIR/logs/plots.err" || {
      echo "Plot generation failed; see $OUTPUT_DIR/logs/plots.err"
    }
fi

echo "Done. Results:"
echo "  $OUTPUT_DIR/results_raw.jsonl"
echo "  $OUTPUT_DIR/results.csv"
echo "  $OUTPUT_DIR/summary_by_model.csv"
echo "  $OUTPUT_DIR/summary.out"
echo "  $OUTPUT_DIR/figures"
