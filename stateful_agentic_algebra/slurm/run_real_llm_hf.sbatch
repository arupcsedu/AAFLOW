#!/bin/bash
#SBATCH --job-name=stateful-hf-llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

# Resource requests are intentionally supplied at submit time. Example:
#   sbatch -A <account> -p <partition> --gres=gpu:<type>:1 ...

SAA_ENV_FILE="${SAA_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
else
  echo "Missing Stateful Agentic Algebra env file: $SAA_ENV_FILE" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-gpt2}"
BACKEND="${BACKEND:-hf}"
CONTEXT_GRID="${CONTEXT_GRID:-128}"
OUTPUT_GRID="${OUTPUT_GRID:-16}"
AGENT_GRID="${AGENT_GRID:-2}"
BRANCH_GRID="${BRANCH_GRID:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/real_llm_hf/${SLURM_JOB_ID:-manual}}"

export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
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

LOCAL_GPU_INDICES=""
if command -v nvidia-smi >/dev/null 2>&1; then
  LOCAL_GPU_INDICES="$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits 2>/dev/null | \
    awk -F, '
      {
        idx=$1; name=$2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (name !~ /Display/ && name ~ /(A100|H100|V100|L40|L4|RTX|Tesla|NVIDIA)/) {
          print idx
        }
      }
    ' | paste -sd, -)"
fi
LOCAL_GPU_COUNT=0
if [[ -n "$LOCAL_GPU_INDICES" ]]; then
  LOCAL_GPU_COUNT="$(awk -F, '{print NF}' <<<"$LOCAL_GPU_INDICES")"
fi
LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-0}"
if [[ -z "${SLURM_JOB_ID:-}" && "$LOCAL_GPU_COUNT" -gt 0 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  LOCAL_GPU_LIMIT="${SAA_LOCAL_GPUS:-$LOCAL_GPU_COUNT}"
  if ! [[ "$LOCAL_GPU_LIMIT" =~ ^[0-9]+$ ]] || [[ "$LOCAL_GPU_LIMIT" -lt 1 ]]; then
    LOCAL_GPU_LIMIT="$LOCAL_GPU_COUNT"
  fi
  if [[ "$LOCAL_GPU_LIMIT" -gt "$LOCAL_GPU_COUNT" ]]; then
    LOCAL_GPU_LIMIT="$LOCAL_GPU_COUNT"
  fi
  export CUDA_VISIBLE_DEVICES="$(cut -d, -f "1-$LOCAL_GPU_LIMIT" <<<"$LOCAL_GPU_INDICES")"
fi

LAUNCHER=()
if [[ -n "${SLURM_JOB_ID:-}" && "${SAA_USE_SRUN:-1}" == "1" && -n "$(command -v srun || true)" ]]; then
  LAUNCHER=(srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}")
fi

echo "Job: ${SLURM_JOB_ID:-manual}" | tee "$OUTPUT_DIR/logs/job_info.log"
echo "Node list: ${SLURM_NODELIST:-unknown}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Launcher: ${LAUNCHER[*]:-direct}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Backend: $BACKEND" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Model: $MODEL_ID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "TP size: $TENSOR_PARALLEL_SIZE" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "HF device map: $HF_DEVICE_MAP" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Detected compute GPU indices: ${LOCAL_GPU_INDICES:-none}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Output: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "HF_HOME: $HF_HOME" | tee -a "$OUTPUT_DIR/logs/job_info.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUTPUT_DIR/logs/nvidia_smi_start.log"
  nvidia-smi -L | tee "$OUTPUT_DIR/logs/nvidia_smi_devices.log"
else
  echo "nvidia-smi not found" | tee "$OUTPUT_DIR/logs/nvidia_smi_start.log"
fi

echo "Running tiny HF smoke test..."
/usr/bin/time -v \
  "${LAUNCHER[@]}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
    --models gpt2 \
    --backend hf \
    --context-grid 16 \
    --output-grid 4 \
    --agent-grid 1 \
    --branch-grid 1 \
    --num-prompts 1 \
    --output-dir "$OUTPUT_DIR/smoke_hf" \
  >"$OUTPUT_DIR/logs/smoke_hf.out" 2>"$OUTPUT_DIR/logs/smoke_hf.err"

echo "Running HF real LLM sweep..."
/usr/bin/time -v \
  "${LAUNCHER[@]}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
    --models "$MODEL_ID" \
    --backend "$BACKEND" \
    --context-grid "$CONTEXT_GRID" \
    --output-grid "$OUTPUT_GRID" \
    --agent-grid "$AGENT_GRID" \
    --branch-grid "$BRANCH_GRID" \
    --num-prompts "$NUM_PROMPTS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --hf-device-map "$HF_DEVICE_MAP" \
    --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/sweep.out" 2>"$OUTPUT_DIR/logs/sweep.err"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUTPUT_DIR/logs/nvidia_smi_end.log"
fi

echo "Done. Results:"
echo "  $OUTPUT_DIR/results_raw.jsonl"
echo "  $OUTPUT_DIR/results.csv"
echo "  $OUTPUT_DIR/summary_by_model.csv"
