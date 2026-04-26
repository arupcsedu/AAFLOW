#!/bin/bash
#SBATCH --job-name=stateful-llm-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/stateful_agentic_algebra/slurm/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/stateful_agentic_algebra/slurm/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

# Override the generic GPU request at submit time for a specific accelerator:
#   sbatch --gres=gpu:a100:2 ...
#   sbatch --gres=gpu:h100:2 ...

PROJECT_ROOT="${PROJECT_ROOT:-/project/bi_dsc_community/drc_rag}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/djy8hg/env/saa_vllm_env/bin/python}"

MODEL_ID="${MODEL_ID:-gpt2,mistralai/Mistral-7B-Instruct-v0.3,meta-llama/Meta-Llama-3-8B-Instruct}"
BACKEND="${BACKEND:-hf,vllm}"
CONTEXT_GRID="${CONTEXT_GRID:-1024,4096,8192}"
OUTPUT_GRID="${OUTPUT_GRID:-64,128}"
AGENT_GRID="${AGENT_GRID:-2,4,8}"
BRANCH_GRID="${BRANCH_GRID:-2,4}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/stateful/real_llm_sweep/${SLURM_JOB_ID:-manual}}"

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p "$OUTPUT_DIR/logs" "$HF_HOME" "$TRANSFORMERS_CACHE"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Job: ${SLURM_JOB_ID:-manual}" | tee "$OUTPUT_DIR/logs/job_info.log"
echo "Node list: ${SLURM_NODELIST:-unknown}" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Backends: $BACKEND" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Models: $MODEL_ID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Context grid: $CONTEXT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Output grid: $OUTPUT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Agent grid: $AGENT_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Branch grid: $BRANCH_GRID" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "TP size: $TENSOR_PARALLEL_SIZE" | tee -a "$OUTPUT_DIR/logs/job_info.log"
echo "Output: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/logs/job_info.log"

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
    --backend hf,vllm \
    --context-grid 16 \
    --output-grid 4 \
    --agent-grid 1 \
    --branch-grid 1 \
    --num-prompts 1 \
    --output-dir "$OUTPUT_DIR/smoke_orchestration" \
    --dry-run \
  >"$OUTPUT_DIR/logs/smoke_orchestration.out" 2>"$OUTPUT_DIR/logs/smoke_orchestration.err"

echo "Running real multi-LLM sweep..."
/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
    --models "$MODEL_ID" \
    --backend "$BACKEND" \
    --context-grid "$CONTEXT_GRID" \
    --output-grid "$OUTPUT_GRID" \
    --agent-grid "$AGENT_GRID" \
    --branch-grid "$BRANCH_GRID" \
    --num-prompts "$NUM_PROMPTS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/sweep.out" 2>"$OUTPUT_DIR/logs/sweep.err"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUTPUT_DIR/logs/nvidia_smi_end.log"
fi

echo "Done. Results:"
echo "  $OUTPUT_DIR/results_raw.jsonl"
echo "  $OUTPUT_DIR/results.csv"
echo "  $OUTPUT_DIR/summary_by_model.csv"
