#!/bin/bash
#SBATCH --job-name=aaflow-hf-models
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-hf-models-%j.out
#SBATCH --error=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-hf-models-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/djy8hg/workdir/AAFLOW}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python}"
HF_HOME="${HF_HOME:-/scratch/djy8hg/huggingface}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/benchmark/slurm_runs_hf_models}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}}"

MODELS="${MODELS:-llama3-8b,mistral-7b}"
FILES="${FILES:-16}"
CHUNKS="${CHUNKS:-128}"
CHUNK_CHARS="${CHUNK_CHARS:-900}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "$RUN_DIR"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_NODELIST=$SLURM_NODELIST"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "MODELS=$MODELS"
  echo "HF_HOME=$HF_HOME"
  echo "PYTHON_BIN=$PYTHON_BIN"
} > "$RUN_DIR/allocation.txt"

/usr/bin/time -v "$PYTHON_BIN" "$PROJECT_ROOT/benchmark/benchmark_hf_models.py" \
  --models "$MODELS" \
  --files "$FILES" \
  --chunks "$CHUNKS" \
  --chunk-chars "$CHUNK_CHARS" \
  --embed-batch-size "$EMBED_BATCH_SIZE" \
  --generation-batch-size "$GENERATION_BATCH_SIZE" \
  --generation-samples "$GENERATION_SAMPLES" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --device cuda \
  --hf-cache-dir "$HF_HOME" \
  --output-dir "$RUN_DIR" \
  > "$RUN_DIR/benchmark.out" \
  2> "$RUN_DIR/time.txt"
