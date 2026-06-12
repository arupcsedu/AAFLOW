#!/bin/bash
#SBATCH --job-name=aaflow-hf-weak
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-hf-weak-%j.out
#SBATCH --error=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-hf-weak-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/djy8hg/workdir/AAFLOW}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python}"
HF_HOME="${HF_HOME:-/scratch/djy8hg/huggingface}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/benchmark/slurm_runs_hf_weak_scaling}"
MODEL="${MODEL:?MODEL must be llama3-8b, mistral-7b, or a Hugging Face model ID}"
GPUS="${GPUS:-${SLURM_NTASKS:-1}}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${MODEL}_${GPUS}gpu}"

CHUNKS_PER_GPU="${CHUNKS_PER_GPU:-128}"
FILES_PER_GPU="${FILES_PER_GPU:-16}"
CHUNK_CHARS="${CHUNK_CHARS:-900}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-4}"
UPSERT_BATCH_SIZE="${UPSERT_BATCH_SIZE:-32}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
AAFLOW_EMBED_BATCH_SIZE="${AAFLOW_EMBED_BATCH_SIZE:-32}"
AAFLOW_GENERATION_BATCH_SIZE="${AAFLOW_GENERATION_BATCH_SIZE:-8}"
AAFLOW_EMBED_BACKEND="${AAFLOW_EMBED_BACKEND:-eager}"
AAFLOW_STRATEGY="${AAFLOW_STRATEGY:-default}"
AAFLOW_COMPILE_MODE="${AAFLOW_COMPILE_MODE:-default}"
AAFLOW_DISABLE_FAISS_OVERLAP="${AAFLOW_DISABLE_FAISS_OVERLAP:-0}"
AAFLOW_TOKEN_BUDGET="${AAFLOW_TOKEN_BUDGET:-4096}"
AAFLOW_REDUCED_EMBED_TOKENS="${AAFLOW_REDUCED_EMBED_TOKENS:-96}"
AAFLOW_BUCKET_SIZES="${AAFLOW_BUCKET_SIZES:-32:64:96:128}"
GENERATION_SAMPLES_PER_GPU="${GENERATION_SAMPLES_PER_GPU:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-default}"
CORPUS_ROOT="${CORPUS_ROOT:-}"
AAFLOW_TUNING_ONLY="${AAFLOW_TUNING_ONLY:-0}"
AAFLOW_OPTIMIZED_ONLY="${AAFLOW_OPTIMIZED_ONLY:-0}"
DISABLE_BSP="${DISABLE_BSP:-0}"

if [[ "$ATTN_IMPLEMENTATION" == "flash_attention_2" ]]; then
  module load gcc/14.2.0 cuda/13.0.2
  if [[ "$PYTHON_BIN" == "/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python" ]]; then
    PYTHON_BIN="/scratch/djy8hg/env/drc_rag_benchmarks_flashattn/bin/python"
  fi
fi

if (( SLURM_NTASKS != GPUS )); then
  echo "SLURM_NTASKS=$SLURM_NTASKS must equal GPUS=$GPUS" >&2
  exit 2
fi

CORPUS_ARGS=()
if [[ -n "$CORPUS_ROOT" ]]; then
  CORPUS_ARGS=(--corpus-root "$CORPUS_ROOT")
fi
TUNING_ARGS=()
if [[ "$AAFLOW_OPTIMIZED_ONLY" == "1" ]]; then
  TUNING_ARGS=(--aaflow-optimized-only)
elif [[ "$AAFLOW_TUNING_ONLY" == "1" ]]; then
  TUNING_ARGS=(--aaflow-tuning-only)
fi
BSP_ARGS=()
if [[ "$DISABLE_BSP" == "1" ]]; then
  BSP_ARGS=(--disable-bsp)
fi
AAFLOW_OVERLAP_ARGS=()
if [[ "$AAFLOW_DISABLE_FAISS_OVERLAP" == "1" ]]; then
  AAFLOW_OVERLAP_ARGS=(--aaflow-disable-faiss-overlap)
fi

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
  echo "MODEL=$MODEL"
  echo "GPUS=$GPUS"
  echo "CHUNKS_PER_GPU=$CHUNKS_PER_GPU"
  echo "TOTAL_CHUNKS=$((GPUS * CHUNKS_PER_GPU))"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "HF_HOME=$HF_HOME"
  echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
  echo "AAFLOW_STRATEGY=$AAFLOW_STRATEGY"
  echo "AAFLOW_COMPILE_MODE=$AAFLOW_COMPILE_MODE"
  echo "AAFLOW_EMBED_BACKEND=$AAFLOW_EMBED_BACKEND"
  echo "AAFLOW_DISABLE_FAISS_OVERLAP=$AAFLOW_DISABLE_FAISS_OVERLAP"
  echo "AAFLOW_TOKEN_BUDGET=$AAFLOW_TOKEN_BUDGET"
  echo "AAFLOW_REDUCED_EMBED_TOKENS=$AAFLOW_REDUCED_EMBED_TOKENS"
  echo "AAFLOW_BUCKET_SIZES=$AAFLOW_BUCKET_SIZES"
} > "$RUN_DIR/allocation.txt"

/usr/bin/time -v \
  srun \
    --nodes=1 \
    --ntasks="$GPUS" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=cores \
    "$PYTHON_BIN" "$PROJECT_ROOT/benchmark/distributed_hf_weak_scaling.py" \
      --model "$MODEL" \
      --chunks-per-gpu "$CHUNKS_PER_GPU" \
      --files-per-gpu "$FILES_PER_GPU" \
      --chunk-chars "$CHUNK_CHARS" \
      --embed-batch-size "$EMBED_BATCH_SIZE" \
      --upsert-batch-size "$UPSERT_BATCH_SIZE" \
      --generation-batch-size "$GENERATION_BATCH_SIZE" \
      --aaflow-embed-batch-size "$AAFLOW_EMBED_BATCH_SIZE" \
      --aaflow-generation-batch-size "$AAFLOW_GENERATION_BATCH_SIZE" \
      --aaflow-embed-backend "$AAFLOW_EMBED_BACKEND" \
      --aaflow-strategy "$AAFLOW_STRATEGY" \
      --aaflow-compile-mode "$AAFLOW_COMPILE_MODE" \
      --aaflow-token-budget "$AAFLOW_TOKEN_BUDGET" \
      --aaflow-reduced-embed-tokens "$AAFLOW_REDUCED_EMBED_TOKENS" \
      --aaflow-bucket-sizes "$AAFLOW_BUCKET_SIZES" \
      --generation-samples-per-gpu "$GENERATION_SAMPLES_PER_GPU" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --dtype "$DTYPE" \
      --attn-implementation "$ATTN_IMPLEMENTATION" \
      --hf-cache-dir "$HF_HOME" \
      --run-dir "$RUN_DIR" \
      "${CORPUS_ARGS[@]}" \
      "${TUNING_ARGS[@]}" \
      "${BSP_ARGS[@]}" \
      "${AAFLOW_OVERLAP_ARGS[@]}" \
    > "$RUN_DIR/benchmark.out" \
    2> "$RUN_DIR/time.txt"
