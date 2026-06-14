#!/bin/bash
#SBATCH --job-name=aaflow-higress-hf
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/djy8hg/workdir/AAFLOW/higress_agentic_benchmark/slurm-hf-higress-%j.out
#SBATCH --error=/scratch/djy8hg/workdir/AAFLOW/higress_agentic_benchmark/slurm-hf-higress-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/djy8hg/workdir/AAFLOW}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python}"
HF_HOME="${HF_HOME:-/scratch/djy8hg/huggingface}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/higress_agentic_benchmark/slurm_runs_hf_higress}"
MODEL="${MODEL:?MODEL must be llama3-8b, mistral-7b, or a Hugging Face model ID}"
GPUS="${GPUS:-${SLURM_NTASKS:-1}}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${MODEL}_${GPUS}gpu}"

CHUNKS_PER_GPU="${CHUNKS_PER_GPU:-16000}"
FILES_PER_GPU="${FILES_PER_GPU:-64}"
CHUNK_CHARS="${CHUNK_CHARS:-900}"
CORPUS_ROOT="${CORPUS_ROOT:-}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
AAFLOW_EMBED_BATCH_SIZE="${AAFLOW_EMBED_BATCH_SIZE:-128}"
AAFLOW_TOKENIZER_PREFETCH="${AAFLOW_TOKENIZER_PREFETCH:-1}"
AAFLOW_DEFER_VECTOR_TRANSFER="${AAFLOW_DEFER_VECTOR_TRANSFER:-0}"
AAFLOW_TF32="${AAFLOW_TF32:-0}"
AAFLOW_COMPILE_MODE="${AAFLOW_COMPILE_MODE:-default}"
AAFLOW_TOKEN_BUDGET="${AAFLOW_TOKEN_BUDGET:-0}"
AAFLOW_BUCKET_SIZES="${AAFLOW_BUCKET_SIZES:-64,128}"
UPSERT_BATCH_SIZE="${UPSERT_BATCH_SIZE:-32}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
GENERATION_SAMPLES_PER_GPU="${GENERATION_SAMPLES_PER_GPU:-64}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ENGINE_FILTER="${ENGINE_FILTER:-}"

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
ENGINE_ARGS=()
if [[ -n "$ENGINE_FILTER" ]]; then
  ENGINE_ARGS=(--engine-filter "$ENGINE_FILTER")
fi
AAFLOW_ARGS=(--aaflow-compile-mode "$AAFLOW_COMPILE_MODE")
if [[ "$AAFLOW_TOKENIZER_PREFETCH" == "1" || "$AAFLOW_TOKENIZER_PREFETCH" == "true" || "$AAFLOW_TOKENIZER_PREFETCH" == "TRUE" ]]; then
  AAFLOW_ARGS+=(--aaflow-tokenizer-prefetch)
fi
if [[ "$AAFLOW_DEFER_VECTOR_TRANSFER" == "1" || "$AAFLOW_DEFER_VECTOR_TRANSFER" == "true" || "$AAFLOW_DEFER_VECTOR_TRANSFER" == "TRUE" ]]; then
  AAFLOW_ARGS+=(--aaflow-defer-vector-transfer)
fi
if [[ "$AAFLOW_TF32" == "1" || "$AAFLOW_TF32" == "true" || "$AAFLOW_TF32" == "TRUE" ]]; then
  AAFLOW_ARGS+=(--aaflow-tf32)
fi
if [[ "$AAFLOW_TOKEN_BUDGET" != "0" ]]; then
  AAFLOW_ARGS+=(--aaflow-token-budget "$AAFLOW_TOKEN_BUDGET" --aaflow-bucket-sizes "$AAFLOW_BUCKET_SIZES")
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
  echo "ENGINE_FILTER=$ENGINE_FILTER"
  echo "AAFLOW_EMBED_BATCH_SIZE=$AAFLOW_EMBED_BATCH_SIZE"
  echo "AAFLOW_TOKENIZER_PREFETCH=$AAFLOW_TOKENIZER_PREFETCH"
  echo "AAFLOW_COMPILE_MODE=$AAFLOW_COMPILE_MODE"
  echo "CORPUS_ROOT=$CORPUS_ROOT"
} > "$RUN_DIR/allocation.txt"

/usr/bin/time -v \
  srun \
    --nodes=1 \
    --ntasks="$GPUS" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=cores \
    "$PYTHON_BIN" "$PROJECT_ROOT/higress_agentic_benchmark/distributed_hf_higress_benchmark.py" \
      --model "$MODEL" \
      --chunks-per-gpu "$CHUNKS_PER_GPU" \
      --files-per-gpu "$FILES_PER_GPU" \
      --chunk-chars "$CHUNK_CHARS" \
      --embed-batch-size "$EMBED_BATCH_SIZE" \
      --aaflow-embed-batch-size "$AAFLOW_EMBED_BATCH_SIZE" \
      --upsert-batch-size "$UPSERT_BATCH_SIZE" \
      --generation-batch-size "$GENERATION_BATCH_SIZE" \
      --generation-samples-per-gpu "$GENERATION_SAMPLES_PER_GPU" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --dtype "$DTYPE" \
      --attn-implementation "$ATTN_IMPLEMENTATION" \
      --hf-cache-dir "$HF_HOME" \
      --run-dir "$RUN_DIR" \
      "${CORPUS_ARGS[@]}" \
      "${ENGINE_ARGS[@]}" \
      "${AAFLOW_ARGS[@]}" \
    > "$RUN_DIR/benchmark.out" \
    2> "$RUN_DIR/time.txt"

printf 'Outputs written to %s\n' "$RUN_DIR"
