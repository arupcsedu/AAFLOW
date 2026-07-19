#!/bin/bash
#SBATCH --job-name=aaflow-higress-q
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/djy8hg/workdir/AAFLOW/higress_agentic_benchmark/slurm-hf-higress-query-%j.out
#SBATCH --error=/scratch/djy8hg/workdir/AAFLOW/higress_agentic_benchmark/slurm-hf-higress-query-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/djy8hg/workdir/AAFLOW}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python}"
HF_HOME="${HF_HOME:-/scratch/djy8hg/huggingface}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/higress_agentic_benchmark/slurm_runs_hf_higress_query}"
MODEL="${MODEL:?MODEL must be llama3-8b, mistral-7b, or a Hugging Face model ID}"
GPUS="${GPUS:-${SLURM_NTASKS:-1}}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${MODEL}_${GPUS}gpu}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/higress_agentic_benchmark/sample_wikitext2_2000}"
FILE_GLOB="${FILE_GLOB:-*.txt}"
MAX_CHARS="${MAX_CHARS:-900}"
OVERLAP_CHARS="${OVERLAP_CHARS:-120}"
QUERY_COUNT="${QUERY_COUNT:-64}"
TOP_K="${TOP_K:-5}"
REPEAT="${REPEAT:-1}"
BENCHMARK_MODE="${BENCHMARK_MODE:-fair_parallelism_plus_overlap}"
VECTOR_BACKEND="${VECTOR_BACKEND:-faiss}"
PHYSICAL_WORKERS="${PHYSICAL_WORKERS:-${GPUS}}"
AAFLOW_PLUS_BATCH_SIZE="${AAFLOW_PLUS_BATCH_SIZE:-0}"
AAFLOW_PLUS_DENSE_CANDIDATES="${AAFLOW_PLUS_DENSE_CANDIDATES:-0}"
AAFLOW_PLUS_EXACT_VECTORIZED="${AAFLOW_PLUS_EXACT_VECTORIZED:-1}"
NON_AGENTIC_DISPATCH_OVERHEAD_MS="${NON_AGENTIC_DISPATCH_OVERHEAD_MS:-0}"
DISABLE_STM="${DISABLE_STM:-1}"
DISABLE_LTM="${DISABLE_LTM:-1}"
DISABLE_EM="${DISABLE_EM:-1}"
ENGINE_FILTER="${ENGINE_FILTER:-HigressRAG,AAFLOW+}"
SCENARIO_FILTER="${SCENARIO_FILTER:-}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

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

EXTRA_ARGS=()
if [[ "$DISABLE_STM" == "1" ]]; then EXTRA_ARGS+=(--disable-stm); fi
if [[ "$DISABLE_LTM" == "1" ]]; then EXTRA_ARGS+=(--disable-ltm); fi
if [[ "$DISABLE_EM" == "1" ]]; then EXTRA_ARGS+=(--disable-em); fi
if [[ -n "$ENGINE_FILTER" ]]; then EXTRA_ARGS+=(--engine-filter "$ENGINE_FILTER"); fi
if [[ -n "$SCENARIO_FILTER" ]]; then EXTRA_ARGS+=(--scenario-filter "$SCENARIO_FILTER"); fi
if [[ "$AAFLOW_PLUS_EXACT_VECTORIZED" != "1" ]]; then EXTRA_ARGS+=(--disable-aaflow-plus-exact-vectorized); fi

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
  echo "DATA_DIR=$DATA_DIR"
  echo "QUERY_COUNT=$QUERY_COUNT"
  echo "REPEAT=$REPEAT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "HF_HOME=$HF_HOME"
  echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
  echo "ENGINE_FILTER=$ENGINE_FILTER"
  echo "SCENARIO_FILTER=$SCENARIO_FILTER"
  echo "AAFLOW_PLUS_BATCH_SIZE=$AAFLOW_PLUS_BATCH_SIZE"
  echo "AAFLOW_PLUS_DENSE_CANDIDATES=$AAFLOW_PLUS_DENSE_CANDIDATES"
  echo "AAFLOW_PLUS_EXACT_VECTORIZED=$AAFLOW_PLUS_EXACT_VECTORIZED"
} > "$RUN_DIR/allocation.txt"

/usr/bin/time -v \
  srun \
    --nodes=1 \
    --ntasks="$GPUS" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=cores \
    "$PYTHON_BIN" "$PROJECT_ROOT/higress_agentic_benchmark/distributed_hf_higress_query_benchmark.py" \
      --model "$MODEL" \
      --data-dir "$DATA_DIR" \
      --file-glob "$FILE_GLOB" \
      --max-chars "$MAX_CHARS" \
      --overlap-chars "$OVERLAP_CHARS" \
      --query-count "$QUERY_COUNT" \
      --top-k "$TOP_K" \
      --repeat "$REPEAT" \
      --benchmark-mode "$BENCHMARK_MODE" \
      --vector-backend "$VECTOR_BACKEND" \
      --physical-workers "$PHYSICAL_WORKERS" \
      --aaflow-plus-batch-size "$AAFLOW_PLUS_BATCH_SIZE" \
      --aaflow-plus-dense-candidates "$AAFLOW_PLUS_DENSE_CANDIDATES" \
      --non-agentic-dispatch-overhead-ms "$NON_AGENTIC_DISPATCH_OVERHEAD_MS" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --dtype "$DTYPE" \
      --attn-implementation "$ATTN_IMPLEMENTATION" \
      --hf-cache-dir "$HF_HOME" \
      --output-dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}" \
    > "$RUN_DIR/benchmark.out" \
    2> "$RUN_DIR/time.txt"

printf 'Outputs written to %s\n' "$RUN_DIR"
