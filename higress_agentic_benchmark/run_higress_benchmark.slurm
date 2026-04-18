#!/bin/bash
#SBATCH --job-name=higress-rag-bench
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --output=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/slurm-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/slurm-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_bench_env/bin/python}
BASE_DIR=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark
OUT_DIR=${OUT_DIR:-$BASE_DIR/slurm_runs/${SLURM_JOB_ID}}
mkdir -p "$OUT_DIR"

CORES_PER_NODE=${CORES_PER_NODE:-${SLURM_CPUS_ON_NODE:-40}}
PHYSICAL_WORKERS=${PHYSICAL_WORKERS:-64}
BENCHMARK_MODE=${BENCHMARK_MODE:-fair_parallelism_plus_overlap}
VECTOR_BACKEND=${VECTOR_BACKEND:-faiss}
DATA_DIR=${DATA_DIR:-/project/bi_dsc_community/drc_rag/data/wikitext2_train}
FILE_GLOB=${FILE_GLOB:-*.txt}
MAX_CHARS=${MAX_CHARS:-900}
OVERLAP_CHARS=${OVERLAP_CHARS:-120}
QUERY_COUNT=${QUERY_COUNT:-8}
TOP_K=${TOP_K:-5}
REPEAT=${REPEAT:-3}
ENGINE_FILTER=${ENGINE_FILTER:-}
MOCK_BASE_LATENCY_MS=${MOCK_BASE_LATENCY_MS:-12}
MOCK_MS_PER_TOKEN=${MOCK_MS_PER_TOKEN:-0.2}
HF_MAX_NEW_TOKENS=${HF_MAX_NEW_TOKENS:-16}
LLM_BACKEND=${LLM_BACKEND:-hf}
HF_MODEL=${HF_MODEL:-distilgpt2}
NON_AGENTIC_DISPATCH_OVERHEAD_MS=${NON_AGENTIC_DISPATCH_OVERHEAD_MS:-20}
DISABLE_STM=${DISABLE_STM:-1}
DISABLE_LTM=${DISABLE_LTM:-1}
DISABLE_EM=${DISABLE_EM:-1}

required_nodes=$(( (PHYSICAL_WORKERS + CORES_PER_NODE - 1) / CORES_PER_NODE ))
allocated_nodes=${SLURM_JOB_NUM_NODES:-1}
if (( allocated_nodes < required_nodes )); then
  echo "Allocated nodes ($allocated_nodes) are insufficient for PHYSICAL_WORKERS=$PHYSICAL_WORKERS with CORES_PER_NODE=$CORES_PER_NODE" >&2
  exit 2
fi

cd /project/bi_dsc_community/drc_rag

cmd=(
  "$PYTHON_BIN" -m higress_agentic_benchmark.distributed_higress_benchmark
  --benchmark-mode "$BENCHMARK_MODE"
  --vector-backend "$VECTOR_BACKEND"
  --physical-workers "$PHYSICAL_WORKERS"
  --non-agentic-dispatch-overhead-ms "$NON_AGENTIC_DISPATCH_OVERHEAD_MS"
  --data-dir "$DATA_DIR"
  --file-glob "$FILE_GLOB"
  --max-chars "$MAX_CHARS"
  --overlap-chars "$OVERLAP_CHARS"
  --query-count "$QUERY_COUNT"
  --top-k "$TOP_K"
  --llm-backend "$LLM_BACKEND"
  --hf-model "$HF_MODEL"
  --hf-max-new-tokens "$HF_MAX_NEW_TOKENS"
  --mock-base-latency-ms "$MOCK_BASE_LATENCY_MS"
  --mock-ms-per-token "$MOCK_MS_PER_TOKEN"
  --repeat "$REPEAT"
  --output-dir "$OUT_DIR"
)

if [[ "$DISABLE_STM" == "1" ]]; then
  cmd+=(--disable-stm)
fi
if [[ "$DISABLE_LTM" == "1" ]]; then
  cmd+=(--disable-ltm)
fi
if [[ "$DISABLE_EM" == "1" ]]; then
  cmd+=(--disable-em)
fi
if [[ -n "$ENGINE_FILTER" ]]; then
  cmd+=(--engine-filter "$ENGINE_FILTER")
fi

/usr/bin/time -v \
  srun --nodes "$allocated_nodes" --ntasks "$PHYSICAL_WORKERS" --ntasks-per-node "$CORES_PER_NODE" --cpus-per-task 1 --cpu-bind=none \
  "${cmd[@]}" >"$OUT_DIR/benchmark.out" 2>"$OUT_DIR/time.txt"

printf 'Outputs written to %s\n' "$OUT_DIR"
