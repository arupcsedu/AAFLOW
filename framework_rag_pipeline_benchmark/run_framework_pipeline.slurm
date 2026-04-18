#!/bin/bash
#SBATCH --job-name=rag-fw-bench
#SBATCH --output=/project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark/slurm-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/scratch/djy8hg/env/drc_rag_bench_env/bin/python}
BASE_DIR=/project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
OUT_DIR=${OUT_DIR:-$BASE_DIR/slurm_runs/${SLURM_JOB_ID}}
mkdir -p "$OUT_DIR"
CORES_PER_NODE=${CORES_PER_NODE:-${SLURM_CPUS_ON_NODE:-16}}

NODES=${NODES:-3200}
FILES=${FILES:-200}
NODE_CHARS=${NODE_CHARS:-900}
GENERATION_SAMPLES=${GENERATION_SAMPLES:-}
GENERATION_OUTPUT_TOKENS=${GENERATION_OUTPUT_TOKENS:-64}
BENCHMARK_MODE=${BENCHMARK_MODE:-}
LOAD_WORKERS=${LOAD_WORKERS:-16}
TRANSFORM_WORKERS=${TRANSFORM_WORKERS:-16}
ASYNC_WORKERS=${ASYNC_WORKERS:-16}
PHYSICAL_WORKERS=${PHYSICAL_WORKERS:-$ASYNC_WORKERS}
EMBED_WORKERS=${EMBED_WORKERS:-}
UPSERT_WORKERS=${UPSERT_WORKERS:-}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-32}
UPSERT_BATCH_SIZE=${UPSERT_BATCH_SIZE:-}
VECTOR_BACKEND=${VECTOR_BACKEND:-faiss}
FAISS_PATH=${FAISS_PATH:-$OUT_DIR/faiss_store}
CHROMA_PATH=${CHROMA_PATH:-$OUT_DIR/chroma_store}
AGENTIC_UPSERT_COALESCE_TARGET=${AGENTIC_UPSERT_COALESCE_TARGET:-}
FRAMEWORK_FILTER=${FRAMEWORK_FILTER:-}
REPEAT=${REPEAT:-1}

if [[ "$VECTOR_BACKEND" == "chroma" ]]; then
  if [[ -z "$GENERATION_SAMPLES" ]]; then
    GENERATION_SAMPLES=200
  fi
  if [[ -z "$BENCHMARK_MODE" ]]; then
    BENCHMARK_MODE=fair_parallelism_plus_overlap
  fi
  if [[ -z "$UPSERT_WORKERS" ]]; then
    UPSERT_WORKERS=2
  fi
  if [[ -z "$UPSERT_BATCH_SIZE" ]]; then
    UPSERT_BATCH_SIZE=2048
  fi
  if [[ -z "$AGENTIC_UPSERT_COALESCE_TARGET" ]]; then
    AGENTIC_UPSERT_COALESCE_TARGET=4096
  fi
else
  if [[ -z "$GENERATION_SAMPLES" ]]; then
    GENERATION_SAMPLES=32
  fi
  if [[ -z "$BENCHMARK_MODE" ]]; then
    BENCHMARK_MODE=fair_parallelism_plus_overlap
  fi
  if [[ -z "$UPSERT_BATCH_SIZE" ]]; then
    UPSERT_BATCH_SIZE=256
  fi
  if [[ -z "$AGENTIC_UPSERT_COALESCE_TARGET" ]]; then
    AGENTIC_UPSERT_COALESCE_TARGET=256
  fi
fi

cd /project/bi_dsc_community/drc_rag

required_nodes=$(( (PHYSICAL_WORKERS + CORES_PER_NODE - 1) / CORES_PER_NODE ))
allocated_nodes=${SLURM_JOB_NUM_NODES:-1}
if (( allocated_nodes < required_nodes )); then
  echo "Allocated nodes ($allocated_nodes) are insufficient for PHYSICAL_WORKERS=$PHYSICAL_WORKERS with CORES_PER_NODE=$CORES_PER_NODE" >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" -m framework_rag_pipeline_benchmark.distributed_framework_benchmark
  --benchmark-mode "$BENCHMARK_MODE"
  --data-dir "$BASE_DIR/slurm_data_${SLURM_JOB_ID}"
  --nodes "$NODES"
  --files "$FILES"
  --node-chars "$NODE_CHARS"
  --generation-samples "$GENERATION_SAMPLES"
  --generation-output-tokens "$GENERATION_OUTPUT_TOKENS"
  --load-workers "$LOAD_WORKERS"
  --transform-workers "$TRANSFORM_WORKERS"
  --async-workers "$ASYNC_WORKERS"
  --physical-workers "$PHYSICAL_WORKERS"
  --vector-backend "$VECTOR_BACKEND"
  --faiss-path "$FAISS_PATH"
  --chroma-path "$CHROMA_PATH"
  --embed-batch-size "$EMBED_BATCH_SIZE"
  --upsert-batch-size "$UPSERT_BATCH_SIZE"
  --agentic-upsert-coalesce-target "$AGENTIC_UPSERT_COALESCE_TARGET"
  --output-dir "$OUT_DIR"
)

if [[ -n "$FRAMEWORK_FILTER" ]]; then
  cmd+=(--framework-filter "$FRAMEWORK_FILTER")
fi

if [[ -n "$REPEAT" ]]; then
  cmd+=(--repeat "$REPEAT")
fi

if [[ -n "$EMBED_WORKERS" ]]; then
  cmd+=(--embed-workers "$EMBED_WORKERS")
fi

if [[ -n "$UPSERT_WORKERS" ]]; then
  cmd+=(--upsert-workers "$UPSERT_WORKERS")
fi

/usr/bin/time -v \
  srun --nodes "$allocated_nodes" --ntasks "$PHYSICAL_WORKERS" --ntasks-per-node "$CORES_PER_NODE" --cpus-per-task 1 --cpu-bind=none \
  "${cmd[@]}" >"$OUT_DIR/benchmark.out" 2>"$OUT_DIR/time.txt"

"$PYTHON_BIN" -m framework_rag_pipeline_benchmark.plot_results \
  --summary-csv "$OUT_DIR/summary.csv" \
  --output-dir "$OUT_DIR/plots"

printf 'Outputs written to %s\n' "$OUT_DIR"
