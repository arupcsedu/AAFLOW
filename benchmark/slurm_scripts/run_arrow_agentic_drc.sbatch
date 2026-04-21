#!/bin/bash
#SBATCH --job-name=arrow-agentic-drc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

PROJECT_ROOT="/project/bi_dsc_community"
BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/benchmark_arrow_configs.py"
DIST_BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/distributed_arrow_agentic_drc.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/drc_rag/benchmark/slurm_runs_arrow_agentic_drc}"
SCENARIO="${SCENARIO:-both}"           # no_ray | ray | both
RUN_TAG="${RUN_TAG:-${SCENARIO}}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${RUN_TAG}}"

DATA_DIR="${DATA_DIR:-}"
PERSIST_DIR="${PERSIST_DIR:-}"
NODES_COUNT="${NODES_COUNT:-4096}"
FILES="${FILES:-512}"
NODE_CHARS="${NODE_CHARS:-900}"
CHUNKS_PER_FILE="${CHUNKS_PER_FILE:-0}"
SEED="${SEED:-7}"
DIM="${DIM:-128}"
WORKERS="${WORKERS:-16}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
UPSERT_BATCH_SIZE="${UPSERT_BATCH_SIZE:-512}"
SINK_BACKEND="${SINK_BACKEND:-faiss}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-local-hash}"  # local-hash | fake
REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-20}"
PER_ITEM_MS="${PER_ITEM_MS:-0.5}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-0}"
RAY_OBJECT_STORE_MEMORY_MB="${RAY_OBJECT_STORE_MEMORY_MB:-0}"
CORES_PER_NODE="${CORES_PER_NODE:-40}"

mkdir -p "$RUN_DIR"
allocated_nodes="${SLURM_JOB_NUM_NODES:-1}"
allocated_physical_cpus=$(( allocated_nodes * CORES_PER_NODE ))

case "$SCENARIO" in
  no_ray|ray|both)
    ;;
  *)
    echo "Unsupported SCENARIO=$SCENARIO; expected no_ray, ray, or both" >&2
    exit 2
    ;;
esac

common_cmd=(
  "$PYTHON_BIN" "$BENCH_SCRIPT"
  --nodes "$NODES_COUNT"
  --files "$FILES"
  --node-chars "$NODE_CHARS"
  --chunks-per-file "$CHUNKS_PER_FILE"
  --seed "$SEED"
  --dim "$DIM"
  --workers "$WORKERS"
  --embed-batch-size "$EMBED_BATCH_SIZE"
  --upsert-batch-size "$UPSERT_BATCH_SIZE"
  --sink-backend "$SINK_BACKEND"
  --embedder-backend "$EMBEDDER_BACKEND"
  --request-overhead-ms "$REQUEST_OVERHEAD_MS"
  --per-item-ms "$PER_ITEM_MS"
)

if [[ -n "$DATA_DIR" ]]; then
  common_cmd+=(--data-dir "$DATA_DIR")
fi
if [[ -n "$PERSIST_DIR" ]]; then
  common_cmd+=(--persist-dir "$PERSIST_DIR")
fi

printf 'SCENARIO=%s\nALLOCATED_NODES=%s\nCORES_PER_NODE=%s\nALLOCATED_PHYSICAL_CPUS=%s\n' \
  "$SCENARIO" "$allocated_nodes" "$CORES_PER_NODE" "$allocated_physical_cpus" \
  | tee "$RUN_DIR/allocation.txt"

run_no_ray() {
  cmd=(
    "$PYTHON_BIN" "$DIST_BENCH_SCRIPT"
    --profile strong
    --physical-workers "$WORKERS"
    --base-nodes "$NODES_COUNT"
    --base-files "$FILES"
    --node-chars "$NODE_CHARS"
    --chunks-per-file "$CHUNKS_PER_FILE"
    --seed "$SEED"
    --dim "$DIM"
    --embed-batch-size "$EMBED_BATCH_SIZE"
    --upsert-batch-size "$UPSERT_BATCH_SIZE"
    --embedder-backend "$EMBEDDER_BACKEND"
    --sink-backend "$SINK_BACKEND"
    --request-overhead-ms "$REQUEST_OVERHEAD_MS"
    --per-item-ms "$PER_ITEM_MS"
    --shared-corpus-root "${SHARED_CORPUS_ROOT:-/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache}"
    --run-dir "$RUN_DIR"
    --local-arrow-workers 1
  )
  if [[ -n "$DATA_DIR" ]]; then
    cmd+=(--data-dir "$DATA_DIR")
  fi
  if [[ -n "$PERSIST_DIR" ]]; then
    cmd+=(--persist-dir "$PERSIST_DIR")
  fi
  /usr/bin/time -v     srun --nodes "$allocated_nodes" --ntasks "$WORKERS" --cpu-bind=none       "${cmd[@]}"       >"$RUN_DIR/arrow_no_ray.out" 2>"$RUN_DIR/arrow_no_ray.time.txt"
}

run_ray() {
  mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
  if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "No hosts resolved from SLURM_NODELIST=$SLURM_NODELIST" >&2
    exit 1
  fi
  head_node="${HOSTS[0]}"
  worker_nodes=()
  if (( ${#HOSTS[@]} > 1 )); then
    worker_nodes=("${HOSTS[@]:1}")
  fi
  ray_port="${RAY_PORT:-6379}"
  head_addr="${head_node}:${ray_port}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_arrow_${SLURM_JOB_ID}}"
  mkdir -p "$RAY_TMPDIR"

  cleanup_ray() {
    set +e
    if [[ -n "${worker_srun_pid:-}" ]]; then
      kill "$worker_srun_pid" 2>/dev/null || true
      wait "$worker_srun_pid" 2>/dev/null || true
    fi
    if [[ -n "${head_srun_pid:-}" ]]; then
      kill "$head_srun_pid" 2>/dev/null || true
      wait "$head_srun_pid" 2>/dev/null || true
    fi
    srun --nodes "$allocated_nodes" --ntasks "$allocated_nodes" --ntasks-per-node 1 --overlap --cpu-bind=none \
      "$PYTHON_BIN" -c 'from ray.scripts.scripts import main; main()' stop --force >/dev/null 2>&1 || true
  }
  trap cleanup_ray EXIT

  srun --nodes "$allocated_nodes" --ntasks "$allocated_nodes" --ntasks-per-node 1 --overlap --cpu-bind=none \
    bash -lc 'set -euo pipefail; "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' stop --force >/dev/null 2>&1 || true; rm -rf "$2"; mkdir -p "$2"' \
    _ "$PYTHON_BIN" "$RAY_TMPDIR" >/dev/null 2>&1

  srun --nodes 1 --ntasks 1 --ntasks-per-node 1 --nodelist "$head_node" --overlap --cpu-bind=none \
    bash -lc 'set -euo pipefail; "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' stop --force >/dev/null 2>&1 || true; exec "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' start --head --node-ip-address="$2" --port="$3" --num-cpus="$4" --include-dashboard=false --block' \
    _ "$PYTHON_BIN" "$head_node" "$ray_port" "$CORES_PER_NODE" >"$RUN_DIR/ray-head.log" 2>&1 &
  head_srun_pid=$!
  sleep 20
  if ! kill -0 "$head_srun_pid" 2>/dev/null; then
    echo "Ray head failed to start. See $RUN_DIR/ray-head.log" >&2
    exit 1
  fi

  if (( ${#worker_nodes[@]} > 0 )); then
    worker_nodelist=$(IFS=,; echo "${worker_nodes[*]}")
    srun --nodes "$((allocated_nodes - 1))" --ntasks "$((allocated_nodes - 1))" --ntasks-per-node 1 --nodelist "$worker_nodelist" --overlap --cpu-bind=none \
      bash -lc 'set -euo pipefail; "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' stop --force >/dev/null 2>&1 || true; exec "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' start --address="$2" --num-cpus="$3" --block' \
      _ "$PYTHON_BIN" "$head_addr" "$CORES_PER_NODE" >"$RUN_DIR/ray-workers.log" 2>&1 &
    worker_srun_pid=$!
    sleep 15
    if ! kill -0 "$worker_srun_pid" 2>/dev/null; then
      echo "Ray workers failed to start. See $RUN_DIR/ray-workers.log" >&2
      exit 1
    fi
  fi

  effective_ray_cpus="$RAY_NUM_CPUS"
  if [[ "$effective_ray_cpus" == "0" ]]; then
    effective_ray_cpus="$allocated_physical_cpus"
  fi

  /usr/bin/time -v \
    srun --nodes 1 --ntasks 1 --ntasks-per-node 1 --nodelist "$head_node" --cpu-bind=none \
      "${common_cmd[@]}" --run-ray \
      --ray-num-cpus "$effective_ray_cpus" \
      --ray-object-store-memory-mb "$RAY_OBJECT_STORE_MEMORY_MB" \
      >"$RUN_DIR/arrow_ray.out" 2>"$RUN_DIR/arrow_ray.time.txt"

  cleanup_ray
  trap - EXIT
}

if [[ "$SCENARIO" == "no_ray" || "$SCENARIO" == "both" ]]; then
  run_no_ray
fi

if [[ "$SCENARIO" == "ray" || "$SCENARIO" == "both" ]]; then
  run_ray
fi

printf 'Outputs written to %s\n' "$RUN_DIR"
