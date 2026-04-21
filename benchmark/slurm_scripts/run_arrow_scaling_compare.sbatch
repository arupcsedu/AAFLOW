#!/bin/bash
#SBATCH --job-name=arrow-scaling-compare
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
DIST_BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/distributed_arrow_scaling.py"
ARROW_BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/benchmark_arrow_configs.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/drc_rag/benchmark/slurm_runs_arrow_scaling}"
PROFILE="${PROFILE:-strong}"          # strong | weak
MODE="${MODE:-both}"                  # no_ray_only | ray_only | both
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${PROFILE}}"

PHYSICAL_WORKERS="${PHYSICAL_WORKERS:-64}"
CORES_PER_NODE="${CORES_PER_NODE:-40}"
BASE_NODES="${BASE_NODES:-10000000}"
BASE_FILES="${BASE_FILES:-4096}"
BASE_NODES_PER_WORKER="${BASE_NODES_PER_WORKER:-95000}"
NODE_CHARS="${NODE_CHARS:-900}"
CHUNKS_PER_FILE="${CHUNKS_PER_FILE:-0}"
SEED="${SEED:-7}"
DIM="${DIM:-128}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
UPSERT_BATCH_SIZE="${UPSERT_BATCH_SIZE:-512}"
SINK_BACKEND="${SINK_BACKEND:-faiss}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-local-hash}"
REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-20}"
PER_ITEM_MS="${PER_ITEM_MS:-0.5}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-0}"
RAY_OBJECT_STORE_MEMORY_MB="${RAY_OBJECT_STORE_MEMORY_MB:-0}"
SHARED_CORPUS_ROOT="${SHARED_CORPUS_ROOT:-/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache}"
PERSIST_DIR="${PERSIST_DIR:-}"

mkdir -p "$RUN_DIR"
allocated_nodes="${SLURM_JOB_NUM_NODES:-1}"
allocated_physical_cpus=$(( allocated_nodes * CORES_PER_NODE ))

printf 'PROFILE=%s\nPHYSICAL_WORKERS=%s\nALLOCATED_NODES=%s\nCORES_PER_NODE=%s\n' \
  "$PROFILE" "$PHYSICAL_WORKERS" "$allocated_nodes" "$CORES_PER_NODE" \
  | tee "$RUN_DIR/allocation.txt"

case "$MODE" in
  no_ray_only|ray_only|both)
    ;;
  *)
    echo "Unsupported MODE=$MODE; expected no_ray_only, ray_only, or both" >&2
    exit 2
    ;;
esac

run_distributed_no_ray() {
  cmd=(
    "$PYTHON_BIN" "$DIST_BENCH_SCRIPT"
    --profile "$PROFILE"
    --physical-workers "$PHYSICAL_WORKERS"
    --base-nodes "$BASE_NODES"
    --base-files "$BASE_FILES"
    --base-nodes-per-worker "$BASE_NODES_PER_WORKER"
    --node-chars "$NODE_CHARS"
    --chunks-per-file "$CHUNKS_PER_FILE"
    --dim "$DIM"
    --embed-batch-size "$EMBED_BATCH_SIZE"
    --upsert-batch-size "$UPSERT_BATCH_SIZE"
    --embedder-backend "$EMBEDDER_BACKEND"
    --sink-backend "$SINK_BACKEND"
    --request-overhead-ms "$REQUEST_OVERHEAD_MS"
    --per-item-ms "$PER_ITEM_MS"
    --run-dir "$RUN_DIR"
    --shared-corpus-root "$SHARED_CORPUS_ROOT"
    --seed "$SEED"
    --local-arrow-workers 1
  )
  if [[ -n "$PERSIST_DIR" ]]; then
    cmd+=(--persist-dir "$PERSIST_DIR")
  fi
  /usr/bin/time -v \
    srun --nodes "$allocated_nodes" --ntasks "$PHYSICAL_WORKERS" --cpu-bind=none \
      "${cmd[@]}" \
      >"$RUN_DIR/no_ray.out" 2>"$RUN_DIR/no_ray.time.txt"
}

run_arrow_ray() {
  shared_corpus_dir="$SHARED_CORPUS_ROOT/nodes_${BASE_NODES}__files_${BASE_FILES}__chars_${NODE_CHARS}__seed_${SEED}"

  mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
  head_node="${HOSTS[0]}"
  worker_nodes=()
  if (( ${#HOSTS[@]} > 1 )); then
    worker_nodes=("${HOSTS[@]:1}")
  fi
  ray_port="${RAY_PORT:-6379}"
  head_addr="${head_node}:${ray_port}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_arrow_scaling_${SLURM_JOB_ID}}"
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

  if (( ${#worker_nodes[@]} > 0 )); then
    worker_nodelist=$(IFS=,; echo "${worker_nodes[*]}")
    srun --nodes "$((allocated_nodes - 1))" --ntasks "$((allocated_nodes - 1))" --ntasks-per-node 1 --nodelist "$worker_nodelist" --overlap --cpu-bind=none \
      bash -lc 'set -euo pipefail; "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' stop --force >/dev/null 2>&1 || true; exec "$1" -c '"'"'from ray.scripts.scripts import main; main()'"'"' start --address="$2" --num-cpus="$3" --block' \
      _ "$PYTHON_BIN" "$head_addr" "$CORES_PER_NODE" >"$RUN_DIR/ray-workers.log" 2>&1 &
    worker_srun_pid=$!
    sleep 15
  fi

  /usr/bin/time -v \
    srun --nodes 1 --ntasks 1 --ntasks-per-node 1 --nodelist "$head_node" --cpu-bind=none \
      "$PYTHON_BIN" "$ARROW_BENCH_SCRIPT" \
      --data-dir "$shared_corpus_dir" \
      --workers "$PHYSICAL_WORKERS" \
      --dim "$DIM" \
      --embed-batch-size "$EMBED_BATCH_SIZE" \
      --upsert-batch-size "$UPSERT_BATCH_SIZE" \
      --sink-backend "$SINK_BACKEND" \
      --embedder-backend "$EMBEDDER_BACKEND" \
      --request-overhead-ms "$REQUEST_OVERHEAD_MS" \
      --per-item-ms "$PER_ITEM_MS" \
      --run-ray \
      --run-ray-baseline \
      --summary-csv "$RUN_DIR/summary_ray.csv" \
      >"$RUN_DIR/ray.out" 2>"$RUN_DIR/ray.time.txt"

  cleanup_ray
  trap - EXIT
}

merge_summaries() {
  RUN_DIR_ENV="$RUN_DIR" "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR_ENV"])
rows = []
for name in ("summary_no_ray.csv", "summary_ray.csv"):
    path = run_dir / name
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(list(csv.DictReader(f)))
order = ["AsyncParallelOnly", "DaskScalableRAG", "HigressRAG", "RayDataScalableRAG", "AAFLOW", "AAFLOW+", "ArrowRayDataScalableRAG"]
rows.sort(key=lambda r: order.index(r["config"]) if r["config"] in order else len(order))
with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    if not rows:
        raise SystemExit(0)
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
PY
}

if [[ "$MODE" == "no_ray_only" || "$MODE" == "both" ]]; then
  run_distributed_no_ray
fi

if [[ "$MODE" == "ray_only" || "$MODE" == "both" ]]; then
  run_arrow_ray
fi

merge_summaries

printf 'Outputs written to %s\n' "$RUN_DIR"
