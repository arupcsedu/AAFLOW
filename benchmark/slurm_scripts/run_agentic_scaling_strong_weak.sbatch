#!/bin/bash
#SBATCH --job-name=ragbench-agentic-scale
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
SCALING_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/agentic_scaling_runner.py"
DISTRIBUTED_NO_RAY_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/distributed_agentic_scaling.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/drc_rag/benchmark/slurm_runs_agentic_scaling}"
PROFILE="${PROFILE:-strong_no_ray}"
PHYSICAL_WORKERS="${PHYSICAL_WORKERS:?PHYSICAL_WORKERS must be set}"
CORES_PER_NODE="${CORES_PER_NODE:-40}"
NO_RAY_TASKS_PER_NODE="${NO_RAY_TASKS_PER_NODE:-}"
RUN_TAG="${RUN_TAG:-${PROFILE}_${PHYSICAL_WORKERS}w}"
RUN_DIR="${RUN_DIR:-$OUTROOT/${SLURM_JOB_ID}/${RUN_TAG}}"

BASE_NODES="${BASE_NODES:-}"
BASE_FILES="${BASE_FILES:-}"
BASE_NODES_PER_WORKER="${BASE_NODES_PER_WORKER:-128}"
NODE_CHARS="${NODE_CHARS:-900}"
CHUNKS_PER_FILE="${CHUNKS_PER_FILE:-}"
DIM="${DIM:-128}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-local-hash}"
SET5_SINK_BACKEND="${SET5_SINK_BACKEND:-faiss}"
SHARED_CORPUS_ROOT="${SHARED_CORPUS_ROOT:-/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache}"
LOCAL_CORPUS_STAGE="${LOCAL_CORPUS_STAGE:-0}"
LOCAL_CORPUS_ROOT="${LOCAL_CORPUS_ROOT:-/tmp/drc_rag_scaling_corpus_cache}"
RAY_INPUT_FORMAT="${RAY_INPUT_FORMAT:-raw}"

mkdir -p "$RUN_DIR"
allocated_nodes="${SLURM_JOB_NUM_NODES:-1}"
allocated_physical_cpus=$(( allocated_nodes * CORES_PER_NODE ))

no_ray_tasks_per_node="$CORES_PER_NODE"
if [[ -n "$NO_RAY_TASKS_PER_NODE" ]]; then
  no_ray_tasks_per_node="$NO_RAY_TASKS_PER_NODE"
fi

if [[ "$PROFILE" == "strong_no_ray" || "$PROFILE" == "weak_no_ray" ]]; then
  if [[ -z "$NO_RAY_TASKS_PER_NODE" ]]; then
    if (( PHYSICAL_WORKERS == 256 )); then
      no_ray_tasks_per_node=20
    else
      no_ray_tasks_per_node="$CORES_PER_NODE"
    fi
  fi
fi

required_nodes_for_tasks=$(( (PHYSICAL_WORKERS + no_ray_tasks_per_node - 1) / no_ray_tasks_per_node ))
required_nodes_for_cores=$(( (PHYSICAL_WORKERS + CORES_PER_NODE - 1) / CORES_PER_NODE ))
required_nodes="$required_nodes_for_cores"
if [[ "$PROFILE" == "strong_no_ray" || "$PROFILE" == "weak_no_ray" ]]; then
  required_nodes="$required_nodes_for_tasks"
fi

if (( allocated_nodes < required_nodes )); then
  echo "Allocated nodes ($allocated_nodes) are insufficient for PHYSICAL_WORKERS=$PHYSICAL_WORKERS with CORES_PER_NODE=$CORES_PER_NODE and tasks_per_node=${no_ray_tasks_per_node}" >&2
  exit 2
fi

scaling_profile="$PROFILE"
scaling_mode="agentic"
case "$PROFILE" in
  strong|weak)
    scaling_mode="agentic"
    ;;
  strong_no_ray)
    scaling_profile="strong"
    scaling_mode="agentic"
    ;;
  weak_no_ray)
    scaling_profile="weak"
    scaling_mode="agentic"
    ;;
  strong_ray_only)
    scaling_profile="strong"
    scaling_mode="ray"
    ;;
  weak_ray_only)
    scaling_profile="weak"
    scaling_mode="ray"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE" >&2
    exit 1
    ;;
esac

if [[ -z "$BASE_NODES" ]]; then
  if [[ "$scaling_mode" == "ray" ]]; then
    BASE_NODES=1000000
  else
    BASE_NODES=100000
  fi
fi
if [[ -z "$BASE_FILES" ]]; then
  if [[ "$scaling_mode" == "ray" ]]; then
    BASE_FILES=1000
  else
    BASE_FILES=100
  fi
fi
if [[ -z "$CHUNKS_PER_FILE" ]]; then
  if [[ "$scaling_mode" == "ray" ]]; then
    CHUNKS_PER_FILE=1000
  else
    CHUNKS_PER_FILE=1000
  fi
fi

if [[ "$LOCAL_CORPUS_STAGE" == "1" && "$scaling_mode" == "ray" ]]; then
  corpus_key="nodes_${BASE_NODES}__files_${BASE_FILES}__chars_${NODE_CHARS}__seed_7"
  if [[ "$RAY_INPUT_FORMAT" == "prechunked" ]]; then
    corpus_key="${corpus_key}__prechunked_lines"
  elif [[ "$RAY_INPUT_FORMAT" == "preembedded" ]]; then
    corpus_key="${corpus_key}__preembedded_npz_dim_${DIM}"
  fi
  source_corpus_dir="${SHARED_CORPUS_ROOT}/${corpus_key}"
  staged_corpus_dir="${LOCAL_CORPUS_ROOT}/${corpus_key}"
  stage_t0=$(date +%s.%N)
  srun --nodes "$allocated_nodes" --ntasks "$allocated_nodes" --ntasks-per-node 1 --overlap --cpu-bind=none \
    bash -lc 'set -euo pipefail; src="$1"; dst="$2"; mkdir -p "$(dirname "$dst")"; rm -rf "$dst"; mkdir -p "$dst"; cp -a "$src"/. "$dst"/' \
    _ "$source_corpus_dir" "$staged_corpus_dir"
  stage_t1=$(date +%s.%N)
  export SHARED_CORPUS_ROOT="$LOCAL_CORPUS_ROOT"
  stage_dt=$(awk "BEGIN {printf \"%.3f\", $stage_t1 - $stage_t0}")
  printf 'LocalCorpusStage(s)=%s\nLocalCorpusRoot=%s\n' \
    "$stage_dt" \
    "$LOCAL_CORPUS_ROOT" | tee "$RUN_DIR/local_corpus_stage.txt"
fi

stage_cpu_cap="$CORES_PER_NODE"
if [[ "$scaling_mode" == "ray" ]]; then
  stage_cpu_cap="$allocated_physical_cpus"
fi

printf 'PROFILE=%s\nPHYSICAL_WORKERS=%s\nALLOCATED_NODES=%s\nCORES_PER_NODE=%s\nNO_RAY_TASKS_PER_NODE=%s\nALLOCATED_PHYSICAL_CPUS=%s\nSTAGE_CPU_CAP=%s\n' \
  "$PROFILE" "$PHYSICAL_WORKERS" "$allocated_nodes" "$CORES_PER_NODE" "$no_ray_tasks_per_node" "$allocated_physical_cpus" "$stage_cpu_cap" \
  | tee "$RUN_DIR/allocation.txt"

if [[ "$PROFILE" == "strong_no_ray" || "$PROFILE" == "weak_no_ray" ]]; then
  /usr/bin/time -v \
    srun --nodes "$allocated_nodes" --ntasks "$PHYSICAL_WORKERS" --ntasks-per-node "$no_ray_tasks_per_node" --cpus-per-task 1 --cpu-bind=none \
      "$PYTHON_BIN" "$DISTRIBUTED_NO_RAY_SCRIPT" \
        --profile "$scaling_profile" \
        --physical-workers "$PHYSICAL_WORKERS" \
        --stage-cpu-cap 1 \
        --base-nodes "$BASE_NODES" \
        --base-files "$BASE_FILES" \
        --base-nodes-per-worker "$BASE_NODES_PER_WORKER" \
        --node-chars "$NODE_CHARS" \
        --chunks-per-file "$CHUNKS_PER_FILE" \
        --dim "$DIM" \
        --embedder-backend "$EMBEDDER_BACKEND" \
        --set5-sink-backend "$SET5_SINK_BACKEND" \
        --shared-corpus-root "$SHARED_CORPUS_ROOT" \
        --run-dir "$RUN_DIR" \
      >"$RUN_DIR/benchmark.out" 2>"$RUN_DIR/time.txt"
else
  cmd=(
    "$PYTHON_BIN" "$SCALING_SCRIPT"
    --profile "$scaling_profile"
    --mode "$scaling_mode"
    --physical-workers "$PHYSICAL_WORKERS"
    --async-workers "$PHYSICAL_WORKERS"
    --stage-cpu-cap "$stage_cpu_cap"
    --base-nodes "$BASE_NODES"
    --base-files "$BASE_FILES"
    --base-nodes-per-worker "$BASE_NODES_PER_WORKER"
    --node-chars "$NODE_CHARS"
    --chunks-per-file "$CHUNKS_PER_FILE"
    --dim "$DIM"
    --embedder-backend "$EMBEDDER_BACKEND"
    --set5-sink-backend "$SET5_SINK_BACKEND"
    --python-bin "$PYTHON_BIN"
    --run-dir "$RUN_DIR"
    --shared-corpus-root "$SHARED_CORPUS_ROOT"
    --ray-input-format "$RAY_INPUT_FORMAT"
  )

  if [[ "$scaling_mode" == "ray" ]]; then
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
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID}}"
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

    # Hard-clean stale Ray state on every allocated node before cluster startup.
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

    if ! env RAY_ADDRESS="$head_addr" EXPECTED_RAY_CPUS="$allocated_physical_cpus" "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, include_dashboard=False)
resources = ray.cluster_resources()
cpus = int(resources.get("CPU", 0) or 0)
expected = int(os.environ["EXPECTED_RAY_CPUS"])
if cpus < expected:
    raise SystemExit(f"Ray cluster readiness failed: only {cpus} CPUs visible, expected at least {expected}")
print(cpus)
PY
    then
      echo "Ray cluster readiness check failed for $head_addr" >&2
      exit 1
    fi

    /usr/bin/time -v env RAY_ADDRESS="$head_addr" "${cmd[@]}" >"$RUN_DIR/benchmark.out" 2>"$RUN_DIR/time.txt"
  else
    /usr/bin/time -v "${cmd[@]}" >"$RUN_DIR/benchmark.out" 2>"$RUN_DIR/time.txt"
  fi
fi
