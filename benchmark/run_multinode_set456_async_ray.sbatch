#!/bin/bash
#SBATCH --job-name=ragbench-set456
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel     # change to your cluster's partition
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

PROJECT_ROOT="/project/bi_dsc_community"
BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/benchmark_configs_1_to_5.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/drc_rag/benchmark/slurm_runs_set456/${SLURM_JOB_ID}}"
PROFILE="${PROFILE:-default}"

# Dataset/workload knobs
NODES="${NODES:-4096}"
FILES="${FILES:-200}"
NODE_CHARS="${NODE_CHARS:-900}"
UPSERT_CAP="${UPSERT_CAP:--1}"
SET5_EMBED_BATCH="${SET5_EMBED_BATCH:-64}"
SET5_UPSERT_BATCH="${SET5_UPSERT_BATCH:-256}"
REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-120}"
PER_ITEM_MS="${PER_ITEM_MS:-2.0}"
BATCH_SCALE_BASELINE="${BATCH_SCALE_BASELINE:-64}"
RAY_OBJECT_STORE_MB="${RAY_OBJECT_STORE_MB:-2048}"
NO_SCALE_SET5_BATCHES="${NO_SCALE_SET5_BATCHES:-1}"
SET45_UPSERT_SHARDS="${SET45_UPSERT_SHARDS:--1}"
STRICT_STAGE_SCALING="${STRICT_STAGE_SCALING:-1}"
RUN_RAY_SET6="${RUN_RAY_SET6:-1}"
RUN_DASK_SET7="${RUN_DASK_SET7:-1}"
RUN_BSP_SET8="${RUN_BSP_SET8:-1}"

# Comma-separated async-workers:ray-num-cpus pairs.
# async-workers is passed once per run and shared by Set4/Set5/Set6/Set7/Set8.
WORKER_CPU_PAIRS="${WORKER_CPU_PAIRS:-4:4,8:8,12:12,16:16}"

# Profile presets. Slurm resources such as -N and --cpus-per-task must still be
# supplied at submit time because SBATCH directives are evaluated before the
# script body runs. Examples:
#   PROFILE=bsp_large WORKER_CPU_PAIRS='128:128,256:256,512:512' \
#   sbatch -N 4 --cpus-per-task=512 --mem=0 \
#     /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
#
#   PROFILE=no_bsp_large WORKER_CPU_PAIRS='128:128,256:256,512:512' \
#   sbatch -N 4 --cpus-per-task=512 --mem=0 \
#     /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
case "$PROFILE" in
  bsp_large)
    RUN_RAY_SET6=0
    RUN_DASK_SET7=0
    RUN_BSP_SET8=1
    NODES="${NODES:-16384}"
    FILES="${FILES:-2048}"
    NODE_CHARS="${NODE_CHARS:-1200}"
    UPSERT_CAP="${UPSERT_CAP:--1}"
    SET45_UPSERT_SHARDS="${SET45_UPSERT_SHARDS:--1}"
    NO_SCALE_SET5_BATCHES="${NO_SCALE_SET5_BATCHES:-1}"
    STRICT_STAGE_SCALING="${STRICT_STAGE_SCALING:-0}"
    SET5_EMBED_BATCH="${SET5_EMBED_BATCH:-64}"
    SET5_UPSERT_BATCH="${SET5_UPSERT_BATCH:-128}"
    REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-180}"
    PER_ITEM_MS="${PER_ITEM_MS:-3.0}"
    BATCH_SCALE_BASELINE="${BATCH_SCALE_BASELINE:-512}"
    RAY_OBJECT_STORE_MB="${RAY_OBJECT_STORE_MB:-4096}"
    WORKER_CPU_PAIRS="${WORKER_CPU_PAIRS:-128:128,256:256,512:512}"
    ;;
  no_bsp_large)
    RUN_RAY_SET6="${RUN_RAY_SET6:-1}"
    RUN_DASK_SET7="${RUN_DASK_SET7:-1}"
    RUN_BSP_SET8=0
    NODES="${NODES:-16384}"
    FILES="${FILES:-2048}"
    NODE_CHARS="${NODE_CHARS:-1200}"
    UPSERT_CAP="${UPSERT_CAP:--1}"
    SET45_UPSERT_SHARDS="${SET45_UPSERT_SHARDS:--1}"
    NO_SCALE_SET5_BATCHES="${NO_SCALE_SET5_BATCHES:-1}"
    STRICT_STAGE_SCALING="${STRICT_STAGE_SCALING:-1}"
    SET5_EMBED_BATCH="${SET5_EMBED_BATCH:-64}"
    SET5_UPSERT_BATCH="${SET5_UPSERT_BATCH:-128}"
    REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-180}"
    PER_ITEM_MS="${PER_ITEM_MS:-3.0}"
    BATCH_SCALE_BASELINE="${BATCH_SCALE_BASELINE:-512}"
    RAY_OBJECT_STORE_MB="${RAY_OBJECT_STORE_MB:-4096}"
    WORKER_CPU_PAIRS="${WORKER_CPU_PAIRS:-128:128,256:256,512:512}"
    ;;
  default)
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTDIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
if [[ ${#HOSTS[@]} -eq 0 ]]; then
  echo "No hosts resolved from SLURM_NODELIST=$SLURM_NODELIST" >&2
  exit 1
fi

echo "Job: $SLURM_JOB_ID"
echo "Profile: $PROFILE"
echo "Hosts: ${HOSTS[*]}"
echo "Pairs: $WORKER_CPU_PAIRS"
echo "Output: $OUTDIR"

IFS=',' read -r -a PAIRS <<< "$WORKER_CPU_PAIRS"

run_one() {
  local host="$1"
  local pair="$2"
  local run_tag="$3"
  local run_dir="$OUTDIR/$run_tag"

  mkdir -p "$run_dir"

  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exclusive --nodelist="$host" \
    --export=ALL,PAIR="$pair",RUN_DIR="$run_dir",PROJECT_ROOT="$PROJECT_ROOT",BENCH_SCRIPT="$BENCH_SCRIPT",PYTHON_BIN="$PYTHON_BIN",NODES="$NODES",FILES="$FILES",NODE_CHARS="$NODE_CHARS",UPSERT_CAP="$UPSERT_CAP",SET5_EMBED_BATCH="$SET5_EMBED_BATCH",SET5_UPSERT_BATCH="$SET5_UPSERT_BATCH",REQUEST_OVERHEAD_MS="$REQUEST_OVERHEAD_MS",PER_ITEM_MS="$PER_ITEM_MS",BATCH_SCALE_BASELINE="$BATCH_SCALE_BASELINE",RAY_OBJECT_STORE_MB="$RAY_OBJECT_STORE_MB",NO_SCALE_SET5_BATCHES="$NO_SCALE_SET5_BATCHES",SET45_UPSERT_SHARDS="$SET45_UPSERT_SHARDS",STRICT_STAGE_SCALING="$STRICT_STAGE_SCALING",RUN_RAY_SET6="$RUN_RAY_SET6",RUN_DASK_SET7="$RUN_DASK_SET7",RUN_BSP_SET8="$RUN_BSP_SET8" \
    bash -lc '
      set -euo pipefail
      cd "$PROJECT_ROOT"
      "$PYTHON_BIN" -c "import chromadb, ray, dask" >/dev/null

      aw="${PAIR%%:*}"
      rcpu="${PAIR##*:}"
      max_step_cpus="${SLURM_CPUS_PER_TASK:-1}"
      if [[ "$rcpu" -gt "$max_step_cpus" ]]; then
        rcpu="$max_step_cpus"
      fi
      if [[ "${UPSERT_CAP}" -lt 0 ]]; then
        upsert_cap="${aw}"
      else
        upsert_cap="${UPSERT_CAP}"
      fi
      if [[ "${SET45_UPSERT_SHARDS}" -lt 0 ]]; then
        set45_upsert_shards="${aw}"
      else
        set45_upsert_shards="${SET45_UPSERT_SHARDS}"
      fi

      out_file="$RUN_DIR/benchmark.out"
      time_file="$RUN_DIR/time.txt"
      extra_flags=()
      if [[ "$NO_SCALE_SET5_BATCHES" -eq 1 ]]; then
        extra_flags+=(--no-scale-set5-batches)
      fi
      if [[ "$STRICT_STAGE_SCALING" -eq 1 ]]; then
        extra_flags+=(--strict-stage-scaling)
      fi
      if [[ "$RUN_RAY_SET6" -eq 1 ]]; then
        extra_flags+=(--run-ray-set6)
      fi
      if [[ "$RUN_DASK_SET7" -eq 1 ]]; then
        extra_flags+=(--run-dask-set7 --dask-workers "$aw")
      fi
      if [[ "$RUN_BSP_SET8" -eq 1 ]]; then
        extra_flags+=(--run-bsp-set8 --bsp-workers "$aw")
      fi

      status=0
      /usr/bin/time -v \
        "$PYTHON_BIN" "$BENCH_SCRIPT" \
          --only-async \
          --nodes "$NODES" \
          --files "$FILES" \
          --node-chars "$NODE_CHARS" \
          --async-workers "$aw" \
          --ray-num-cpus "$rcpu" \
          --upsert-workers-cap "$upsert_cap" \
          --set5-embed-batch "$SET5_EMBED_BATCH" \
          --set5-upsert-batch "$SET5_UPSERT_BATCH" \
          --set45-upsert-shards "$set45_upsert_shards" \
          --request-overhead-ms "$REQUEST_OVERHEAD_MS" \
          --per-item-ms "$PER_ITEM_MS" \
          --batch-scale-baseline "$BATCH_SCALE_BASELINE" \
          --ray-object-store-memory-mb "$RAY_OBJECT_STORE_MB" \
          "${extra_flags[@]}" \
        >"$out_file" 2>"$time_file" || status=$?

      cpu_pct=$(awk -F": +" "/Percent of CPU this job got/ {print \$2}" "$time_file" | tr -d "%" | xargs)
      max_rss_kb=$(awk -F": +" "/Maximum resident set size/ {print \$2}" "$time_file" | xargs)
      elapsed=$(awk -F": +" "/Elapsed \(wall clock\) time/ {print \$2}" "$time_file" | xargs)

      {
        echo "host,async_workers,ray_num_cpus,elapsed,max_rss_kb,cpu_pct,exit_code"
        printf "%s,%s,%s,%s,%s,%s,%s\n" "$(hostname -s)" "$aw" "$rcpu" "$elapsed" "$max_rss_kb" "$cpu_pct" "$status"
      } > "$RUN_DIR/summary.csv"

      exit "$status"
    '
}

idx=0
fail=0
while [[ $idx -lt ${#PAIRS[@]} ]]; do
  pids=()
  wave_tags=()

  for host in "${HOSTS[@]}"; do
    if [[ $idx -ge ${#PAIRS[@]} ]]; then
      break
    fi
    pair="${PAIRS[$idx]}"
    run_tag="run_${idx}_$(echo "$pair" | tr ':' '_')_${host}"
    wave_tags+=("$run_tag")

    run_one "$host" "$pair" "$run_tag" &
    pids+=("$!")
    idx=$((idx + 1))
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  echo "Completed wave: ${wave_tags[*]}"
done

agg_csv="$OUTDIR/resource_summary.csv"
{
  echo "host,async_workers,ray_num_cpus,elapsed,max_rss_kb,cpu_pct,exit_code"
  find "$OUTDIR" -mindepth 2 -maxdepth 2 -name summary.csv -print0 \
    | xargs -0 -I{} tail -n +2 {} \
    | sort -t, -k2,2n -k3,3n
} > "$agg_csv"

awk -F, 'NR==1{print;next} {rss_gb=$5/1024/1024; printf "%s,%s,%s,%s,%.3f,%s,%s\n", $1,$2,$3,$4,rss_gb,$6,$7}' "$agg_csv" > "$OUTDIR/resource_summary_gb.csv"

echo "Wrote aggregate resource summary: $agg_csv"
echo "Wrote RSS in GiB: $OUTDIR/resource_summary_gb.csv"

if [[ $fail -ne 0 ]]; then
  echo "One or more runs failed. Check per-run folders in $OUTDIR" >&2
  exit 1
fi

echo "All runs completed successfully."
