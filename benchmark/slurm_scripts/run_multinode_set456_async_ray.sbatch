#!/bin/bash
#SBATCH --job-name=ragbench-set456
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
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
AGENTIC_ORDER="${AGENTIC_ORDER:-}"

# Dataset/workload knobs
NODES="${NODES:-16384}"
FILES="${FILES:-1024}"
NODE_CHARS="${NODE_CHARS:-900}"
UPSERT_CAP="${UPSERT_CAP:--1}"
DIM="${DIM:-128}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-local-hash}"
SET5_SINK_BACKEND="${SET5_SINK_BACKEND:-chroma}"
SET5_EMBED_BATCH="${SET5_EMBED_BATCH:-64}"
SET5_UPSERT_BATCH="${SET5_UPSERT_BATCH:-256}"
SET5_LOAD_WORKERS="${SET5_LOAD_WORKERS:-0}"
SET5_TRANSFORM_WORKERS="${SET5_TRANSFORM_WORKERS:-0}"
SET5_EMBED_WORKERS="${SET5_EMBED_WORKERS:-0}"
SET5_UPSERT_WORKERS="${SET5_UPSERT_WORKERS:-0}"
SET5_UPSERT_TIMEOUT_MS="${SET5_UPSERT_TIMEOUT_MS:-0}"
SET5_UPSERT_SHARDS="${SET5_UPSERT_SHARDS:-0}"
REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-180}"
PER_ITEM_MS="${PER_ITEM_MS:-3.0}"
BATCH_SCALE_BASELINE="${BATCH_SCALE_BASELINE:-64}"
RAY_OBJECT_STORE_MB="${RAY_OBJECT_STORE_MB:-2048}"
NO_SCALE_SET5_BATCHES="${NO_SCALE_SET5_BATCHES:-1}"
SET45_UPSERT_SHARDS="${SET45_UPSERT_SHARDS:--1}"
STRICT_STAGE_SCALING="${STRICT_STAGE_SCALING:-1}"
RUN_RAY_SET6="${RUN_RAY_SET6:-1}"
RUN_DASK_SET7="${RUN_DASK_SET7:-1}"
RUN_BSP_SET8="${RUN_BSP_SET8:-1}"
RUN_HIGRESS_SET9="${RUN_HIGRESS_SET9:-1}"

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
#
#   PROFILE=no_ray \
#   sbatch /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
#   PROFILE=ray_only \
#   sbatch /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
case "$PROFILE" in
  no_ray)
    RUN_RAY_SET6=0
    ;;
  ray_only)
    RUN_RAY_SET6=1
    RUN_DASK_SET7=0
    RUN_BSP_SET8=0
    RUN_HIGRESS_SET9=0
    ;;
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
    --export=ALL,PAIR="$pair",RUN_DIR="$run_dir",PROJECT_ROOT="$PROJECT_ROOT",BENCH_SCRIPT="$BENCH_SCRIPT",PYTHON_BIN="$PYTHON_BIN",NODES="$NODES",FILES="$FILES",NODE_CHARS="$NODE_CHARS",UPSERT_CAP="$UPSERT_CAP",DIM="$DIM",EMBEDDER_BACKEND="$EMBEDDER_BACKEND",SINK_BACKEND="$SET5_SINK_BACKEND",SET5_SINK_BACKEND="$SET5_SINK_BACKEND",SET5_EMBED_BATCH="$SET5_EMBED_BATCH",SET5_UPSERT_BATCH="$SET5_UPSERT_BATCH",SET5_LOAD_WORKERS="$SET5_LOAD_WORKERS",SET5_TRANSFORM_WORKERS="$SET5_TRANSFORM_WORKERS",SET5_EMBED_WORKERS="$SET5_EMBED_WORKERS",SET5_UPSERT_WORKERS="$SET5_UPSERT_WORKERS",SET5_UPSERT_TIMEOUT_MS="$SET5_UPSERT_TIMEOUT_MS",SET5_UPSERT_SHARDS="$SET5_UPSERT_SHARDS",REQUEST_OVERHEAD_MS="$REQUEST_OVERHEAD_MS",PER_ITEM_MS="$PER_ITEM_MS",BATCH_SCALE_BASELINE="$BATCH_SCALE_BASELINE",RAY_OBJECT_STORE_MB="$RAY_OBJECT_STORE_MB",NO_SCALE_SET5_BATCHES="$NO_SCALE_SET5_BATCHES",SET45_UPSERT_SHARDS="$SET45_UPSERT_SHARDS",STRICT_STAGE_SCALING="$STRICT_STAGE_SCALING",RUN_RAY_SET6="$RUN_RAY_SET6",RUN_DASK_SET7="$RUN_DASK_SET7",RUN_BSP_SET8="$RUN_BSP_SET8",RUN_HIGRESS_SET9="$RUN_HIGRESS_SET9" \
    bash -lc '
      set -euo pipefail
      cd "$PROJECT_ROOT"
      "$PYTHON_BIN" -c "import chromadb, ray, dask, faiss" >/dev/null

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
      agentic_order="${AGENTIC_ORDER:-}"
      if [[ -z "$agentic_order" ]]; then
        if [[ "$SET5_SINK_BACKEND" == "faiss" ]]; then
          agentic_order="last"
        else
          agentic_order="early"
        fi
      fi

      if [[ "$SET5_SINK_BACKEND" == "faiss" ]]; then
        if [[ "$SET5_LOAD_WORKERS" -gt 0 ]]; then
          set5_load_workers="$SET5_LOAD_WORKERS"
        else
          case "$aw" in
            4) set5_load_workers=8 ;;
            *) set5_load_workers="$aw" ;;
          esac
        fi
        if [[ "$SET5_TRANSFORM_WORKERS" -gt 0 ]]; then
          set5_transform_workers="$SET5_TRANSFORM_WORKERS"
        else
          case "$aw" in
            4) set5_transform_workers=8 ;;
            *) set5_transform_workers="$aw" ;;
          esac
        fi
        if [[ "$SET5_EMBED_WORKERS" -gt 0 ]]; then
          set5_embed_workers="$SET5_EMBED_WORKERS"
        else
          case "$aw" in
            4) set5_embed_workers=4 ;;
            *) set5_embed_workers="$aw" ;;
          esac
        fi
        if [[ "$SET5_UPSERT_WORKERS" -gt 0 ]]; then
          set5_upsert_workers="$SET5_UPSERT_WORKERS"
        else
          set5_upsert_workers=1
        fi
        if [[ "$SET5_UPSERT_SHARDS" -gt 0 ]]; then
          set5_upsert_shards="$SET5_UPSERT_SHARDS"
        else
          set5_upsert_shards=1
        fi
        if [[ "$SET5_UPSERT_TIMEOUT_MS" != "0" && "$SET5_UPSERT_TIMEOUT_MS" != "0.0" ]]; then
          set5_upsert_timeout_ms="$SET5_UPSERT_TIMEOUT_MS"
        else
          set5_upsert_timeout_ms=2
        fi
        if [[ "$SET5_EMBED_BATCH" -lt 512 ]]; then
          set5_embed_batch=512
        else
          set5_embed_batch="$SET5_EMBED_BATCH"
        fi
        case "$aw" in
          4) set5_upsert_batch=2048 ;;
          8) set5_upsert_batch=2048 ;;
          12) set5_upsert_batch=3072 ;;
          16) set5_upsert_batch=4096 ;;
          *) set5_upsert_batch="$SET5_UPSERT_BATCH" ;;
        esac
      else

      if [[ "$SET5_EMBED_WORKERS" -gt 0 ]]; then
        set5_embed_workers="$SET5_EMBED_WORKERS"
      else
        case "$aw" in
          4) set5_embed_workers=8 ;;
          8) set5_embed_workers=12 ;;
          12) set5_embed_workers=16 ;;
          16) set5_embed_workers=16 ;;
          *) set5_embed_workers="$aw" ;;
        esac
      fi
      if [[ "$SET5_UPSERT_WORKERS" -gt 0 ]]; then
        set5_upsert_workers="$SET5_UPSERT_WORKERS"
      else
        case "$aw" in
          4) set5_upsert_workers=2 ;;
          8) set5_upsert_workers=4 ;;
          12) set5_upsert_workers=3 ;;
          16) set5_upsert_workers=4 ;;
          *) set5_upsert_workers=$(( aw / 4 )) ;;
        esac
        if [[ "$set5_upsert_workers" -lt 1 ]]; then set5_upsert_workers=1; fi
      fi
      if [[ "$set5_embed_workers" -gt "$max_step_cpus" ]]; then
        set5_embed_workers="$max_step_cpus"
      fi
      if [[ "$set5_upsert_workers" -gt "$max_step_cpus" ]]; then
        set5_upsert_workers="$max_step_cpus"
      fi
      if [[ "$SET5_UPSERT_SHARDS" -gt 0 ]]; then
        set5_upsert_shards="$SET5_UPSERT_SHARDS"
      else
        case "$aw" in
          4) set5_upsert_shards=16 ;;
          8) set5_upsert_shards=16 ;;
          12) set5_upsert_shards=24 ;;
          16) set5_upsert_shards=32 ;;
          *) set5_upsert_shards=$(( set5_upsert_workers * 8 )) ;;
        esac
      fi
      if [[ "$SET5_UPSERT_TIMEOUT_MS" != "0" && "$SET5_UPSERT_TIMEOUT_MS" != "0.0" ]]; then
        set5_upsert_timeout_ms="$SET5_UPSERT_TIMEOUT_MS"
      else
        case "$aw" in
          4) set5_upsert_timeout_ms=2 ;;
          8) set5_upsert_timeout_ms=4 ;;
          12) set5_upsert_timeout_ms=2 ;;
          16) set5_upsert_timeout_ms=2 ;;
          *) set5_upsert_timeout_ms=2 ;;
        esac
      fi
      set5_load_workers="${SET5_LOAD_WORKERS:-0}"
      set5_transform_workers="${SET5_TRANSFORM_WORKERS:-0}"
      set5_embed_batch="$SET5_EMBED_BATCH"
      set5_upsert_batch="$SET5_UPSERT_BATCH"
      fi

      out_file="$RUN_DIR/benchmark.out"
      time_file="$RUN_DIR/time.txt"
      extra_flags=()
      mode_flag="--only-async"
      if [[ "$PROFILE" == "ray_only" ]]; then
        mode_flag="--only-ray"
      fi
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
      if [[ "$RUN_HIGRESS_SET9" -eq 1 ]]; then
        extra_flags+=(--run-higress-set9 --higress-workers "$aw")
      fi

      status=0
      /usr/bin/time -v \
        "$PYTHON_BIN" "$BENCH_SCRIPT" \
          "$mode_flag" \
          --nodes "$NODES" \
          --files "$FILES" \
          --node-chars "$NODE_CHARS" \
          --dim "$DIM" \
          --embedder-backend "$EMBEDDER_BACKEND" \
          --sink-backend "$SINK_BACKEND" \
          --agentic-order "$agentic_order" \
          --async-workers "$aw" \
          --ray-num-cpus "$rcpu" \
          --upsert-workers-cap "$upsert_cap" \
          --set5-sink-backend "$SET5_SINK_BACKEND" \
          --set5-embed-batch "$set5_embed_batch" \
          --set5-upsert-batch "$set5_upsert_batch" \
          --set5-load-workers "$set5_load_workers" \
          --set5-transform-workers "$set5_transform_workers" \
          --set5-embed-workers "$set5_embed_workers" \
          --set5-upsert-workers "$set5_upsert_workers" \
          --set5-upsert-timeout-ms "$set5_upsert_timeout_ms" \
          --set5-upsert-shards "$set5_upsert_shards" \
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
