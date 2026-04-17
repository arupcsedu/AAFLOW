#!/bin/bash
#SBATCH --job-name=ragbench-agentic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/benchmark/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

PROJECT_ROOT="/project/bi_dsc_community"
BENCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/benchmark_configs_1_to_5.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/drc_rag/benchmark/slurm_runs_agentic_only/${SLURM_JOB_ID}}"

NODES="${NODES:-8000}"
FILES="${FILES:-500}"
NODE_CHARS="${NODE_CHARS:-900}"
CHUNKS_PER_FILE="${CHUNKS_PER_FILE:-64}"
DIM="${DIM:-128}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-local-hash}"
SET5_SINK_BACKEND="${SET5_SINK_BACKEND:-chroma}"
SET5_EMBED_BATCH="${SET5_EMBED_BATCH:-64}"
SET5_UPSERT_BATCH="${SET5_UPSERT_BATCH:-2048}"
SET5_UPSERT_COALESCE_MULTIPLIER="${SET5_UPSERT_COALESCE_MULTIPLIER:-64}"
REQUEST_OVERHEAD_MS="${REQUEST_OVERHEAD_MS:-180}"
PER_ITEM_MS="${PER_ITEM_MS:-3.0}"
BATCH_SCALE_BASELINE="${BATCH_SCALE_BASELINE:-64}"
NO_SCALE_SET5_BATCHES="${NO_SCALE_SET5_BATCHES:-1}"
STRICT_STAGE_SCALING="${STRICT_STAGE_SCALING:-1}"
WORKER_CPU_PAIRS="${WORKER_CPU_PAIRS:-4:4,8:8,12:12,16:16}"

mkdir -p "$OUTDIR"
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
IFS=',' read -r -a PAIRS <<< "$WORKER_CPU_PAIRS"

run_one() {
  local host="$1"
  local pair="$2"
  local run_tag="$3"
  local run_dir="$OUTDIR/$run_tag"
  mkdir -p "$run_dir"

  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exclusive --nodelist="$host" \
    --export=ALL,PAIR="$pair",RUN_DIR="$run_dir",PROJECT_ROOT="$PROJECT_ROOT",BENCH_SCRIPT="$BENCH_SCRIPT",PYTHON_BIN="$PYTHON_BIN",NODES="$NODES",FILES="$FILES",NODE_CHARS="$NODE_CHARS",CHUNKS_PER_FILE="$CHUNKS_PER_FILE",DIM="$DIM",EMBEDDER_BACKEND="$EMBEDDER_BACKEND",SET5_SINK_BACKEND="$SET5_SINK_BACKEND",SET5_EMBED_BATCH="$SET5_EMBED_BATCH",SET5_UPSERT_BATCH="$SET5_UPSERT_BATCH",SET5_UPSERT_COALESCE_MULTIPLIER="$SET5_UPSERT_COALESCE_MULTIPLIER",REQUEST_OVERHEAD_MS="$REQUEST_OVERHEAD_MS",PER_ITEM_MS="$PER_ITEM_MS",BATCH_SCALE_BASELINE="$BATCH_SCALE_BASELINE",NO_SCALE_SET5_BATCHES="$NO_SCALE_SET5_BATCHES",STRICT_STAGE_SCALING="$STRICT_STAGE_SCALING" \
    bash -lc '
      set -euo pipefail
      cd "$PROJECT_ROOT"
      aw="${PAIR%%:*}"
      rcpu="${PAIR##*:}"
      max_step_cpus="${SLURM_CPUS_PER_TASK:-1}"
      if [[ "$rcpu" -gt "$max_step_cpus" ]]; then rcpu="$max_step_cpus"; fi

      case "$aw" in
        4)  set5_load_workers=4;  set5_transform_workers=4;  set5_embed_workers=8;  set5_upsert_workers=2; set5_upsert_shards=8;  set5_upsert_timeout_ms=6;  set5_upsert_batch_runtime=2048; set5_upsert_coalesce_runtime=64  ;;
        8)  set5_load_workers=8;  set5_transform_workers=8;  set5_embed_workers=12; set5_upsert_workers=2; set5_upsert_shards=8;  set5_upsert_timeout_ms=10; set5_upsert_batch_runtime=3072; set5_upsert_coalesce_runtime=96  ;;
        12) set5_load_workers=12; set5_transform_workers=12; set5_embed_workers=16; set5_upsert_workers=1; set5_upsert_shards=8;  set5_upsert_timeout_ms=12; set5_upsert_batch_runtime=4096; set5_upsert_coalesce_runtime=128 ;;
        16) set5_load_workers=16; set5_transform_workers=16; set5_embed_workers=16; set5_upsert_workers=1; set5_upsert_shards=8;  set5_upsert_timeout_ms=16; set5_upsert_batch_runtime=6144; set5_upsert_coalesce_runtime=192 ;;
        *)  set5_load_workers="$max_step_cpus"; set5_transform_workers="$max_step_cpus"; set5_embed_workers="$max_step_cpus"; set5_upsert_workers=1; set5_upsert_shards=8; set5_upsert_timeout_ms=16; set5_upsert_batch_runtime="$SET5_UPSERT_BATCH"; set5_upsert_coalesce_runtime="$SET5_UPSERT_COALESCE_MULTIPLIER" ;;
      esac

      if [[ "$SET5_SINK_BACKEND" == "thin-batched" ]]; then
        case "$aw" in
          4)  set5_upsert_workers=2; set5_upsert_shards=12; set5_upsert_timeout_ms=4; set5_upsert_batch_runtime=1024; set5_upsert_coalesce_runtime=32 ;;
          8)  set5_upsert_workers=3; set5_upsert_shards=12; set5_upsert_timeout_ms=6; set5_upsert_batch_runtime=1536; set5_upsert_coalesce_runtime=48 ;;
          12) set5_upsert_workers=2; set5_upsert_shards=16; set5_upsert_timeout_ms=6; set5_upsert_batch_runtime=2048; set5_upsert_coalesce_runtime=64 ;;
          16) set5_upsert_workers=2; set5_upsert_shards=16; set5_upsert_timeout_ms=8; set5_upsert_batch_runtime=2048; set5_upsert_coalesce_runtime=64 ;;
          *)  set5_upsert_workers=2; set5_upsert_shards=16; set5_upsert_timeout_ms=8; set5_upsert_batch_runtime=2048; set5_upsert_coalesce_runtime=64 ;;
        esac
      fi

      out_file="$RUN_DIR/benchmark.out"
      time_file="$RUN_DIR/time.txt"
      extra_flags=()
      if [[ "$NO_SCALE_SET5_BATCHES" -eq 1 ]]; then extra_flags+=(--no-scale-set5-batches); fi
      if [[ "$STRICT_STAGE_SCALING" -eq 1 ]]; then extra_flags+=(--strict-stage-scaling); fi

      status=0
      /usr/bin/time -v \
        "$PYTHON_BIN" "$BENCH_SCRIPT" \
          --only-agentic \
          --nodes "$NODES" \
          --files "$FILES" \
          --chunks-per-file "$CHUNKS_PER_FILE" \
          --node-chars "$NODE_CHARS" \
          --dim "$DIM" \
          --embedder-backend "$EMBEDDER_BACKEND" \
          --set5-sink-backend "$SET5_SINK_BACKEND" \
          --async-workers "$aw" \
          --set5-load-workers "$set5_load_workers" \
          --set5-transform-workers "$set5_transform_workers" \
          --ray-num-cpus "$rcpu" \
          --set5-embed-batch "$SET5_EMBED_BATCH" \
          --set5-upsert-batch "$set5_upsert_batch_runtime" \
          --set5-upsert-coalesce-multiplier "$set5_upsert_coalesce_runtime" \
          --set5-embed-workers "$set5_embed_workers" \
          --set5-upsert-workers "$set5_upsert_workers" \
          --set5-upsert-timeout-ms "$set5_upsert_timeout_ms" \
          --set5-upsert-shards "$set5_upsert_shards" \
          --set45-upsert-shards "$aw" \
          --request-overhead-ms "$REQUEST_OVERHEAD_MS" \
          --per-item-ms "$PER_ITEM_MS" \
          --batch-scale-baseline "$BATCH_SCALE_BASELINE" \
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
    ' &
}

idx=0
for pair in "${PAIRS[@]}"; do
  host="${HOSTS[$(( idx % ${#HOSTS[@]} ))]}"
  aw="${pair%%:*}"
  rcpu="${pair##*:}"
  run_tag="run_${idx}_${aw}_${rcpu}_$(echo "$host" | tr ',' '_')"
  run_one "$host" "$pair" "$run_tag"
  idx=$((idx + 1))
done

wait
