#!/bin/bash
#SBATCH --job-name=conv-rag-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -A bii_dsc_community
#SBATCH --partition=parallel
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/slurm-%x-%j.out
#SBATCH --error=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/slurm-%x-%j.err

set -euo pipefail
export SLURM_CPU_BIND=none

PROJECT_ROOT="/project/bi_dsc_community/drc_rag"
SCRIPT="$PROJECT_ROOT/higress_agentic_benchmark/benchmark_conversational_retrieval.py"
CONDA_ENV="/scratch/djy8hg/env/drc_rag_bench_env"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/wikitext2_train}"
FILE_GLOB="${FILE_GLOB:-*}"
SAMPLE_FILES="${SAMPLE_FILES:-2000}"
MAX_CHARS="${MAX_CHARS:-4000}"
OVERLAP_CHARS="${OVERLAP_CHARS:-200}"
NUM_CASES="${NUM_CASES:-2000}"
TOP_K="${TOP_K:-1600}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/higress_agentic_benchmark/slurm_runs/${SLURM_JOB_ID}}"

mkdir -p "$OUTDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "Job: $SLURM_JOB_ID"
echo "Data: $DATA_DIR"
echo "Sample files: $SAMPLE_FILES"
echo "Cases: $NUM_CASES"
echo "Top-k: $TOP_K"
echo "Workers: $NUM_WORKERS"
echo "Output: $OUTDIR"

cd "$PROJECT_ROOT"

/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$PYTHON_BIN" -m higress_agentic_benchmark.benchmark_conversational_retrieval \
    --data-dir "$DATA_DIR" \
    --file-glob "$FILE_GLOB" \
    --sample-files "$SAMPLE_FILES" \
    --max-chars "$MAX_CHARS" \
    --overlap-chars "$OVERLAP_CHARS" \
    --num-cases "$NUM_CASES" \
    --top-k "$TOP_K" \
    --progress-every "$PROGRESS_EVERY" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTDIR" \
  >"$OUTDIR/benchmark.out" 2>"$OUTDIR/time.txt"

echo "Done. Outputs:"
echo "  $OUTDIR/retrieval_cases.csv"
echo "  $OUTDIR/retrieval_results.csv"
echo "  $OUTDIR/retrieval_summary.csv"
echo "  $OUTDIR/retrieval_summary.json"
