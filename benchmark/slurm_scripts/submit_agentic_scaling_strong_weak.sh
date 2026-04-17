#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project/bi_dsc_community"
SBATCH_SCRIPT="$PROJECT_ROOT/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch"
PROFILE="${PROFILE:-strong_no_ray}"
CORES_PER_NODE="${CORES_PER_NODE:-40}"
WORKERS="${WORKERS:-256,512,1024,2048,4096,8192}"
DRY_RUN="${DRY_RUN:-0}"
COMMON_EXPORTS="${COMMON_EXPORTS:-EMBEDDER_BACKEND=local-hash,SET5_SINK_BACKEND=faiss}"

IFS=',' read -r -a WLIST <<< "$WORKERS"
for w in "${WLIST[@]}"; do
  nodes=$(( (w + CORES_PER_NODE - 1) / CORES_PER_NODE ))
  job_name="ragscale-${PROFILE}-${w}w"
  export_str="ALL,PROFILE=${PROFILE},PHYSICAL_WORKERS=${w},CORES_PER_NODE=${CORES_PER_NODE},${COMMON_EXPORTS}"
  cmd=(sbatch --job-name "$job_name" --nodes "$nodes" --export "$export_str" "$SBATCH_SCRIPT")
  test_cmd=(sbatch --test-only --job-name "$job_name" --nodes "$nodes" --export "$export_str" "$SBATCH_SCRIPT")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    if "${test_cmd[@]}" >/dev/null 2>&1; then
      "${cmd[@]}"
    else
      echo "Skipping ${job_name}: unschedulable with nodes=${nodes} on current partition/account limits." >&2
    fi
  fi
done
