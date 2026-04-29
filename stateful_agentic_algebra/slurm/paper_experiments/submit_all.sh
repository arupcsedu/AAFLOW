#!/bin/bash
# Submit the full paper matrix:
#   6 experiments x 2 models x 3 backends = 36 Slurm jobs.
#
# Override partition/reservation at submit time if needed, for example:
#   SBATCH_EXTRA_ARGS="--partition=bii-gpu --reservation=bi_fox_dgx" \
#     bash stateful_agentic_algebra/slurm/paper_experiments/submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

for script in "$SCRIPT_DIR"/run_exp*.sbatch; do
  echo "submitting $script"
  # shellcheck disable=SC2086
  sbatch $SBATCH_EXTRA_ARGS "$script"
done
