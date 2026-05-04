#!/bin/bash
# Submit the full paper matrix:
#   6 experiments x 2 models x 3 backends = 36 Slurm jobs.
#
# Set SAA_SLURM_* in stateful_agentic_algebra/env.sh or pass extra sbatch
# flags at submit time, for example:
#   SBATCH_EXTRA_ARGS="-A <account> -p <partition> --gres=gpu:<type>:2" \
#     bash stateful_agentic_algebra/slurm/paper_experiments/submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAA_ENV_FILE="${SAA_ENV_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
else
  echo "Missing Stateful Agentic Algebra env file: $SAA_ENV_FILE" >&2
  exit 1
fi

SBATCH_ARGS=()
if [[ -n "${SAA_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(-A "$SAA_SLURM_ACCOUNT")
fi
if [[ -n "${SAA_SLURM_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(-p "$SAA_SLURM_PARTITION")
fi
if [[ -n "${SAA_SLURM_RESERVATION:-}" ]]; then
  SBATCH_ARGS+=(--reservation="$SAA_SLURM_RESERVATION")
fi
if [[ -n "${SAA_SLURM_GRES:-}" ]]; then
  SBATCH_ARGS+=(--gres="$SAA_SLURM_GRES")
fi
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

for script in "$SCRIPT_DIR"/run_exp*.sbatch; do
  echo "submitting $script"
  # shellcheck disable=SC2086
  sbatch "${SBATCH_ARGS[@]}" $SBATCH_EXTRA_ARGS "$script"
done
