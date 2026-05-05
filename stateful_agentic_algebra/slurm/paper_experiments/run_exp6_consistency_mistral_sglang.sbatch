#!/bin/bash
#SBATCH --job-name=exp6_consistency_mistral_sglang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SAA_ENV_FILE:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/stateful_agentic_algebra/env.sh" ]]; then
    SAA_ENV_FILE="$SLURM_SUBMIT_DIR/stateful_agentic_algebra/env.sh"
  else
    SAA_ENV_FILE="$(cd "$SCRIPT_DIR/../.." && pwd)/env.sh"
  fi
fi
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
else
  echo "Missing Stateful Agentic Algebra env file: $SAA_ENV_FILE" >&2
  exit 1
fi

export CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/stateful_agentic_algebra/configs/paper_experiments/exp6_consistency_mistral_sglang.yaml}"
exec bash "$PROJECT_ROOT/stateful_agentic_algebra/slurm/run_paper_experiment.sbatch"
