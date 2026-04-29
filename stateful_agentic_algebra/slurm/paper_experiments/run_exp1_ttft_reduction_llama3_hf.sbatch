#!/bin/bash
#SBATCH --job-name=exp1_ttft_reduction_llama3_hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${PRJ_PATH:-/project/bi_dsc_community/drc_rag}}"
export CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/stateful_agentic_algebra/configs/paper_experiments/exp1_ttft_reduction_llama3_hf.yaml}"
exec bash "$PROJECT_ROOT/stateful_agentic_algebra/slurm/run_paper_experiment.sbatch"
