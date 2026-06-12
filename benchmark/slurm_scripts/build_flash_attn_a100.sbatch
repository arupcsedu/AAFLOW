#!/bin/bash
#SBATCH --job-name=build-flash-attn-a100
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-flash-attn-build-%j.out
#SBATCH --error=/scratch/djy8hg/workdir/AAFLOW/benchmark/slurm-flash-attn-build-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge/26.1.0 gcc/14.2.0 cuda/13.0.2

BASE=/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421
FLASH=/scratch/djy8hg/env/drc_rag_benchmarks_flashattn
if [ ! -x "$FLASH/bin/python" ]; then
  "$BASE/bin/python" -m venv --system-site-packages "$FLASH"
fi

export CUDA_HOME=/apps/software/standard/core/cuda/13.0.2
export PATH="$FLASH/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_NO_BUILD_ISOLATION=1
export MAX_JOBS=1
export NVCC_THREADS=1
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0
export TMPDIR=/scratch/djy8hg/tmp_flash_attn_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" /scratch/djy8hg/wheelhouse

"$FLASH/bin/python" -m pip install 'setuptools<82' 'packaging<26' wheel ninja
"$FLASH/bin/python" -m pip wheel flash-attn==2.8.3.post1 \
  --no-build-isolation \
  --no-cache-dir \
  --wheel-dir /scratch/djy8hg/wheelhouse \
  -v
"$FLASH/bin/python" -m pip install /scratch/djy8hg/wheelhouse/flash_attn-*.whl
"$FLASH/bin/python" - <<'PY'
import flash_attn
print('flash_attn version', getattr(flash_attn, '__version__', 'unknown'))
from flash_attn import flash_attn_func
print('flash_attn_func import ok', flash_attn_func)
PY
