#!/usr/bin/env bash
# Central local paths for Stateful Agentic Algebra.
#
# New users should edit only this file for a different checkout, Python
# environment location, or scratch/data/cache location.

# Repository checkout for DGX A100/H100 systems. This should point to the root of the cloned repository
#export PRJ_PATH="${PRJ_PATH:-/raid/${USER}/drc_rag}"

# Parent directory containing Python virtual environments.
#export ENV_PATH="${ENV_PATH:-/raid/${USER}/venv}"

# Scratch/data/cache directory for generated outputs and model caches.
#export DATA_PATH="${DATA_PATH:-/raid/${USER}/stateful_aaflow}"

# Repository checkout for Rivanna A100/H100 systems. This should point to the root of the cloned repository
export PRJ_PATH="${PRJ_PATH:-/project/bi_dsc_community/drc_rag}"


# Parent directory containing Python virtual environments.
export ENV_PATH="${ENV_PATH:-/scratch/${USER}/env}"

# Scratch/data/cache directory for generated outputs and model caches.
export DATA_PATH="${DATA_PATH:-/scratch/${USER}/stateful_aaflow}"

# Named environments used by the Slurm scripts. Keep these derived from
# ENV_PATH unless your site uses separate environment roots.
export SAA_VLLM_ENV="${SAA_VLLM_ENV:-$ENV_PATH/saa_vllm_env}"
export SAA_BENCH_ENV="${SAA_BENCH_ENV:-$ENV_PATH/drc_rag_bench_env}"

# Optional Slurm submission defaults. Leave these empty for clusters that have
# suitable defaults, or set them once here and let submit helpers consume them.
export SAA_SLURM_ACCOUNT="${SAA_SLURM_ACCOUNT:-}"
export SAA_SLURM_PARTITION="${SAA_SLURM_PARTITION:-}"
export SAA_SLURM_RESERVATION="${SAA_SLURM_RESERVATION:-}"
export SAA_SLURM_GRES="${SAA_SLURM_GRES:-}"

# CUDA toolkit selection. Keep CUDA dependencies inside the Python
# environments. Do not use /usr/local CUDA here: it makes runs depend on node
# image state and can mismatch the CUDA wheels installed in the venvs.
export SAA_CUDA_HOME="${SAA_CUDA_HOME:-/raid/${USER}/cuda-12.8}"
SAA_CUDA_CANDIDATES=(
  "$SAA_VLLM_ENV"/lib/python*/site-packages/nvidia/cuda_nvcc
  "$SAA_BENCH_ENV"/lib/python*/site-packages/nvidia/cuda_nvcc
  "$SAA_CUDA_HOME"
)
SAA_SELECTED_CUDA_HOME=""
for SAA_CUDA_CANDIDATE in "${SAA_CUDA_CANDIDATES[@]}"; do
  if [[ -n "$SAA_CUDA_CANDIDATE" ]] && {
    [[ -x "$SAA_CUDA_CANDIDATE/bin/nvcc" ]] ||
    [[ -x "$SAA_CUDA_CANDIDATE/bin/ptxas" ]] ||
    [[ -f "$SAA_CUDA_CANDIDATE/nvvm/lib64/libnvvm.so" ]]
  }; then
    SAA_SELECTED_CUDA_HOME="$SAA_CUDA_CANDIDATE"
    break
  fi
done
if [[ -n "$SAA_SELECTED_CUDA_HOME" ]]; then
  export CUDA_HOME="$SAA_SELECTED_CUDA_HOME"
else
  export CUDA_HOME="${CUDA_HOME:-$SAA_CUDA_HOME}"
fi
export CUDA_PATH="$CUDA_HOME"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Backward-compatible names used by older scripts.
export PROJECT_ROOT="${PROJECT_ROOT:-$PRJ_PATH}"
export PYTHON_BIN="${PYTHON_BIN:-$SAA_VLLM_ENV/bin/python}"
export SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-$SAA_BENCH_ENV/bin/python}"
export PLOT_PYTHON_BIN="${PLOT_PYTHON_BIN:-$SGLANG_PYTHON_BIN}"

SAA_NVIDIA_LIB_PATHS=""
for SAA_PY in "$PYTHON_BIN" "$SGLANG_PYTHON_BIN"; do
  if [[ -x "$SAA_PY" ]]; then
    SAA_PY_NVIDIA_LIB_PATHS="$("$SAA_PY" - <<'PY' 2>/dev/null || true
import site
from pathlib import Path

roots = []
try:
    roots.extend(Path(path) for path in site.getsitepackages())
except Exception:
    pass
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/lib") if path.is_dir()))
PY
)"
    if [[ -n "$SAA_PY_NVIDIA_LIB_PATHS" ]]; then
      SAA_NVIDIA_LIB_PATHS="${SAA_NVIDIA_LIB_PATHS:+$SAA_NVIDIA_LIB_PATHS:}$SAA_PY_NVIDIA_LIB_PATHS"
    fi
  fi
done

# Make venv-provided helper executables such as `ninja`, `sglang`, and `vllm`
# visible to subprocesses launched by model-serving frameworks.
SAA_CUDA_PATH_PREFIX=""
SAA_CUDA_LD_PREFIX=""
if [[ -d "$CUDA_HOME/bin" ]]; then
  SAA_CUDA_PATH_PREFIX="$CUDA_HOME/bin:"
fi
for SAA_CUDA_LIB_DIR in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/nvvm/lib64"; do
  if [[ -d "$SAA_CUDA_LIB_DIR" ]]; then
    SAA_CUDA_LD_PREFIX="${SAA_CUDA_LD_PREFIX:+$SAA_CUDA_LD_PREFIX:}$SAA_CUDA_LIB_DIR"
  fi
done
if [[ -n "$SAA_CUDA_PATH_PREFIX" || -n "$SAA_CUDA_LD_PREFIX" ]]; then
  export PATH="$SAA_CUDA_PATH_PREFIX$(dirname "$SGLANG_PYTHON_BIN"):$(dirname "$PYTHON_BIN"):${PATH:-}"
  export LD_LIBRARY_PATH="${SAA_CUDA_LD_PREFIX}${SAA_NVIDIA_LIB_PATHS:+:$SAA_NVIDIA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
else
  export PATH="$(dirname "$SGLANG_PYTHON_BIN"):$(dirname "$PYTHON_BIN"):${PATH:-}"
  if [[ -n "$SAA_NVIDIA_LIB_PATHS" ]]; then
    export LD_LIBRARY_PATH="$SAA_NVIDIA_LIB_PATHS:${LD_LIBRARY_PATH:-}"
  fi
fi

# Hugging Face cache locations. These keep large model files out of $HOME.
export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
