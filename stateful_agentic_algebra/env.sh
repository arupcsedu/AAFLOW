#!/usr/bin/env bash
# Central local paths for Stateful Agentic Algebra.
#
# New users should edit only this file for a different checkout, Python
# environment location, or scratch/data/cache location.

# Repository checkout.
export PRJ_PATH="${PRJ_PATH:-/raid/${USER}/drc_rag}"

# Parent directory containing Python virtual environments.
export ENV_PATH="${ENV_PATH:-/raid/${USER}/venv}"

# Scratch/data/cache directory for generated outputs and model caches.
export DATA_PATH="${DATA_PATH:-/raid/${USER}/stateful_aaflow}"

# Named environments used by the Slurm scripts. Keep these derived from
# ENV_PATH unless your site uses separate environment roots.
export SAA_VLLM_ENV="${SAA_VLLM_ENV:-$ENV_PATH/saa_vllm_env}"
export SAA_BENCH_ENV="${SAA_BENCH_ENV:-$ENV_PATH/saa_sglang_env}"

# Optional Slurm submission defaults. Leave these empty for clusters that have
# suitable defaults, or set them once here and let submit helpers consume them.
export SAA_SLURM_ACCOUNT="${SAA_SLURM_ACCOUNT:-}"
export SAA_SLURM_PARTITION="${SAA_SLURM_PARTITION:-}"
export SAA_SLURM_RESERVATION="${SAA_SLURM_RESERVATION:-}"
export SAA_SLURM_GRES="${SAA_SLURM_GRES:-}"

# Optional user-local CUDA toolkit. SGLang/FlashInfer may invoke `nvcc` for
# JIT kernels, so this should point to CUDA 12.8+ on A100/H100 systems. Prefer
# the project CUDA toolkit when present, even if the login shell inherited an
# older CUDA_HOME such as /usr/local/cuda.
export SAA_CUDA_HOME="${SAA_CUDA_HOME:-/raid/${USER}/cuda-12.8}"
if [[ -x "$SAA_CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$SAA_CUDA_HOME"
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

# Make venv-provided helper executables such as `ninja`, `sglang`, and `vllm`
# visible to subprocesses launched by model-serving frameworks.
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$(dirname "$SGLANG_PYTHON_BIN"):$(dirname "$PYTHON_BIN"):${PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
  export PATH="$(dirname "$SGLANG_PYTHON_BIN"):$(dirname "$PYTHON_BIN"):${PATH:-}"
fi

# Hugging Face cache locations. These keep large model files out of $HOME.
export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
