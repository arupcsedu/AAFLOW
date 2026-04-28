#!/usr/bin/env bash
# Central local paths for Stateful Agentic Algebra.
#
# New users should edit only this file for a different checkout, Python
# environment location, or scratch/data/cache location.

# Repository checkout.
export PRJ_PATH="${PRJ_PATH:-/project/bi_dsc_community/drc_rag}"

# Parent directory containing Python virtual environments.
export ENV_PATH="${ENV_PATH:-/scratch/${USER}/env}"

# Scratch/data/cache directory for generated outputs and model caches.
export DATA_PATH="${DATA_PATH:-/scratch/${USER}/stateful_aaflow}"

# Named environments used by the Slurm scripts. Keep these derived from
# ENV_PATH unless your site uses separate environment roots.
export SAA_VLLM_ENV="${SAA_VLLM_ENV:-$ENV_PATH/saa_vllm_env}"
export SAA_BENCH_ENV="${SAA_BENCH_ENV:-$ENV_PATH/drc_rag_bench_env}"

# Backward-compatible names used by older scripts.
export PROJECT_ROOT="${PROJECT_ROOT:-$PRJ_PATH}"
export PYTHON_BIN="${PYTHON_BIN:-$SAA_VLLM_ENV/bin/python}"
export SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-$SAA_BENCH_ENV/bin/python}"
export PLOT_PYTHON_BIN="${PLOT_PYTHON_BIN:-$SGLANG_PYTHON_BIN}"

# Hugging Face cache locations. These keep large model files out of $HOME.
export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
