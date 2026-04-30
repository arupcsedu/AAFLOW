#!/bin/bash
#SBATCH --job-name=saa-paper-exp
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
export SLURM_CPU_BIND=none

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAA_ENV_FILE="${SAA_ENV_FILE:-$SCRIPT_DIR/../env.sh}"
if [[ -f "$SAA_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SAA_ENV_FILE"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${PRJ_PATH:-/project/bi_dsc_community/drc_rag}}"
ENV_PATH="${ENV_PATH:-/scratch/${USER}/env}"
DATA_PATH="${DATA_PATH:-/scratch/${USER}/stateful_aaflow}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/saa_vllm_env/bin/python}"
SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-$ENV_PATH/drc_rag_bench_env/bin/python}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must point to a paper experiment YAML file}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PLOT_PYTHON_BIN="${PLOT_PYTHON_BIN:-$ENV_PATH/drc_rag_bench_env/bin/python}"
SGLANG_GCC_MODULE="${SGLANG_GCC_MODULE:-gcc/12.4.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
JOB_PORT_OFFSET=$(( (${SLURM_JOB_ID:-0} % 400) * 50 ))
export SGLANG_PORT="${SGLANG_PORT:-$((12000 + JOB_PORT_OFFSET))}"
export VLLM_PORT="${VLLM_PORT:-$((18000 + JOB_PORT_OFFSET))}"

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.hf_token"
fi

export HF_HOME="${HF_HOME:-$DATA_PATH/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load "$SGLANG_GCC_MODULE" || true
  if [[ -n "$CUDA_MODULE" ]]; then
    module load "$CUDA_MODULE" || true
  fi
fi
export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
NVIDIA_LIB_PATHS="$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import site
from pathlib import Path

roots = [Path(p) for p in site.getsitepackages()]
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/lib") if path.is_dir()))
PY
)"
SGLANG_ENV_LIB="$("$SGLANG_PYTHON_BIN" - <<'PY' 2>/dev/null || true
import sys
from pathlib import Path

env_root = Path(sys.executable).resolve().parents[1]
lib = env_root / "lib"
print(str(lib) if (lib / "libstdc++.so.6").exists() else "")
PY
)"
LD_PREPEND=""
if [[ -n "$SGLANG_ENV_LIB" ]]; then
  LD_PREPEND="$SGLANG_ENV_LIB"
fi
if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
  LD_PREPEND="${LD_PREPEND:+$LD_PREPEND:}$NVIDIA_LIB_PATHS"
fi
if [[ -n "$LD_PREPEND" ]]; then
  export LD_LIBRARY_PATH="$LD_PREPEND:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export SGLANG_PYTHON_BIN

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_ROOT"

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Config: $CONFIG_PATH"
echo "Output override: ${OUTPUT_DIR:-from config}"
echo "Project: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "SGLang Python: $SGLANG_PYTHON_BIN"
echo "SGLang GCC module: $SGLANG_GCC_MODULE"
echo "SGLang port: $SGLANG_PORT"
echo "vLLM port: $VLLM_PORT"
echo "HF_HOME: $HF_HOME"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi -L
else
  echo "nvidia-smi not found"
fi

RUN_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR/logs"
  RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

RESOLVED_OUTPUT_DIR="$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "${OUTPUT_DIR:-}"
import sys
from pathlib import Path
from stateful_agentic_algebra.config_utils import load_config_file

config = load_config_file(sys.argv[1])
output = sys.argv[2] or str(config.get("output_dir", "runs/stateful/paper_experiment"))
print(str(Path(output)))
PY
)"

PLOT_NAMES="$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
from stateful_agentic_algebra.config_utils import config_value, load_config_file

config = load_config_file(sys.argv[1])
experiment_id = int(config_value(config, "experiment_id", default=1))
mapping = {
    1: "real_ttft_vs_context_by_baseline,real_ttft_speedup_over_dense",
    2: "real_total_latency_vs_agents_by_baseline,real_speedup_vs_agents",
    3: "real_transfer_recompute_crossover",
    4: "real_kv_memory_vs_branch",
    5: "real_throughput_by_baseline,real_omega_by_baseline",
    6: "real_consistency_exact_match_by_model",
}
print(mapping.get(experiment_id, ""))
PY
)"

/usr/bin/time -v \
  srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_BIN" -m stateful_agentic_algebra.paper_experiment_runner "${RUN_ARGS[@]}"

rm -rf "$RESOLVED_OUTPUT_DIR/figures"
PLOT_OK=0
if [[ -s "$RESOLVED_OUTPUT_DIR/results.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/results.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm \
    --plot-names "$PLOT_NAMES" && PLOT_OK=1 || true
elif [[ -s "$RESOLVED_OUTPUT_DIR/crossover.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/crossover.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm \
    --plot-names "$PLOT_NAMES" && PLOT_OK=1 || true
elif [[ -s "$RESOLVED_OUTPUT_DIR/consistency.csv" ]]; then
  "$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
    --results "$RESOLVED_OUTPUT_DIR/consistency.csv" \
    --output-dir "$RESOLVED_OUTPUT_DIR/figures" \
    --real-llm \
    --plot-names "$PLOT_NAMES" && PLOT_OK=1 || true
fi
if [[ "$PLOT_OK" -eq 1 ]]; then
  rm -f "$RESOLVED_OUTPUT_DIR/logs/plots.err"
fi
if [[ -s "$RESOLVED_OUTPUT_DIR/results_raw.jsonl" ]]; then
  "$PYTHON_BIN" - <<'PY' "$RESOLVED_OUTPUT_DIR"
import json
import sys
from dataclasses import fields
from pathlib import Path

from stateful_agentic_algebra.multi_llm_runner import MultiLLMConfig, _write_summary_out

output_dir = Path(sys.argv[1])
field_names = {field.name for field in fields(MultiLLMConfig)}
rows = [json.loads(line) for line in (output_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
config_kwargs = {key: config_data[key] for key in field_names if key in config_data}
config_kwargs.setdefault("models", config_data.get("models", [config_data.get("model_id", "gpt2")]))
config_kwargs.setdefault("backends", config_data.get("backends", [config_data.get("backend", "hf")]))
config_kwargs.setdefault("context_grid", config_data.get("context_grid", [1024]))
config_kwargs.setdefault("output_grid", config_data.get("output_grid", [64]))
config_kwargs.setdefault("agent_grid", config_data.get("agent_grid", [8]))
config_kwargs.setdefault("branch_grid", config_data.get("branch_grid", [4]))
config_kwargs.setdefault("num_prompts", config_data.get("num_prompts", 16))
config_kwargs.setdefault("output_dir", str(output_dir))
config_kwargs.setdefault("seeds", config_data.get("seeds", [0]))
_write_summary_out(output_dir / "summary.out", rows, MultiLLMConfig(**config_kwargs))
PY
else
  "$PYTHON_BIN" - <<'PY' "$RESOLVED_OUTPUT_DIR"
import json
import sys
from pathlib import Path

from stateful_agentic_algebra.paper_experiment_runner import _write_artifact_summary

output_dir = Path(sys.argv[1])
config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
_write_artifact_summary(output_dir / "summary.out", config_data, int(config_data.get("experiment_id", 0)))
PY
fi

echo "Done. Outputs:"
echo "  $RESOLVED_OUTPUT_DIR/benchmark.out"
echo "  $RESOLVED_OUTPUT_DIR/figures"
