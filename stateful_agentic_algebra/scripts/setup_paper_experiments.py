"""Generate full-paper experiment configs and Slurm wrappers.

The generated matrix is:
  6 experiments x 2 models x 3 backends = 36 configs and 36 sbatch wrappers.

This script is intentionally deterministic so users can regenerate the setup
after editing the grids below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "stateful_agentic_algebra" / "configs" / "paper_experiments"
SLURM_DIR = ROOT / "stateful_agentic_algebra" / "slurm" / "paper_experiments"

MODELS = {
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_grid": [1024, 4096, 8192, 16384, 24576, 28672, 30720, 32640],
        "fixed_context": 8192,
        "sglang_context": 32768,
    },
    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "context_grid": [1024, 2048, 4096, 6144, 7680, 8128],
        "fixed_context": 4096,
        "sglang_context": 8192,
    },
}

BACKENDS = ("hf", "vllm", "sglang")

EXPERIMENTS = {
    1: {
        "name": "ttft_reduction",
        "description": "TTFT vs context length",
    },
    2: {
        "name": "multi_agent_scaling",
        "description": "Latency and speedup vs number of agents",
    },
    3: {
        "name": "transfer_recompute",
        "description": "KV transfer vs dense recomputation crossover",
    },
    4: {
        "name": "memory_efficiency",
        "description": "Peak KV memory vs branch factor",
    },
    5: {
        "name": "throughput_overhead",
        "description": "Throughput and framework overhead Omega",
    },
    6: {
        "name": "consistency",
        "description": "Dense vs cached exact-match consistency",
    },
}


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for exp_id, exp in EXPERIMENTS.items():
        for model_key, model in MODELS.items():
            for backend in BACKENDS:
                name = f"exp{exp_id}_{exp['name']}_{model_key}_{backend}"
                config = _config_for(exp_id, exp["name"], model_key, model, backend, name)
                config_path = CONFIG_DIR / f"{name}.yaml"
                slurm_path = SLURM_DIR / f"run_{name}.sbatch"
                config_path.write_text(_yaml(config), encoding="utf-8")
                slurm_path.write_text(_slurm_wrapper(name, config_path), encoding="utf-8")
                slurm_path.chmod(0o755)
                generated.extend([config_path, slurm_path])
    print(f"generated {len(generated)} files")
    for path in generated:
        print(path)


def _config_for(exp_id: int, exp_name: str, model_key: str, model: dict[str, Any], backend: str, name: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "models": [model["model_id"]],
        "backends": [backend],
        "backend": backend,
        "output_grid": [64],
        "seeds": [0],
        "tensor_parallel_size": 2,
        "bandwidth_bytes_per_sec": 25_000_000_000,
        "network_latency_sec": 0.00005,
        "resume_overhead_sec": 0.0001,
        "omega_state_sec": 0.00005,
        "omega_text_sec": 0.00005,
        "hf_device": "auto",
        "progress": True,
        "skip_invalid_context": True,
        "output_dir": f"runs/stateful/full_paper/{name}",
    }
    if backend == "sglang":
        common["sglang_server_extra_args"] = (
            "--disable-overlap-schedule --disable-cuda-graph --skip-server-warmup "
            f"--context-length {model['sglang_context']} --max-prefill-tokens {model['sglang_context']}"
        )

    if exp_id == 1:
        common.update(
            {
                "context_grid": model["context_grid"],
                "agent_grid": [16],
                "branch_grid": [8],
                "num_prompts": 32,
            }
        )
    elif exp_id == 2:
        common.update(
            {
                "context_grid": [model["fixed_context"]],
                "agent_grid": [1, 2, 4, 8, 16],
                "branch_grid": [4],
                "num_prompts": 16,
            }
        )
    elif exp_id == 3:
        common.update(
            {
                "context_grid": model["context_grid"],
                "output_tokens": 64,
                "agent_grid": [16],
                "branch_grid": [8],
                "num_prompts": 16,
                "bandwidths": ["10Gbps", "25Gbps", "100Gbps", "200Gbps", "400Gbps"],
                "latencies": ["rdma_10us:10us", "ethernet_100us:100us"],
                "metadata_only": False,
            }
        )
    elif exp_id == 4:
        common.update(
            {
                "context_grid": [model["fixed_context"]],
                "agent_grid": [8],
                "branch_grid": [1, 2, 4, 8, 16],
                "num_prompts": 16,
            }
        )
    elif exp_id == 5:
        common.update(
            {
                "context_grid": [1024, model["fixed_context"]],
                "output_grid": [64, 128],
                "agent_grid": [4, 8, 16],
                "branch_grid": [4],
                "num_prompts": 32,
            }
        )
    elif exp_id == 6:
        common.update(
            {
                "context_grid": [min(model["fixed_context"], 4096)],
                "output_grid": [64],
                "agent_grid": [1],
                "branch_grid": [1],
                "num_prompts": 8,
                "seed": 7,
            }
        )
    return common


def _yaml(payload: dict[str, Any]) -> str:
    lines = ["# Generated by stateful_agentic_algebra/scripts/setup_paper_experiments.py"]
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            text = str(value)
            if any(ch in text for ch in [":", "#"]):
                text = '"' + text.replace('"', '\\"') + '"'
            lines.append(f"{key}: {text}")
    return "\n".join(lines) + "\n"


def _slurm_wrapper(name: str, config_path: Path) -> str:
    rel_config = config_path.relative_to(ROOT)
    return f"""#!/bin/bash
#SBATCH --job-name={name}
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
PROJECT_ROOT="${{PROJECT_ROOT:-${{PRJ_PATH:-/project/bi_dsc_community/drc_rag}}}}"
export CONFIG_PATH="${{CONFIG_PATH:-$PROJECT_ROOT/stateful_agentic_algebra/configs/paper_experiments/{rel_config.name}}}"
exec bash "$PROJECT_ROOT/stateful_agentic_algebra/slurm/run_paper_experiment.sbatch"
"""


if __name__ == "__main__":
    main()
