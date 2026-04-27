# Artifact Evaluation: Stateful Agentic Algebra

This artifact evaluates Stateful Agentic Algebra, a standalone extension under
`stateful_agentic_algebra/` for KV-state-aware agentic workflow experiments.
The artifact is mock-safe by default: reviewers can run the smoke test and
mock paper sweep without GPU-only dependencies.

## Installation

Clone the repository and enter the repo root:

```bash
git clone https://github.com/arupcsedu/AAFLOW.git
cd AAFLOW
```

Use any Python 3.10+ environment with `matplotlib` and `pytest` for the mock
artifact path. Mock experiments do not require CUDA, vLLM, SGLang, or
Hugging Face model downloads.

```bash
python -m pip install matplotlib pytest
```

For the validated UVA cluster setup, use the checked-out repo and the split
Python environments below. The split is intentional: the vLLM stack and the
SGLang stack have conflicting CUDA/PyTorch dependency constraints.

```bash
export PROJECT_ROOT=/project/bi_dsc_community/drc_rag
cd "$PROJECT_ROOT"

# Main runner, HF backend, and vLLM backend.
export PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python

# SGLang serving backend.
export SGLANG_PYTHON_BIN=/scratch/djy8hg/env/drc_rag_bench_env/bin/python

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
```

Validate the environments before launching a long Slurm job:

```bash
$PYTHON_BIN -c "import torch, transformers; print('main ok', torch.__version__)"
$PYTHON_BIN -c "import vllm; print('vllm ok')"
$SGLANG_PYTHON_BIN -c "import torch, sglang; print('sglang ok', torch.__version__)"
```

Set Hugging Face cache directories to scratch before downloading large models.
If these variables are not set, model weights usually download under
`~/.cache/huggingface/hub/`, which may fill home storage.

```bash
export HF_HOME=/scratch/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Gated Hugging Face models require access approval and a token:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
```

For SGLang on this cluster, load a modern compiler and CUDA module before
starting the server. Without GCC 12, SGLang JIT compilation can fail with
`fatal error: version: No such file or directory`.

```bash
module load gcc/12.4.0 cuda/12.8.0
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export SGLANG_SERVER_EXTRA_ARGS='--disable-overlap-schedule --disable-cuda-graph'
```

## Smoke Test

Run:

```bash
python -m stateful_agentic_algebra.smoke_test
```

Expected result:

```text
STATEFUL AAFLOW SMOKE TEST PASSED
```

Expected runtime:

- CPU/mock smoke: usually under one minute.
- Mock config sweeps: seconds to a few minutes depending on grid size.
- HF `gpt2` smoke: minutes on CPU, faster on GPU.
- 7B+ real-model runs: depends on model, context length, GPU memory, and
  tensor parallelism.

The smoke test creates:

```text
runs/stateful/smoke/
  config.json
  results.csv
  results.json
  skipped_baselines.json
  figures/
```

## What The Artifact Implements

Stateful Agentic Algebra models agentic workflows with explicit state:

- `KVState` and `KVBlock` metadata.
- KV materialize, transfer, fork, restricted merge, and evict operators.
- A compiler that emits stateful execution graphs.
- A scheduler for transfer-versus-recompute decisions.
- A mock runtime and optional HF/vLLM measurement tools.
- Baselines with a shared metric schema.

It extends AAFLOW without modifying the existing AAFLOW pipeline. Optional
AAFLOW imports are handled through `aaflow_adapter.py`; failed imports fall
back to standalone operation.

## Mock Paper Reproduction

Run the full mock sweep:

```bash
python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/full_paper_sweep.yaml
```

Smaller configs are available for quicker checks:

```bash
python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/smoke.yaml

python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/ttft_context_sweep.yaml

python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/multi_agent_scaling.yaml

python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/transfer_recompute_crossover.yaml

python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/memory_efficiency.yaml

python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/throughput_overhead.yaml
```

Expected output files:

- `results.json`
- `results.csv`
- `config.json`
- `skipped_baselines.json`

## Reproduce Figures

Synthetic/mock figures:

```bash
python -m stateful_agentic_algebra.plots \
  --results runs/stateful/full_paper_sweep/results.csv \
  --output-dir runs/stateful/full_paper_sweep/figures
```

Expected figure files:

- `figures/*.png`
- `figures/*.pdf`
- `figures/*.svg`

Real LLM figures:

```bash
python -m stateful_agentic_algebra.plots \
  --results runs/stateful/real_llm_full/results.csv \
  --output-dir runs/stateful/real_llm_full/figures \
  --real-llm
```

Expected real-LLM figure files:

- `real_ttft_vs_context_by_model.{png,pdf}`
- `real_ttft_speedup_over_dense.{png,pdf}`
- `real_transfer_recompute_crossover.{png,pdf}`
- `real_kv_memory_vs_context.{png,pdf}`
- `real_vllm_throughput_vs_request_rate.{png,pdf}`
- `real_tpot_vs_context.{png,pdf}`
- `real_itl_vs_context.{png,pdf}`
- `real_consistency_exact_match_by_model.{png,pdf}`

## Real Large-LLM Benchmarks

### Hardware Requirements

- Small smoke: CPU or GPU with `gpt2` or `distilgpt2`.
- 7B models: at least one A100 80GB or H100 is recommended.
- Larger models: tensor parallelism across multiple GPUs is recommended.
- On this cluster, request A100 through the `gpu` partition, for example:
  `sbatch -p gpu --gres=gpu:a100:1 --export=ALL stateful_agentic_algebra/slurm/run_real_llm_sweep.sbatch`.
  SGLang serving does not run on V100 here because current SGLang requires
  compute capability sm75 or newer.

For a small real-framework validation with non-skipped rows, use a small model
and short context first:

```bash
export PROJECT_ROOT=/project/bi_dsc_community/drc_rag
export PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python
export SGLANG_PYTHON_BIN=/scratch/djy8hg/env/drc_rag_bench_env/bin/python
export MODEL_ID='gpt2,distilgpt2'
export BACKEND='hf,vllm,sglang'
export CONTEXT_GRID='16'
export OUTPUT_GRID='4'
export AGENT_GRID='2'
export BRANCH_GRID='2'
export NUM_PROMPTS='2'
export TENSOR_PARALLEL_SIZE='1'
export OUTPUT_DIR="$PROJECT_ROOT/runs/stateful/real_llm_multibackend_test"

sbatch -p gpu --gres=gpu:a100:1 --export=ALL \
  stateful_agentic_algebra/slurm/run_real_llm_sweep.sbatch
```

This writes `benchmark.out` in the output directory with a compact table of
TTFT, total latency, throughput, and status for each backend/model row.

### Authentication And Cache Location

Gated Hugging Face models require access approval and:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

`HF_TOKEN` may also be used by Hugging Face tooling.

Set Hugging Face cache directories to scratch before downloading large models.
If these variables are not set, model weights usually download under
`~/.cache/huggingface/hub/`, which may fill home storage.

```bash
export HF_HOME=/scratch/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Check usage:

```bash
du -sh "$HF_HOME" 2>/dev/null || du -sh ~/.cache/huggingface 2>/dev/null
```

### HF KV Microbenchmark

```bash
python -m stateful_agentic_algebra.hf_kv_backend \
  --model-id gpt2 \
  --context-tokens 512 \
  --output-tokens 32 \
  --device auto \
  --output-dir runs/stateful/hf_real_gpt2
```

Expected outputs:

- `metrics.json`
- `kv_metadata.json`
- `generated_dense.txt`
- `generated_cached.txt`

### vLLM Serving Benchmark

```bash
python -m stateful_agentic_algebra.vllm_benchmark \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --input-len 4096 \
  --output-len 128 \
  --num-prompts 32 \
  --request-rate 4 \
  --tensor-parallel-size 1 \
  --output-dir runs/stateful/vllm_llama3_8b
```

If vLLM is missing, the command exits cleanly unless `--require-vllm` is used.

### SGLang Serving Benchmark

The multi-model runner launches SGLang through `SGLANG_PYTHON_BIN`. On this
cluster, use the SGLang env and server flags from the installation section:

```bash
export SGLANG_PYTHON_BIN=/scratch/djy8hg/env/drc_rag_bench_env/bin/python
export SGLANG_SERVER_EXTRA_ARGS='--disable-overlap-schedule --disable-cuda-graph'
```

If SGLang's packaged `bench_serving` command fails because optional benchmark
dependencies such as `datasets` are not installed, the wrapper falls back to
the OpenAI-compatible HTTP endpoint exposed by the live SGLang server and still
records real serving latency.

### Multi-Model Sweep

```bash
python -m stateful_agentic_algebra.multi_llm_runner \
  --config stateful_agentic_algebra/configs/real_llm_full_paper.yaml
```

Expected outputs:

- `results_raw.jsonl`
- `results.csv`
- `summary_by_model.csv`
- `benchmark.out`
- `logs/`
- `hf_measurements/`
- `vllm_runs/`
- `sglang_runs/`
- per-model HF/vLLM/SGLang artifacts when available

### Transfer/Recompute Crossover

```bash
python -m stateful_agentic_algebra.transfer_crossover_real \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --context-grid 1024,4096,8192,16384,32768 \
  --output-dir runs/stateful/transfer_crossover_real
```

Expected outputs:

- `crossover.csv`
- `crossover.json`
- `plot_transfer_vs_recompute.png`
- `plot_transfer_vs_recompute.pdf`

### Consistency Benchmark

```bash
python -m stateful_agentic_algebra.consistency_benchmark \
  --model-id gpt2 \
  --context-tokens 512 \
  --output-tokens 32 \
  --num-prompts 8 \
  --output-dir runs/stateful/consistency_gpt2
```

Expected outputs:

- `consistency.csv`
- `consistency_summary.json`

## Add Real Results After Mock Simulation

Recommended workflow:

1. Run `python -m stateful_agentic_algebra.smoke_test`.
2. Run the full mock sweep and generate synthetic figures.
3. Run HF KV microbenchmarks on `gpt2` to validate the real backend.
4. After GPU allocation and model access are ready, run 7B+ HF or vLLM sweeps.
5. Use `multi_llm_runner.py` to aggregate rows into `results.csv`.
6. Regenerate plots with `--real-llm`.
7. Archive `config.json`, raw CSVs, logs, and version information with figures.

## Baselines

- `AAFLOW+`: proposed stateful KV runtime.
- `dense_prefill`: synthetic text-passing baseline with no KV reuse.
- `aaflow_text`: optional AAFLOW text baseline.
- `vllm_local_prefix`: optional vLLM local-prefix baseline.
- `sglang_prefix`: optional SGLang prefix baseline.
- `distserve_style`: simulated disaggregated prefill/decode baseline.

## Metrics

The primary output metrics are:

- `ttft_sec`
- `total_latency_sec`
- `prefill_sec`
- `decode_sec`
- `transfer_sec`
- `resume_sec`
- `omega_sec`
- `throughput_tokens_per_sec`
- `kv_total_bytes`
- `kv_peak_bytes`
- `kv_transferred_bytes`
- `kv_reuse_ratio`
- lifecycle counts: `transfer_count`, `materialize_count`, `fork_count`,
  `merge_count`, `evict_count`
- workload fields: `num_agents`, `branch_factor`, `context_tokens`,
  `output_tokens`, `baseline_name`, `workload_name`, `run_id`, `seed`
- `output_agreement_rate` when comparable text outputs are available

## Troubleshooting

- CUDA OOM: reduce context length, output length, request count, or use a
  smaller model/tensor parallelism.
- Missing vLLM: install vLLM in a GPU-compatible environment or skip vLLM rows.
- vLLM/SGLang dependency conflicts: keep the main/vLLM environment and SGLang
  environment separate, then pass `PYTHON_BIN` and `SGLANG_PYTHON_BIN` to the
  Slurm script.
- SGLang on V100: current SGLang requires compute capability sm75 or newer on
  this setup. Request A100 with `-p gpu --gres=gpu:a100:1`.
- SGLang JIT compile error mentioning `<version>`: load `gcc/12.4.0` and
  `cuda/12.8.0`, then set `CC` and `CXX` before launching SGLang.
- Gated model access: request access on Hugging Face and set
  `HUGGINGFACE_HUB_TOKEN`.
- Hugging Face cache fills home storage: set `HF_HOME`,
  `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to a scratch path before
  model download.
- Tokenizer mismatch: use the tokenizer paired with the model.
- Unsupported KV export: vLLM KV export/import is intentionally a placeholder
  when no stable public API is available.
- CUDA library mismatch: verify driver, CUDA runtime, cuDNN, PyTorch, and vLLM
  versions.

## Reproducibility Checklist

Report with real results:

- GPU type and count
- NVIDIA driver
- CUDA version
- PyTorch version
- vLLM version, if used
- model ID and revision, if pinned
- config file
- random seed
- whether deterministic decoding was used
