# Stateful Agentic Algebra

Stateful Agentic Algebra is a standalone experimental layer for evaluating
state reuse in agentic LLM workflows. It lives under
`stateful_agentic_algebra/` and does not rewrite the existing AAFLOW pipeline.

The package can run entirely in CPU/mock mode. Optional integrations with
AAFLOW, Hugging Face Transformers, vLLM, SGLang, KVCOMM, UCX, NCCL, and CUDA
are loaded lazily and skipped or simulated when unavailable.

## What It Implements

The module models agentic execution as a stateful graph:

- `KVState`: an explicit KV-cache state object with model identity, tokenizer
  identity, model config hash, position encoding, lineage, owner node/device,
  KV block metadata, and byte accounting.
- Stateful operators:
  - `kv_materialize`
  - `kv_transfer`
  - `kv_fork`
  - restricted `kv_merge`
  - `kv_evict`
- A compiler that lowers workflow templates into `G_s = (V, E_d, E_s)`,
  separating data edges from state edges.
- A scheduler that decides whether to transfer KV state or recompute from text.
- A runtime that executes compiled mock workflows and records a shared metric
  schema.
- Baselines and plotting tools for paper-style sweeps.

## How It Extends AAFLOW

AAFLOW already benchmarks retrieval and text-passing agentic pipelines. This
module adds a separate state-aware layer:

- Existing AAFLOW behavior is left unchanged.
- `aaflow_adapter.py` optionally imports existing AAFLOW metrics and agent
  components when available.
- `aaflow_text` represents AAFLOW-style text passing.
- `AAFLOW+` represents the proposed stateful path with explicit KV lifecycle
  operations.
- If AAFLOW imports fail, this module still runs in standalone mock mode.

## Environment Setup

Use `stateful_agentic_algebra/env.sh` as the single local path file. New users
should change only these three variables inside that file for a different checkout, environment
root, or scratch/cache location:

```bash
export PRJ_PATH=/raid/${USER}/drc_rag #Change it for your project home directory
source "$PRJ_PATH/stateful_agentic_algebra/env.sh"
cd "$PRJ_PATH"
export PYTHONPATH="$PRJ_PATH:${PYTHONPATH:-}"
```


`env.sh` derives:

- `SAA_VLLM_ENV=$ENV_PATH/saa_vllm_env`
- `SAA_BENCH_ENV=$ENV_PATH/saa_sglang_env` 
- `PYTHON_BIN=$SAA_VLLM_ENV/bin/python`
- `SGLANG_PYTHON_BIN=$SAA_BENCH_ENV/bin/python`
- `PLOT_PYTHON_BIN=$SGLANG_PYTHON_BIN`
- `HF_HOME=$DATA_PATH/huggingface`
- `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`
- `TRANSFORMERS_CACHE=$HF_HOME/transformers`

For a clean third-party setup, create a dedicated SGLang environment named
`saa_sglang_env` and point `SAA_BENCH_ENV` to it which is defined in the `stateful_agentic_algebra/env.sh`.  create `saa_sglang_env` for a cleaner new install for slurm run.

## vLLM/HF Environment: `saa_vllm_env`

Use `saa_vllm_env` for Hugging Face KV measurements and vLLM serving
benchmarks.

Create or recreate it:

```bash
cd "$PRJ_PATH"
python3 -m venv "$ENV_PATH/saa_vllm_env"
source "$ENV_PATH/saa_vllm_env/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r stateful_agentic_algebra/requirements.txt
```

The requirements file is a full `pip freeze --all` snapshot of the working
vLLM/HF stack. It is large because it includes CUDA/PyTorch/vLLM wheels. For
mock-only runs, skip this environment and use the lightweight mock instructions
below.

Verify `saa_vllm_env`:

```bash
"$ENV_PATH/saa_vllm_env/bin/python" - <<'PY'
import importlib.util

for name in ["torch", "transformers", "vllm", "matplotlib"]:
    print(name, "installed" if importlib.util.find_spec(name) else "missing")

import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
```

Current login-shell verification on this machine:

```text
torch: installed
transformers: installed
vllm: installed
matplotlib: missing
torch_version: 2.9.0+cu128
cuda_available: True
cuda_device_count: 5
```

`cuda_available=False` is expected on login shells without a GPU allocation.
Inside an A100/H100 Slurm allocation, CUDA should be visible. Install
`matplotlib` into `saa_vllm_env` only if you want this same interpreter to
generate figures:

```bash
"$ENV_PATH/saa_vllm_env/bin/python" -m pip install matplotlib
```

## SGLang Environment: `saa_sglang_env`

SGLang and vLLM often require different pinned CUDA/PyTorch packages. Keep
SGLang in a separate environment.

Create a clean SGLang environment:

```bash
deactivate #if you are in the saa_vllm_env environment
cd "$PRJ_PATH"
python3 -m venv "$ENV_PATH/saa_sglang_env"
source "$ENV_PATH/saa_sglang_env/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r stateful_agentic_algebra/slang_requirements.txt
```

Then use it with the shared path setup:

```bash
export SAA_BENCH_ENV="$ENV_PATH/saa_sglang_env"
export SGLANG_PYTHON_BIN="$SAA_BENCH_ENV/bin/python"
```


SGLang JIT compilation needs a modern host compiler on this cluster:

```bash
module load gcc/12.4.0 cuda/12.8.0 
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export SGLANG_SERVER_EXTRA_ARGS='--skip-server-warmup'
```

Install gcc/12.4.0 cuda/12.8.0 if you don't have preloaded module.

Verify the SGLang environment:

```bash
"$SGLANG_PYTHON_BIN" - <<'PY'
import importlib.util

for name in ["torch", "transformers", "sglang", "matplotlib"]:
    print(name, "installed" if importlib.util.find_spec(name) else "missing")

import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
```

Current login-shell verification :

```text
torch: installed
transformers: installed
sglang: installed
matplotlib: installed
torch_version: 2.9.1+cu128
cuda_available: False
cuda_device_count: 0
```

Again, CUDA is expected to be unavailable on the login shell and visible inside GPU allocations.

## Cache And Authentication

Keep large Hugging Face downloads out of the home directory:

```bash
source "$PRJ_PATH/stateful_agentic_algebra/env.sh"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Without these variables, Hugging Face typically downloads models under:

```text
~/.cache/huggingface/hub/
```

Gated models require Hugging Face access approval and a token:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

Check cache usage:

```bash
du -sh "$HF_HOME" 2>/dev/null || du -sh ~/.cache/huggingface 2>/dev/null
```

## Mock LLM Tests

Mock mode validates the Stateful Agentic Algebra code without downloading
models or requiring GPUs. It uses deterministic synthetic prompts, simulated KV bytes, and the same CSV/JSON metric schema as real-model runs.

Lightweight mock environment:

```bash
cd "$PRJ_PATH"
python3 -m venv "$ENV_PATH/saa_mock_env"
source "$ENV_PATH/saa_mock_env/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install pytest pyyaml matplotlib
export PYTHONPATH="$PRJ_PATH:${PYTHONPATH:-}"
```

Smoke test:

```bash
python -c "import stateful_agentic_algebra; print('ok')"
python -m stateful_agentic_algebra.smoke_test
```

Expected message:

```text
STATEFUL AAFLOW SMOKE TEST PASSED
```

Small mock sweep:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner \
  --all-baselines \
  --all-workloads \
  --context-grid 1024,4096 \
  --agent-grid 2,4 \
  --branch-grid 2 \
  --output-tokens 64 \
  --num-requests 3 \
  --output-dir runs/stateful/mock_llm_test
```

## Config-Driven Runs

Small config smoke:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/smoke.yaml
```

Full mock paper sweep:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/full_paper_sweep.yaml
```

Real full-paper configs live under:

```text
stateful_agentic_algebra/configs/paper_experiments/
```

## Real LLM Benchmarks

Hugging Face KV microbenchmark:

```bash
"$PYTHON_BIN" -m stateful_agentic_algebra.hf_kv_backend \
  --model-id gpt2 \
  --context-tokens 512 \
  --output-tokens 32 \
  --device auto \
  --output-dir runs/stateful/hf_real_gpt2
```

vLLM serving benchmark:

```bash
"$PYTHON_BIN" -m stateful_agentic_algebra.vllm_benchmark \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --input-len 4096 \
  --output-len 128 \
  --num-prompts 32 \
  --request-rate 4 \
  --tensor-parallel-size 1 \
  --output-dir runs/stateful/vllm_llama3_8b
```

SGLang serving benchmark:

```bash
"$PYTHON_BIN" -m stateful_agentic_algebra.sglang_benchmark \
  --model-id gpt2 \
  --input-len 512 \
  --output-len 32 \
  --num-prompts 8 \
  --tensor-parallel-size 1 \
  --python-bin "$SGLANG_PYTHON_BIN" \
  --output-dir runs/stateful/sglang_gpt2 \
  --extra-args --skip-server-warmup
```

Multi-model runner:

```bash
"$PYTHON_BIN" -m stateful_agentic_algebra.multi_llm_runner \
  --config stateful_agentic_algebra/configs/real_llm_full_paper.yaml
```

## Slurm Runs

The Slurm scripts source `stateful_agentic_algebra/env.sh` and honor
`PYTHON_BIN`, `SGLANG_PYTHON_BIN`, `HF_HOME`, and cache variables.

Single backend/model sweep:

```bash
export MODEL_ID='mistralai/Mistral-7B-Instruct-v0.3'
export BACKEND='hf'
export CONTEXT_GRID='1024,4096,8192'
export OUTPUT_GRID='64'
export NUM_PROMPTS='4'
export TENSOR_PARALLEL_SIZE='2'
export OUTPUT_DIR='runs/stateful/manual_hf_mistral'

sbatch --partition=bii-gpu --reservation=bi_fox_dgx --export=ALL \
  stateful_agentic_algebra/slurm/run_real_llm_sweep.sbatch
```

For comma-separated grids, export variables first and use `--export=ALL`; do
not put comma-separated values directly inside `sbatch --export=...`.

For SGLang on this cluster, use A100/H100 nodes. V100 nodes are below SGLang's
current minimum compute capability.

## Plot Generation

Mock/synthetic plots:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
  --results runs/stateful/full_paper_sweep/results.csv \
  --output-dir runs/stateful/full_paper_sweep/figures
```

Real LLM plots:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
  --results runs/stateful/real_llm_full/results.csv \
  --output-dir runs/stateful/real_llm_full/figures \
  --real-llm
```

Generate only selected real-LLM plots:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.plots \
  --results runs/stateful/full_paper/exp2_multi_agent_scaling_mistral_hf/results.csv \
  --output-dir runs/stateful/full_paper/exp2_multi_agent_scaling_mistral_hf/figures \
  --real-llm \
  --plot-names real_speedup_vs_agents
```

Figures are saved as PNG, PDF, and SVG.

## Baselines

- `AAFLOW+`: proposed Stateful Agentic Algebra runtime with explicit KV
  materialize, transfer, fork, merge, and evict operations.
- `dense_prefill`: text-passing baseline where every agent independently pays
  context prefill and has no KV reuse.
- `aaflow_text`: optional AAFLOW text baseline using AAFLOW imports when
  available.
- `vllm_local_prefix`: optional vLLM/local-prefix reuse baseline.
- `sglang_prefix`: optional SGLang prefix reuse baseline.
- `kvcomm_prefix`: KVCOMM-style anchor-based cross-context KV reuse baseline.
- `distserve_style`: simulated disaggregated prefill/decode baseline, not an
  exact DistServe implementation.

List baselines:

```bash
"$PLOT_PYTHON_BIN" -m stateful_agentic_algebra.experiment_runner --list-baselines
```

## Metrics

All experiment rows use a common schema:

- `ttft_sec`: time to first token.
- `total_latency_sec`: end-to-end latency.
- `prefill_sec`: context prefill or recomputation cost.
- `decode_sec`: decode cost for generated tokens.
- `transfer_sec`: simulated or measured KV transfer time.
- `resume_sec`: cost to resume decode from existing KV state.
- `omega_sec`: framework overhead `Omega`.
- `throughput_tokens_per_sec`: generated-token throughput.
- `kv_total_bytes`: total KV state size.
- `kv_peak_bytes`: peak KV memory footprint.
- `kv_transferred_bytes`: bytes moved between nodes/devices.
- `kv_reuse_ratio`: fraction of execution that reused KV state.
- `transfer_count`: number of KV transfers.
- `materialize_count`: number of KV materializations.
- `fork_count`: number of KV forks.
- `merge_count`: number of restricted KV merges.
- `evict_count`: number of KV evictions.
- `num_agents`: number of agents in the workload.
- `branch_factor`: branching factor for tree/debate workloads.
- `context_tokens`: prompt/context length.
- `output_tokens`: generated token count.
- `baseline_name`: baseline identifier.
- `workload_name`: workload identifier.
- `run_id`: unique run row identifier.
- `seed`: deterministic workload seed.
- `output_agreement_rate`: agreement rate when comparable text outputs exist;
  otherwise null.

## Expected Output Files

Experiment runner:

- `results.json`
- `results.csv`
- `config.json`
- `skipped_baselines.json`
- `benchmark.out`
- `summary.out`

Real model tools:

- `metrics.json`
- `kv_metadata.json`
- `generated_dense.txt`
- `generated_cached.txt`
- `crossover.csv`
- `crossover.json`
- `consistency.csv`
- `consistency_summary.json`

Plotting:

- `figures/*.png`
- `figures/*.pdf`
- `figures/*.svg`

## Troubleshooting

- Missing vLLM: vLLM benchmarks are skipped unless `--require-vllm` is used.
- Missing SGLang: `sglang_prefix` falls back to simulated prefix metrics, and
  the real `sglang` backend is skipped unless SGLang is available through
  `SGLANG_PYTHON_BIN`.
- Missing KVCOMM checkout: `kvcomm_prefix` still runs as a measured-profile
  simulation. Set `KVCOMM_REPO=/path/to/KVCOMM` to record an external checkout.
- Gated model access: request access on Hugging Face and export
  `HUGGINGFACE_HUB_TOKEN`.
- Hugging Face cache fills home storage: set `HF_HOME`,
  `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to scratch before
  downloading large models.
- CUDA OOM: reduce context length/output length/request count, use a smaller
  model, or increase tensor parallelism.
- CUDA not visible on login shell: run inside a Slurm GPU allocation and check
  `nvidia-smi`.
- UCX/NCCL unavailable: transport falls back to mock/local simulation.
- Unsupported vLLM KV export: the wrapper raises `NotImplementedError` for
  unstable KV export/import APIs rather than failing at import time.

## File Map

- `state_objects.py`: KV state and compatibility objects.
- `operators.py`: stateful KV lifecycle operators.
- `compiler.py`: stateful execution graph construction.
- `scheduler.py`: transfer-vs-recompute decisions.
- `transport.py`: mock, local-file, and optional UCX transport APIs.
- `runtime.py`: mock runtime for stateful workflows.
- `metrics_stateful.py`: metric recorder and CSV/JSON aggregation.
- `workloads.py`: deterministic synthetic workloads.
- `baselines.py`: baseline adapters.
- `experiment_runner.py`: config and CLI runner.
- `hf_kv_backend.py`: Hugging Face KV measurement backend.
- `vllm_benchmark.py`: vLLM server/bench wrapper.
- `sglang_benchmark.py`: SGLang server/bench wrapper.
- `multi_llm_runner.py`: real-model benchmark matrix.
- `transfer_crossover_real.py`: KV transfer/recompute crossover analysis.
- `consistency_benchmark.py`: dense-vs-cached consistency measurement.
- `plots.py`: publication figure generation.
