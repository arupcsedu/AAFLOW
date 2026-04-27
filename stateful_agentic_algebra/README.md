# Stateful Agentic Algebra

Stateful Agentic Algebra is a standalone experimental layer for evaluating
state reuse in agentic LLM workflows. It is implemented under
`stateful_agentic_algebra/` and does not rewrite the existing AAFLOW pipeline.

The package can run entirely in mock CPU mode. Optional integrations with
AAFLOW, Hugging Face Transformers, vLLM, SGLang, UCX, NCCL, and CUDA are loaded
lazily and skipped or simulated when unavailable.

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
- A compiler that lowers workflow templates into
  `G_s = (V, E_d, E_s)`, separating data edges from state edges.
- A scheduler that decides whether to transfer KV state or recompute from text.
- A runtime that executes compiled mock workflows and records a shared metric
  schema.
- Baselines and plotting tools for paper-style sweeps.

## How It Extends AAFLOW

AAFLOW already benchmarks retrieval and text-passing agentic pipelines. This
module adds a separate state-aware layer on top:

- Existing AAFLOW behavior is left unchanged.
- `aaflow_adapter.py` optionally imports existing AAFLOW metrics and agent
  components when available.
- `AAFLOWTextBaseline` represents AAFLOW-style text passing, while
  `AAFLOW+` introduces explicit KV lifecycle operations.
- If AAFLOW imports fail, the module remains usable in standalone mock mode.

## Quick Start

From the repository root:

```bash
python -c "import stateful_agentic_algebra; print('ok')"
python -m stateful_agentic_algebra.smoke_test
```

The smoke test is CPU/mock-safe and should not require GPU-only dependencies.
It writes outputs under:

```text
runs/stateful/smoke/
```

Expected terminal message:

```text
STATEFUL AAFLOW SMOKE TEST PASSED
```

## Mock LLM Tests

Mock mode is the default path for validating the Stateful Agentic Algebra code
without downloading models or requiring GPUs. It uses deterministic synthetic
token counts, simulated KV bytes, and the same metric schema used by real-model
runs.

Use the project benchmark environment:

```bash
cd /project/bi_dsc_community/drc_rag
export PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

On this cluster, CUDA Python wheels may need their packaged NVIDIA libraries in
`LD_LIBRARY_PATH`, even for import checks:

```bash
export LD_LIBRARY_PATH="$($PYTHON_BIN - <<'PY'
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages()]
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass
print(":".join(str(p) for root in roots for p in root.glob("nvidia/*/lib") if p.is_dir()))
PY
):${LD_LIBRARY_PATH:-}"
```

Run the smoke test:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.smoke_test
```

Run all mock baselines on a small grid:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.experiment_runner \
  --all-baselines \
  --all-workloads \
  --context-grid 1024,4096 \
  --agent-grid 2,4 \
  --branch-grid 2 \
  --output-tokens 64 \
  --num-requests 3 \
  --output-dir runs/stateful/mock_llm_test
```

Expected files:

```text
runs/stateful/mock_llm_test/results.csv
runs/stateful/mock_llm_test/results.json
runs/stateful/mock_llm_test/config.json
runs/stateful/mock_llm_test/skipped_baselines.json
runs/stateful/mock_llm_test/benchmark.out
```

The main mock paper-style run used in this repository is:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.experiment_runner \
  --all-baselines \
  --all-workloads \
  --context-grid 1024,4096,8192 \
  --agent-grid 2,4,8 \
  --branch-grid 2,4 \
  --output-tokens 64 \
  --num-requests 5 \
  --output-dir runs/stateful/all_baselines_mock
```

## Config-Driven Runs

Small config smoke:

```bash
python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/smoke.yaml
```

Full mock paper sweep:

```bash
python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/full_paper_sweep.yaml
```

The config files define:

- `baselines`
- `workloads`
- `context_grid`
- `output_grid`
- `agent_grid`
- `branch_grid`
- `num_requests`
- `backend_type`
- `output_dir`

## Real LLM Runs

The current real-LLM benchmark environment is:

```bash
export PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python
```

`/scratch/djy8hg/env/saa_vllm_env/bin/python` is the Python used by the Slurm
scripts. It currently points to the benchmark environment under
`/scratch/djy8hg/env/drc_rag_bench_env`, so use `PYTHON_BIN` rather than plain
`python` in shell commands and batch scripts.

Recommended shell setup:

```bash
cd /project/bi_dsc_community/drc_rag
export PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
```

Add packaged NVIDIA wheel libraries:

```bash
export LD_LIBRARY_PATH="$($PYTHON_BIN - <<'PY'
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages()]
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass
print(":".join(str(p) for root in roots for p in root.glob("nvidia/*/lib") if p.is_dir()))
PY
):${LD_LIBRARY_PATH:-}"
```

Verify imports:

```bash
$PYTHON_BIN - <<'PY'
import importlib
for name in ["torch", "transformers", "vllm", "sglang"]:
    try:
        mod = importlib.import_module(name)
        print(name, "ok", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "failed", type(exc).__name__, exc)
PY
```

Install or refresh the environment only from a GPU/login shell where large wheel
downloads are acceptable:

```bash
$PYTHON_BIN -m pip install -U pip
$PYTHON_BIN -m pip install pytest
$PYTHON_BIN -m pip install vllm
$PYTHON_BIN -m pip install sglang
```

SGLang and vLLM can require different pinned versions of `torch`,
`transformers`, FlashInfer, and related CUDA packages. For production real-LLM
serving sweeps, separate vLLM-only and SGLang-only environments are cleaner.
The mock experiments do not depend on either package importing successfully.

Hugging Face KV microbenchmark:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.hf_kv_backend \
  --model-id gpt2 \
  --context-tokens 512 \
  --output-tokens 32 \
  --device auto \
  --output-dir runs/stateful/hf_real_gpt2
```

vLLM serving benchmark:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.vllm_benchmark \
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
export SGLANG_PYTHON_BIN=/scratch/djy8hg/env/drc_rag_bench_env/bin/python
$PYTHON_BIN -m stateful_agentic_algebra.sglang_benchmark \
  --model-id gpt2 \
  --input-len 512 \
  --output-len 32 \
  --num-prompts 8 \
  --tensor-parallel-size 1 \
  --python-bin "$SGLANG_PYTHON_BIN" \
  --output-dir runs/stateful/sglang_gpt2
```

SGLang and vLLM often need different pinned Python packages. Keep using
`PYTHON_BIN=/scratch/djy8hg/env/saa_vllm_env/bin/python` for the main runner and
set `SGLANG_PYTHON_BIN` when SGLang is installed in a separate environment.

Multi-model sweep:

```bash
$PYTHON_BIN -m stateful_agentic_algebra.multi_llm_runner \
  --config stateful_agentic_algebra/configs/real_llm_full_paper.yaml
```

Slurm multi-model sweep:

```bash
export MODEL_ID='gpt2'
export BACKEND='hf'
export CONTEXT_GRID='512'
export OUTPUT_GRID='64'
export NUM_PROMPTS='4'
sbatch -p gpu --gres=gpu:a100:1 --export=ALL \
  stateful_agentic_algebra/slurm/run_real_llm_sweep.sbatch
```

For comma-separated grids, always export variables first so Slurm does not split
`--export` values on commas:

```bash
export MODEL_ID='gpt2,mistralai/Mistral-7B-Instruct-v0.3'
export BACKEND='hf,vllm,sglang'
export SGLANG_PYTHON_BIN=/scratch/djy8hg/env/drc_rag_bench_env/bin/python
export CONTEXT_GRID='512,960'
export OUTPUT_GRID='64'
export NUM_PROMPTS='8'
sbatch -p gpu --gres=gpu:a100:1 --export=ALL \
  stateful_agentic_algebra/slurm/run_real_llm_sweep.sbatch
```

For SGLang on this cluster, use the A100 `gpu` partition. V100 nodes are below
SGLang's current minimum compute capability, and SGLang's JIT kernels need a
newer host compiler. The sweep script loads `gcc/12.4.0` and `cuda/12.8.0` by
default and passes `--disable-overlap-schedule --disable-cuda-graph` to avoid
the JIT paths that fail under the system GCC 8 toolchain.

Gated Hugging Face models require access approval and:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

Large model weights should be cached on scratch or another high-capacity
filesystem instead of the default home-directory cache. Set these variables in
the same shell before running HF or vLLM benchmarks:

```bash
export HF_HOME=/scratch/$USER/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Without these variables, Hugging Face typically downloads models under:

```text
~/.cache/huggingface/hub/
```

Check current cache usage with:

```bash
du -sh "$HF_HOME" 2>/dev/null || du -sh ~/.cache/huggingface 2>/dev/null
```

## Plot Generation

Mock/synthetic paper plots:

```bash
python -m stateful_agentic_algebra.plots \
  --results runs/stateful/full_paper_sweep/results.csv \
  --output-dir runs/stateful/full_paper_sweep/figures
```

Real LLM plots:

```bash
python -m stateful_agentic_algebra.plots \
  --results runs/stateful/real_llm_full/results.csv \
  --output-dir runs/stateful/real_llm_full/figures \
  --real-llm
```

Synthetic plots are saved as SVG, PNG, and PDF. Real LLM plots are saved as PNG
and PDF.

## Baselines

- `AAFLOW+`: the proposed Stateful Agentic Algebra runtime with explicit
  KV materialize, transfer, fork, merge, and evict operations.
- `dense_prefill`: synthetic dense prefill/text-passing baseline where every
  agent independently pays context prefill and has no KV reuse.
- `aaflow_text`: optional AAFLOW text baseline that reuses AAFLOW RagAgent or
  LLM generator paths when importable; otherwise it falls back or skips.
- `vllm_local_prefix`: optional vLLM/local-prefix baseline. If vLLM is missing,
  it is skipped or simulated depending on the runner path.
- `sglang_prefix`: optional SGLang prefix baseline. If SGLang is installed, it
  is labeled as SGLang-backed; if not, the runner emits simulated SGLang-prefix
  metrics with a clear reason instead of dropping all rows.
- `distserve_style`: simulated disaggregated prefill/decode baseline. It is
  labeled as DistServe-style simulation, not an exact DistServe implementation.

List available baselines:

```bash
python -m stateful_agentic_algebra.experiment_runner --list-baselines
```

## Metrics

All experiment rows use a common schema:

- `ttft_sec`: time to first token.
- `total_latency_sec`: end-to-end latency for the run.
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
- `figures/*.svg` for synthetic plots

## Troubleshooting Optional Dependencies

- Missing vLLM: vLLM benchmarks are skipped unless `--require-vllm` is used.
- Missing SGLang: `sglang_prefix` falls back to simulated prefix metrics, and
  the real `sglang` backend is skipped unless SGLang is available through
  `SGLANG_PYTHON_BIN` or the active Python environment.
- Missing Hugging Face packages: mock mode still works; install
  `transformers`, `torch`, and `huggingface_hub` for HF runs.
- Gated model access: request access on Hugging Face and export
  `HUGGINGFACE_HUB_TOKEN`.
- Hugging Face cache fills home storage: set `HF_HOME`,
  `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to a scratch path before
  downloading large models.
- CUDA OOM: reduce context length/output length/request count, use a smaller
  model, or increase tensor parallelism.
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
