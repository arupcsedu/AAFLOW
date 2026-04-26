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
artifact path. The existing benchmark environment can also be used:

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
cd /project/bi_dsc_community/drc_rag
```

Minimal packages for mock mode:

```bash
python -m pip install matplotlib pytest
```

Optional packages for real-model runs:

```bash
python -m pip install torch transformers huggingface_hub
```

Optional vLLM serving benchmarks require a compatible CUDA/PyTorch/vLLM stack.
Install and validate vLLM in a separate GPU environment when needed.

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

### Authentication

Gated Hugging Face models require access approval and:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

`HF_TOKEN` may also be used by Hugging Face tooling.

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

### Multi-Model Sweep

```bash
python -m stateful_agentic_algebra.multi_llm_runner \
  --config stateful_agentic_algebra/configs/real_llm_full_paper.yaml
```

Expected outputs:

- `results_raw.jsonl`
- `results.csv`
- `summary_by_model.csv`
- per-model HF/vLLM artifacts when available

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

- `ours_stateful`: proposed stateful KV runtime.
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
- Gated model access: request access on Hugging Face and set
  `HUGGINGFACE_HUB_TOKEN`.
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
