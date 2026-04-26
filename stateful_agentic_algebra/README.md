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
  `ours_stateful` introduces explicit KV lifecycle operations.
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

Hugging Face KV microbenchmark:

```bash
python -m stateful_agentic_algebra.hf_kv_backend \
  --model-id gpt2 \
  --context-tokens 512 \
  --output-tokens 32 \
  --device auto \
  --output-dir runs/stateful/hf_real_gpt2
```

vLLM serving benchmark:

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

Multi-model sweep:

```bash
python -m stateful_agentic_algebra.multi_llm_runner \
  --config stateful_agentic_algebra/configs/real_llm_full_paper.yaml
```

Gated Hugging Face models require access approval and:

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
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

- `ours_stateful`: the proposed Stateful Agentic Algebra runtime with explicit
  KV materialize, transfer, fork, merge, and evict operations.
- `dense_prefill`: synthetic dense prefill/text-passing baseline where every
  agent independently pays context prefill and has no KV reuse.
- `aaflow_text`: optional AAFLOW text baseline that reuses AAFLOW RagAgent or
  LLM generator paths when importable; otherwise it falls back or skips.
- `vllm_local_prefix`: optional vLLM/local-prefix baseline. If vLLM is missing,
  it is skipped or simulated depending on the runner path.
- `sglang_prefix`: optional SGLang prefix baseline. If SGLang is missing, it is
  skipped with a clear reason.
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
- Missing SGLang: `sglang_prefix` is skipped.
- Missing Hugging Face packages: mock mode still works; install
  `transformers`, `torch`, and `huggingface_hub` for HF runs.
- Gated model access: request access on Hugging Face and export
  `HUGGINGFACE_HUB_TOKEN`.
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
- `multi_llm_runner.py`: real-model benchmark matrix.
- `transfer_crossover_real.py`: KV transfer/recompute crossover analysis.
- `consistency_benchmark.py`: dense-vs-cached consistency measurement.
- `plots.py`: publication figure generation.
