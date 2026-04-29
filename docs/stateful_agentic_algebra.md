# Stateful Agentic Algebra Guide

This guide explains how to run the Stateful Agentic Algebra artifact in the
AAFLOW repository. The default path is CPU/mock-safe and does not require GPU
libraries.

## Overview

Stateful Agentic Algebra adds explicit KV-cache state management to agentic LLM
workflow experiments. It models workflows as stateful graphs with separate data
and state edges:

- Data edge: text, prompts, retrieved passages, summaries, or generated tokens.
- State edge: KV cache state represented by `KVState`.

The core lifecycle is:

1. `kv_materialize`: prefill a prompt and produce a KV state.
2. `kv_fork`: share prefix state across branches.
3. `kv_transfer`: move state between nodes or devices.
4. `kv_merge`: restricted merge for compatible state summaries or segments.
5. `kv_evict`: release state and record eviction metadata.

## Relationship To AAFLOW

The existing AAFLOW retrieval and orchestration benchmarks remain unchanged.
The stateful package lives in `stateful_agentic_algebra/` and can optionally
reuse AAFLOW components through `aaflow_adapter.py`.

Use this module when evaluating:

- KV reuse versus text-passing recomputation.
- Agent branching and tree-of-thought workflows.
- Transfer/recompute crossover under different bandwidth assumptions.
- Real-model KV size and latency measurements.
- Framework overhead `Omega` across baselines.

## Smoke Test

Set local paths once before running commands:

```bash
export PRJ_PATH=/path/to/AAFLOW
export ENV_PATH=/path/to/python/envs
export DATA_PATH=/path/to/scratch_or_data
source stateful_agentic_algebra/env.sh
cd "$PRJ_PATH"
export PYTHONPATH="$PRJ_PATH:${PYTHONPATH:-}"
```

Run from the repository root:

```bash
python -m stateful_agentic_algebra.smoke_test
```

Expected:

- exit code `0`
- directory `runs/stateful/smoke/`
- terminal output `STATEFUL AAFLOW SMOKE TEST PASSED`
- JSON/CSV results and at least one generated plot

The smoke test runs `AAFLOW+` and `dense_prefill` in mock mode, plus
optional baselines when they are available.

## Full Paper Experiments

Mock paper sweep:

```bash
python -m stateful_agentic_algebra.experiment_runner \
  --config stateful_agentic_algebra/configs/full_paper_sweep.yaml
```

Smaller named configs:

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

Each config specifies baselines, workloads, grids, request count, backend type,
and output directory.

## Plotting

Synthetic/mock plots:

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

Synthetic figures include:

- TTFT vs context length
- total latency vs number of agents
- transfer vs recomputation crossover
- KV memory footprint vs branch factor
- throughput vs number of requests
- framework overhead `Omega`
- KV reuse ratio by workload

Real LLM figures include:

- TTFT vs context length grouped by model
- TTFT speedup over dense prefill
- transfer/recompute crossover by model and bandwidth
- KV memory footprint vs context length
- vLLM throughput vs request rate
- TPOT and ITL when available
- consistency exact-match rate by model

## Baselines

| Baseline | Meaning |
| --- | --- |
| `AAFLOW+` | Proposed stateful runtime with explicit KV lifecycle operations. |
| `dense_prefill` | Synthetic dense prefill/text-passing baseline; every agent recomputes the context. |
| `aaflow_text` | Optional AAFLOW text-passing adapter using AAFLOW agents/metrics when importable. |
| `vllm_local_prefix` | Optional local-prefix reuse baseline through vLLM; skipped when vLLM is missing. |
| `sglang_prefix` | Optional SGLang prefix baseline; skipped when SGLang is missing. |
| `kvcomm_prefix` | KVCOMM-style anchor-based cross-context KV reuse baseline. Uses measured-profile simulation unless `KVCOMM_REPO` points to a FastMAS/KVCOMM checkout. |
| `distserve_style` | Simulation of disaggregated prefill/decode. It is not claimed to be exact DistServe. |

List baseline availability:

```bash
python -m stateful_agentic_algebra.experiment_runner --list-baselines
```

## Metrics

| Metric | Meaning |
| --- | --- |
| `ttft_sec` | Time to first token. |
| `total_latency_sec` | End-to-end run latency. |
| `prefill_sec` | Prompt/context prefill or recomputation time. |
| `decode_sec` | Decode time for output tokens. |
| `transfer_sec` | KV state transfer time. |
| `resume_sec` | Cost to resume from KV state. |
| `omega_sec` | Framework overhead `Omega`. |
| `throughput_tokens_per_sec` | Generated-token throughput. |
| `kv_total_bytes` | Total KV state size. |
| `kv_peak_bytes` | Peak KV memory. |
| `kv_transferred_bytes` | Bytes transferred between placements. |
| `kv_reuse_ratio` | Fraction of branch/agent work that reused KV state. |
| `transfer_count` | Number of KV transfers. |
| `materialize_count` | Number of KV materializations. |
| `fork_count` | Number of KV forks. |
| `merge_count` | Number of restricted merges. |
| `evict_count` | Number of KV evictions. |
| `num_agents` | Agent count. |
| `branch_factor` | Branching factor. |
| `context_tokens` | Prompt/context length. |
| `output_tokens` | Generated token count. |
| `baseline_name` | Baseline identifier. |
| `workload_name` | Workload identifier. |
| `run_id` | Unique run row ID. |
| `seed` | Workload/generation seed. |
| `output_agreement_rate` | Text agreement when comparable outputs are available; otherwise null. |

## Expected Outputs

Experiment outputs:

- `results.json`
- `results.csv`
- `config.json`
- `skipped_baselines.json`

Real measurement outputs:

- `metrics.json`
- `kv_metadata.json`
- `generated_dense.txt`
- `generated_cached.txt`
- `crossover.csv`
- `consistency.csv`
- `consistency_summary.json`

Plot outputs:

- `figures/*.png`
- `figures/*.pdf`
- `figures/*.svg` for synthetic plots

## Optional Dependency Troubleshooting

- vLLM missing: vLLM paths are skipped unless `--require-vllm` is passed.
- SGLang missing: `sglang_prefix` is skipped.
- KVCOMM checkout missing: `kvcomm_prefix` still runs as a measured-profile
  baseline; set `KVCOMM_REPO=/path/to/KVCOMM` to record external availability.
- Hugging Face packages missing: mock mode works; install `torch` and
  `transformers` for real HF KV measurement.
- Gated models: request access on Hugging Face and export
  `HUGGINGFACE_HUB_TOKEN`.
- CUDA OOM: reduce context/output/request size or use tensor parallelism.
- CUDA library mismatch: verify PyTorch, CUDA runtime, and cuDNN versions are
  consistent in the active environment.
- UCX/NCCL unavailable: mock or local-file transport still works.
- vLLM KV export unsupported: placeholder methods raise clear
  `NotImplementedError` because stable public APIs vary by vLLM version.

## Real Results After Mock Simulation

The recommended workflow is:

1. Run `python -m stateful_agentic_algebra.smoke_test`.
2. Run the mock paper sweep to validate configs and plotting.
3. Run `hf_kv_backend.py` with `gpt2` or `distilgpt2`.
4. Run real 7B models after GPU allocation and Hugging Face access are ready.
5. Run vLLM serving benchmarks when vLLM is installed.
6. Re-run `plots.py --real-llm` on the real `results.csv`.

Keep `config.json`, logs, raw CSVs, and version information with the generated
figures for reproducibility.

Before downloading large Hugging Face models, move the cache away from home
storage:

```bash
export HF_HOME=$DATA_PATH/huggingface
export HUGGINGFACE_HUB_CACHE=$DATA_PATH/huggingface/hub
export TRANSFORMERS_CACHE=$DATA_PATH/huggingface/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

If these are unset, downloads normally go to:

```text
~/.cache/huggingface/hub/
```

Check cache usage with:

```bash
du -sh "$HF_HOME" 2>/dev/null || du -sh ~/.cache/huggingface 2>/dev/null
```
