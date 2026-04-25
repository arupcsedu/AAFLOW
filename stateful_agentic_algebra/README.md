# Stateful Agentic Algebra

This package adds a standalone experimental layer for **Stateful Agentic
Algebra** without rewriting the existing AAFLOW pipeline.

It can reuse AAFLOW components when they are already constructed, but every
heavy dependency is optional. If vLLM, SGLang, UCX, NCCL, CUDA, FAISS, Chroma,
or AAFLOW itself is unavailable, the runtime falls back to deterministic mock
retrieval, mock generation, and simulated KV transfer.

## Paper Mapping

- `state_objects.py`
  - Stateful operator algebra data objects.
  - Explicit `KVState` object with lineage, ownership, memory size, reuse, and
    cost fields.

- `operators.py`
  - Algebra primitives for `materialize`, `transfer`, `fork`,
    `restricted_merge`, `evict`, `retrieve`, and `generate`.

- `compiler.py`
  - Lowers symbolic `OperatorSpec` nodes into an executable plan.

- `scheduler.py`
  - Executes compiled operator plans and leaves lifecycle semantics to the
    operators.

- `kv_manager.py`
  - Implements KV materialize, transfer, fork, restricted merge, and evict.

- `transport.py`
  - Detects optional UCX/NCCL-like transports and falls back to mock transfer.

- `runtime.py`
  - Runtime facade that can wrap AAFLOW agents or use simulation.

- `metrics_stateful.py`
  - Records TTFT, transfer cost, recompute cost, throughput, memory, reuse
    ratio, and framework overhead Omega.

- `workloads.py`, `baselines.py`, `experiment_runner.py`, `plots.py`
  - Synthetic workloads, recompute/reuse baselines, CLI runner, and plotting.

## Smoke Test

```bash
python -c "import stateful_agentic_algebra; print('ok')"
python -m stateful_agentic_algebra.experiment_runner --branches 4
```

