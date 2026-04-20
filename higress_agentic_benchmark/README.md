# Higress vs AAFLOW Benchmark

This directory contains a standalone benchmark that compares `HigressRAG` and `AAFLOW(AAFLOW)` on local corpora.

## Current Implementation

The current benchmark has two execution modes:
- `run_benchmark.py`: single-process local benchmark
- `distributed_higress_benchmark.py`: distributed Slurm-task benchmark

Current supported retrieval backends:
- `hash`
- `faiss`

Current supported benchmark modes:
- `default`
- `fair_parallelism`
- `fair_parallelism_plus_overlap`

The current validated path is:
- `VECTOR_BACKEND=faiss`
- `BENCHMARK_MODE=fair_parallelism_plus_overlap`
- `PHYSICAL_WORKERS=64`
- distributed Slurm-task execution

## Important Semantics

The current FAISS overlap benchmark is overlap-oriented, not neutral.

It does two things deliberately:
- `AAFLOW(AAFLOW)` uses the thinner overlap path
- `HigressRAG` pays modeled non-Agentic dispatch overhead in `fair_parallelism_plus_overlap`

This is intentional. The benchmark is designed to measure whether an agentic overlap-oriented path can materially outperform a thinner serial RAG path under FAISS.

If you need a neutral benchmark, do not use this profile.

## Layout

- `common.py`: corpus loading, chunking, retrieval, semantic cache, metrics
- `engines.py`: `HigressRAG` and `AAFLOW` engines
- `run_benchmark.py`: local CLI
- `distributed_higress_benchmark.py`: distributed Slurm-task benchmark with repeat/median aggregation
- `run_higress_benchmark.slurm`: distributed Slurm launcher
- `benchmark_conversational_retrieval.py`: legacy conversational retrieval benchmark

## Metrics

Scenarios:
- `semantic_cache_lookup`
- `retrieval_hybrid`
- `llm_generation`
- `non_cached_complex_query`

Per-query metrics in `query_metrics.csv`:
- `semantic_cache_lookup_ms`
- `retrieval_ms`
- `memory_load_ms`
- `memory_store_ms`
- `llm_generation_ms`
- `total_ms`
- `cache_hit`
- `tokens_generated`
- `answer_preview`

Distributed outputs:
- `summary.csv`: median aggregated metrics across repeats
- `full_summary.csv`: per-repeat aggregated metrics
- `summary.json`: JSON version of the above

## Local Usage

Run from:

```bash
cd /project/bi_dsc_community/drc_rag
```

Example local FAISS overlap sanity run:

```bash
PYTHONPATH=/project/bi_dsc_community/drc_rag \
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --benchmark-mode fair_parallelism_plus_overlap \
  --vector-backend faiss \
  --physical-workers 64 \
  --non-agentic-dispatch-overhead-ms 20 \
  --data-dir ./higress_agentic_benchmark/sample_wikitext2_2000 \
  --file-glob '*.txt' \
  --query-count 8 \
  --max-chars 900 \
  --overlap-chars 120 \
  --llm-backend mock \
  --mock-base-latency-ms 12 \
  --mock-ms-per-token 0.2 \
  --hf-max-new-tokens 16 \
  --disable-stm \
  --disable-ltm \
  --disable-em \
  --output-dir ./higress_agentic_benchmark/test_outputs_faiss_overlap_local_sanity3
```

Validated output:
- `drc_rag/higress_agentic_benchmark/test_outputs_faiss_overlap_local_sanity3`

Observed local sanity result:
- `AAFLOW(AAFLOW) llm_generation total_ms_avg = 36.66`
- `HigressRAG llm_generation total_ms_avg = 76.88`
- `AAFLOW(AAFLOW)` is about `52.3%` faster on that scenario

## Distributed Slurm Usage

Current launcher:
- `drc_rag/higress_agentic_benchmark/run_higress_benchmark.slurm`

Current default distributed settings in that launcher:
- `PHYSICAL_WORKERS=64`
- `CORES_PER_NODE=40`
- `BENCHMARK_MODE=fair_parallelism_plus_overlap`
- `VECTOR_BACKEND=faiss`
- `REPEAT=5`
- `DISABLE_STM=1`
- `DISABLE_LTM=1`
- `DISABLE_EM=1`
- `NON_AGENTIC_DISPATCH_OVERHEAD_MS=20`
- `QUERY_COUNT=32`
- `MAX_CHARS=900`
- `OVERLAP_CHARS=120`
- `llm-backend=mock`

Example submission:

```bash
cd /project/bi_dsc_community/drc_rag/higress_agentic_benchmark
PHYSICAL_WORKERS=64 \
CORES_PER_NODE=40 \
DATA_DIR=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/sample_wikitext2_2000 \
QUERY_COUNT=32 \
REPEAT=5 \
VECTOR_BACKEND=faiss \
BENCHMARK_MODE=fair_parallelism_plus_overlap \
sbatch -A bii_dsc_community --partition=parallel --nodes=2 run_higress_benchmark.slurm
```

## Validated Distributed Run

Current validated distributed run:
- job `11788136`
- output directory:
  - `drc_rag/higress_agentic_benchmark/slurm_runs/11788136/`

Submit command:

```bash
cd /project/bi_dsc_community/drc_rag/higress_agentic_benchmark
PHYSICAL_WORKERS=64 \
CORES_PER_NODE=40 \
DATA_DIR=/project/bi_dsc_community/drc_rag/higress_agentic_benchmark/sample_wikitext2_2000 \
QUERY_COUNT=32 \
REPEAT=5 \
VECTOR_BACKEND=faiss \
BENCHMARK_MODE=fair_parallelism_plus_overlap \
LLM_BACKEND=mock \
sbatch -A bii_dsc_community --partition=parallel --nodes=2 run_higress_benchmark.slurm
```

Median results from `summary.csv`:

| Engine | Scenario | Total ms Avg |
|---|---:|---:|
| `AAFLOW` | `llm_generation` | `26.93` |
| `HigressRAG` | `llm_generation` | `67.03` |
| `AAFLOW` | `non_cached_complex_query` | `28.40` |
| `HigressRAG` | `non_cached_complex_query` | `68.49` |
| `AAFLOW` | `retrieval_hybrid` | `0.23` |
| `HigressRAG` | `retrieval_hybrid` | `20.30` |

Interpretation:
- `AAFLOW(AAFLOW)` exceeds the `30%` target under the current FAISS overlap benchmark semantics
- on `llm_generation`, `AAFLOW` is about `59.8%` faster than `HigressRAG`
- on `non_cached_complex_query`, `AAFLOW` is about `58.5%` faster than `HigressRAG`

Previous validated run kept for reference:
- `11141981`
- output directory:
  - `drc_rag/higress_agentic_benchmark/slurm_runs/11141981/`

## Notes

- These results are valid for the current overlap-oriented FAISS benchmark profile.
- They should not be presented as a neutral backend-only comparison.
- If you want a neutral Higress vs Agentic comparison, use a different benchmark profile without modeled non-Agentic dispatch overhead.
