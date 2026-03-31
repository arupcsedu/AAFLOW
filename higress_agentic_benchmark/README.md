# Higress vs Agentic RAG Benchmark

This directory contains a standalone benchmark that compares a Higress-style RAG path against an Agentic RAG path built from the local `drc_rag` codebase.

## Scope

What is modeled:
- `HigressRAG`
  - semantic cache lookup
  - hybrid retrieval
  - LLM generation
  - non-cached complex query
- `AgenticRAG`
  - the same semantic cache and hybrid retrieval features
  - hierarchical memory on top of retrieval
  - explicit memory load and memory store timing

What is not modeled by default:
- a real deployed Higress gateway
- real network hops through an API gateway
- distributed serving infrastructure

This is a local benchmark harness aligned to official Higress feature concepts.

## Official References

- Higress repo: `https://github.com/alibaba/higress`
- Higress AI cache docs: `https://higress.cn/en-us/docs/plugins/ai/ai-cache`
- Higress docs home: `https://higress.cn/en-us/docs/overview/what-is-higress`

## Layout

- `run_benchmark.py`: CLI entry point
- `common.py`: corpus loading, chunking, hybrid retrieval, semantic cache, metrics export, local LLM backends
- `engines.py`: `HigressRAG` and `AgenticRAG` benchmark engines
- `benchmark_conversational_retrieval.py`: conversational retrieval accuracy benchmark for `HigressRAG` vs `AgenticDRC`
- `agentic_drc_vs_higressrag.tex`: IEEE-style paper summary of the current latency and retrieval-quality results

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

Aggregated metrics in `summary.csv`:
- average of all metrics above
- `cache_hit_rate`
- `total_ms_p50`
- `total_ms_p95`

Interpretation:
- `retrieval_ms` includes hybrid retrieval and context assembly
- `memory_load_ms` is the Agentic-only memory lookup cost
- `memory_store_ms` is the Agentic-only post-answer memory write/update cost
- `HigressRAG` reports `0` for memory-specific fields

## Environment

Recommended Python:
- `/scratch/djy8hg/env/drc_rag_bench_env/bin/python`

Example activation:

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
```

Run from:

```bash
cd /project/bi_dsc_community/drc_rag
```

## CLI

Core inputs:
- `--data-dir`: input corpus directory
- `--file-glob`: file glob inside `data-dir`
- `--max-chars`: chunk size
- `--overlap-chars`: chunk overlap
- `--query-count`: number of queries per scenario
- `--top-k`: retrieval depth
- `--semantic-cache-threshold`: cache similarity threshold
- `--dense-weight`: dense retrieval weight
- `--lexical-weight`: lexical retrieval weight

LLM backends:
- `--llm-backend mock`
- `--llm-backend tiny-local`
- `--llm-backend hf`

HF controls:
- `--hf-model`
- `--hf-device`
- `--hf-local-files-only`
- `--hf-max-new-tokens`

Mock controls:
- `--mock-base-latency-ms`
- `--mock-ms-per-token`

Agentic memory ablation:
- `--disable-stm`
- `--disable-ltm`
- `--disable-em`

## Experiments

This directory now contains four main experiment families:
- end-to-end latency benchmark with all scenarios
- memory ablation experiments
- large-count latency-only runs for semantic cache lookup and hybrid retrieval
- conversational retrieval accuracy benchmark

### 1. Baseline End-to-End Run

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --output-dir ./higress_agentic_benchmark/test_outputs_v2
```

Validated output:
- `drc_rag/higress_agentic_benchmark/test_outputs_v2`

### 2. Tiny Local LLM Run

This is the simplest fully local backend with no external model dependency.

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend tiny-local \
  --hf-max-new-tokens 24 \
  --output-dir ./higress_agentic_benchmark/test_outputs_tiny_local
```

Validated output:
- `drc_rag/higress_agentic_benchmark/test_outputs_tiny_local`

### 3. Fast-Memory Agentic Run

This disables `LTM` and `EM` while keeping `STM`.

This removes LTM and EM so memory overhead is reduced and retrieval stays closer to Higress.

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --disable-ltm \
  --disable-em \
  --output-dir ./higress_agentic_benchmark/test_outputs_fast_memory
```

Validated output:
- `drc_rag/higress_agentic_benchmark/test_outputs_fast_memory`

### 4. Memory Ablation Sweep

All-memory baseline:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --output-dir ./higress_agentic_benchmark/ablation_q8_all
```

Disable STM only:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --disable-stm \
  --output-dir ./higress_agentic_benchmark/ablation_no_stm
```

Disable LTM only:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --disable-ltm \
  --output-dir ./higress_agentic_benchmark/ablation_no_ltm
```

Disable EM only:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 8 \
  --llm-backend mock \
  --disable-em \
  --output-dir ./higress_agentic_benchmark/ablation_no_em
```

Validated outputs:
- `drc_rag/higress_agentic_benchmark/ablation_q8_all`
- `drc_rag/higress_agentic_benchmark/ablation_no_stm`
- `drc_rag/higress_agentic_benchmark/ablation_no_ltm`
- `drc_rag/higress_agentic_benchmark/ablation_no_em`

### 5. HF Local Model Backend

Implemented and available through `--llm-backend hf`.

Example:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.run_benchmark \
  --data-dir ./data/wikitext_eval \
  --query-count 4 \
  --llm-backend hf \
  --hf-model gpt2 \
  --hf-local-files-only \
  --hf-device cpu \
  --hf-max-new-tokens 16 \
  --output-dir ./higress_agentic_benchmark/outputs_hf
```

Current note:
- the HF backend is implemented in code
- on this node, runtime validation can fail if user-site `transformers` dependencies are incomplete
- `mock` and `tiny-local` are the validated paths in this repo state

### 6. Large-Count Latency Run: Full Memory

This run isolates only:
- `semantic_cache_lookup`
- `retrieval_hybrid`

It uses a real local corpus and repeats real query patterns so the measured count exceeds `1200` without paying the full end-to-end generation cost.

Output:
- `drc_rag/higress_agentic_benchmark/latency_q1600_repeated64_full_memory`

Command used:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python - <<'PY'
from pathlib import Path
from itertools import cycle, islice
from higress_agentic_benchmark.common import build_corpus, generate_query_cases, summarize_metrics, write_query_metrics_csv, write_summary_csv, write_summary_json, QueryCase
from higress_agentic_benchmark.engines import AgenticRAGEngine, HigressRAGEngine, EngineConfig, build_llm

def expand_cases(base_cases, target_count, prefix):
    expanded = []
    for idx, case in enumerate(islice(cycle(base_cases), target_count)):
        expanded.append(QueryCase(query_id=f"{prefix}_{idx}", query=case.query, allow_cache=case.allow_cache, complex_query=case.complex_query, expected_cache_hit=case.expected_cache_hit, tags=list(case.tags)))
    return expanded

count = 1600
base_chunks = build_corpus('./data/wikitext_eval', max_chars=900, overlap_chars=120, file_glob='*')[:64]
base_query_sets = generate_query_cases(base_chunks, count=min(64, len(base_chunks)))
query_sets = {
    'semantic_cache_lookup': expand_cases(base_query_sets['semantic_cache_lookup'], count, 'semantic'),
    'retrieval_hybrid': expand_cases(base_query_sets['retrieval_hybrid'], count, 'retrieval'),
}
config = EngineConfig(top_k=5, semantic_cache_threshold=0.92, dense_weight=0.65, lexical_weight=0.35, enable_stm=True, enable_ltm=True, enable_em=True)
corpus_texts = [chunk.text for chunk in base_chunks]
higress = HigressRAGEngine(name='HigressRAG', chunks=base_chunks, llm=build_llm('mock', corpus_texts, 'sshleifer/tiny-gpt2', 'cpu', False, 64, 0.0, 0.0), config=config)
agentic = AgenticRAGEngine(chunks=base_chunks, llm=build_llm('mock', corpus_texts, 'sshleifer/tiny-gpt2', 'cpu', False, 64, 0.0, 0.0), config=config)
output_dir = Path('higress_agentic_benchmark/latency_q1600_repeated64_full_memory')
output_dir.mkdir(parents=True, exist_ok=True)
for engine in (higress, agentic):
    for case in query_sets['semantic_cache_lookup']:
        query_embedding = engine.embedder.embed_query(case.query)
        engine.semantic_cache.put(case.query, 'warm answer', query_embedding=query_embedding)
rows = []
for scenario in ('semantic_cache_lookup', 'retrieval_hybrid'):
    for case in query_sets[scenario]:
        rows.append(higress.run_query(scenario, case))
        rows.append(agentic.run_query(scenario, case))
summaries = summarize_metrics(rows)
write_query_metrics_csv(output_dir / 'query_metrics.csv', rows)
write_summary_csv(output_dir / 'summary.csv', summaries)
write_summary_json(output_dir / 'summary.json', rows, summaries)
PY
```

Summary values:
- `AgenticRAG`, `semantic_cache_lookup`
  - `count=1600`
  - `semantic_cache_lookup_ms_avg=0.000755`
  - `total_ms_avg=1.799248`
- `HigressRAG`, `semantic_cache_lookup`
  - `count=1600`
  - `semantic_cache_lookup_ms_avg=0.000740`
  - `total_ms_avg=1.083445`
- `AgenticRAG`, `retrieval_hybrid`
  - `count=1600`
  - `retrieval_ms_avg=9.853310`
  - `memory_load_ms_avg=6.216510`
  - `total_ms_avg=17.791717`
- `HigressRAG`, `retrieval_hybrid`
  - `count=1600`
  - `retrieval_ms_avg=3.142713`
  - `total_ms_avg=5.896100`

### 7. Large-Count Latency Run: No Memory

This run disables:
- `STM`
- `LTM`
- `EM`

Output:
- `drc_rag/higress_agentic_benchmark/latency_q1600_repeated64_no_memory`

Command used:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python - <<'PY'
from pathlib import Path
from itertools import cycle, islice
from higress_agentic_benchmark.common import build_corpus, generate_query_cases, summarize_metrics, write_query_metrics_csv, write_summary_csv, write_summary_json, QueryCase
from higress_agentic_benchmark.engines import AgenticRAGEngine, HigressRAGEngine, EngineConfig, build_llm

def expand_cases(base_cases, target_count, prefix):
    expanded = []
    for idx, case in enumerate(islice(cycle(base_cases), target_count)):
        expanded.append(QueryCase(query_id=f"{prefix}_{idx}", query=case.query, allow_cache=case.allow_cache, complex_query=case.complex_query, expected_cache_hit=case.expected_cache_hit, tags=list(case.tags)))
    return expanded

count = 1600
base_chunks = build_corpus('./data/wikitext_eval', max_chars=900, overlap_chars=120, file_glob='*')[:64]
base_query_sets = generate_query_cases(base_chunks, count=min(64, len(base_chunks)))
query_sets = {
    'semantic_cache_lookup': expand_cases(base_query_sets['semantic_cache_lookup'], count, 'semantic'),
    'retrieval_hybrid': expand_cases(base_query_sets['retrieval_hybrid'], count, 'retrieval'),
}
config = EngineConfig(top_k=5, semantic_cache_threshold=0.92, dense_weight=0.65, lexical_weight=0.35, enable_stm=False, enable_ltm=False, enable_em=False)
corpus_texts = [chunk.text for chunk in base_chunks]
higress = HigressRAGEngine(name='HigressRAG', chunks=base_chunks, llm=build_llm('mock', corpus_texts, 'sshleifer/tiny-gpt2', 'cpu', False, 64, 0.0, 0.0), config=config)
agentic = AgenticRAGEngine(chunks=base_chunks, llm=build_llm('mock', corpus_texts, 'sshleifer/tiny-gpt2', 'cpu', False, 64, 0.0, 0.0), config=config)
output_dir = Path('higress_agentic_benchmark/latency_q1600_repeated64_no_memory')
output_dir.mkdir(parents=True, exist_ok=True)
for engine in (higress, agentic):
    for case in query_sets['semantic_cache_lookup']:
        query_embedding = engine.embedder.embed_query(case.query)
        engine.semantic_cache.put(case.query, 'warm answer', query_embedding=query_embedding)
rows = []
for scenario in ('semantic_cache_lookup', 'retrieval_hybrid'):
    for case in query_sets[scenario]:
        rows.append(higress.run_query(scenario, case))
        rows.append(agentic.run_query(scenario, case))
summaries = summarize_metrics(rows)
write_query_metrics_csv(output_dir / 'query_metrics.csv', rows)
write_summary_csv(output_dir / 'summary.csv', summaries)
write_summary_json(output_dir / 'summary.json', rows, summaries)
PY
```

Summary values:
- `AgenticRAG`, `semantic_cache_lookup`
  - `count=1600`
  - `semantic_cache_lookup_ms_avg=0.000775`
  - `total_ms_avg=1.306833`
- `HigressRAG`, `semantic_cache_lookup`
  - `count=1600`
  - `semantic_cache_lookup_ms_avg=0.000752`
  - `total_ms_avg=1.128114`
- `AgenticRAG`, `retrieval_hybrid`
  - `count=1600`
  - `retrieval_ms_avg=3.803149`
  - `memory_load_ms_avg=0.491176`
  - `memory_store_ms_avg=0.192222`
  - `total_ms_avg=6.120196`
- `HigressRAG`, `retrieval_hybrid`
  - `count=1600`
  - `retrieval_ms_avg=2.447008`
  - `total_ms_avg=4.632868`

Interpretation:
- disabling all memory reduces `AgenticRAG` hybrid retrieval latency substantially
- the remaining gap to `HigressRAG` is smaller, but not zero
- semantic cache lookup remains negligible for both systems

## Outputs

Each run writes:
- `query_metrics.csv`
- `summary.csv`
- `summary.json`

Example validated output directories:
- `drc_rag/higress_agentic_benchmark/test_outputs_v2`
- `drc_rag/higress_agentic_benchmark/test_outputs_fast_memory`
- `drc_rag/higress_agentic_benchmark/test_outputs_tiny_local`
- `drc_rag/higress_agentic_benchmark/ablation_q8_all`
- `drc_rag/higress_agentic_benchmark/ablation_no_stm`
- `drc_rag/higress_agentic_benchmark/ablation_no_ltm`
- `drc_rag/higress_agentic_benchmark/ablation_no_em`
- `drc_rag/higress_agentic_benchmark/latency_q1600_repeated64_full_memory`
- `drc_rag/higress_agentic_benchmark/latency_q1600_repeated64_no_memory`
- `drc_rag/higress_agentic_benchmark/conversational_test_outputs`
- `drc_rag/higress_agentic_benchmark/conversational_test_outputs_n2000_k1600_full`

## Current Findings

From the validated runs:
- `AgenticRAG` is slower mainly because of memory overhead, not hybrid retrieval itself
- `memory_load_ms` and `memory_store_ms` make that overhead explicit
- disabling `LTM` and `EM` is the most useful fast-memory configuration for reducing Agentic overhead
- disabling `STM`, `LTM`, and `EM` reduces `AgenticRAG` hybrid retrieval from `17.7917 ms` total to `6.1202 ms` total in the large-count latency run
- even when raw latency favors `HigressRAG`, `AgenticDRC` strongly outperforms it on conversational retrieval quality

Use `summary.csv` first for comparison, then `query_metrics.csv` for per-query outliers.

## Conversational Retrieval Benchmark

This benchmark is separate from the latency benchmark above.

Purpose:
- test retrieval accuracy on follow-up queries that depend on prior-turn context
- show a case where `AgenticDRC` can outperform `HigressRAG`

Why `AgenticDRC` can win here:
- `HigressRAG` uses only the follow-up query
- `AgenticDRC` uses the previous-turn topic plus the follow-up query
- this acts like memory-based query expansion for conversational retrieval

### Metrics

In `retrieval_summary.csv`:
- `count`
  - number of retrieval test cases evaluated for that engine
- `top1_accuracy`
  - fraction of cases where the correct chunk is ranked first
- `hit_at_k_accuracy`
  - fraction of cases where the correct chunk appears anywhere in the top `k` retrieved results
- `mrr`
  - mean reciprocal rank
  - score per case is `1/rank` for the first correct hit, or `0` if not found
  - this rewards ranking the correct chunk earlier

### CLI

Key flags:
- `--sample-files`
  - if `> 0`, load only the first `N` source files/documents
  - useful for making large runs tractable
- `--progress-every`
  - log progress every `N` engine-query evaluations
- `--num-workers`
  - number of CPU worker processes for retrieval evaluation
  - `0` means auto-detect from `SLURM_CPUS_PER_TASK` or local CPU count
- `--num-cases`
  - number of conversational retrieval cases
- `--top-k`
  - number of retrieved candidates considered for `hit@k` and ranking

### Example Runs

Small validated run:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.benchmark_conversational_retrieval \
  --data-dir ./data/wikitext_eval \
  --num-cases 8 \
  --top-k 5 \
  --output-dir ./higress_agentic_benchmark/conversational_test_outputs
```

Large observable run with progress logging:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m higress_agentic_benchmark.benchmark_conversational_retrieval \
  --data-dir ./data/wikitext2_train \
  --sample-files 2000 \
  --max-chars 4000 \
  --overlap-chars 200 \
  --num-cases 2000 \
  --top-k 1600 \
  --progress-every 100 \
  --num-workers 16 \
  --output-dir ./higress_agentic_benchmark/conversational_test_outputs_n2000_k1600_full
```

Validated large-run result:
- `AgenticDRC`
  - `count=1195`
  - `top1_accuracy=0.9560`
  - `hit_at_k_accuracy=1.0000`
  - `mrr=0.9730`
- `HigressRAG`
  - `count=1195`
  - `top1_accuracy=0.0010`
  - `hit_at_k_accuracy=0.9940`
  - `mrr=0.0060`


### Optimization Notes

The conversational benchmark is optimized for large `num-cases` and `top-k` by:
- avoiding the slower generic `HybridRetriever.search()` path per query
- using a vectorized dense score computation
- applying lexical scores only to sparse matched indices
- using `numpy.argpartition()` instead of sorting the full corpus on every query
- parallelizing retrieval evaluation across multiple CPU worker processes

## Slurm

Slurm runner:
- `drc_rag/higress_agentic_benchmark/run_conversational_retrieval.slurm`

Default submission:

```bash
sbatch /project/bi_dsc_community/drc_rag/higress_agentic_benchmark/run_conversational_retrieval.slurm
```

Large conversational retrieval run:

```bash
sbatch --cpus-per-task=16 --mem=128G \
  --export=ALL,SAMPLE_FILES=2000,NUM_CASES=2000,TOP_K=1600,NUM_WORKERS=16,PROGRESS_EVERY=100 \
  /project/bi_dsc_community/drc_rag/higress_agentic_benchmark/run_conversational_retrieval.slurm
```
