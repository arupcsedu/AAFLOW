# Framework RAG Pipeline Benchmark

This directory contains a standalone RAG pipeline benchmark for comparing RAG ingestion and generation stages across:
- `LangChain`
- `LangGraph`
- `CrewAI`
- `AutoGen`
- `AAFLOW(AAFLOW)`

It measures:
- `load_s`
- `transform_s`
- `generation_s`
- `tokens_per_second`
- `embed_s`
- `upsert_s`
- `total_s`

## Scope

This benchmark is stage-oriented. It does not measure answer quality.
All frameworks run against the same synthetic corpus and the same backend settings so the comparison stays focused on orchestration, batching, overlap, and sink behavior.

Supported benchmark modes:
- `default`
- `fair_parallelism`
- `fair_parallelism_plus_overlap`

Use:
- `fair_parallelism` for equal worker caps and equal stage batch sizing.
- `fair_parallelism_plus_overlap` for overlap-oriented comparisons where `AAFLOW(AAFLOW)` keeps a native overlapping pipeline and non-Agentic frameworks incur modeled dispatch overhead per batch.

Important note:
- the current FAISS overlap benchmark is intentionally designed to expose orchestration and stage-overlap advantages.
- it is not a neutral “identical overhead” benchmark.
- if you use the FAISS overlap profile, document that benchmark semantics explicitly.

## Current Layout

Important files:
- `run_pipeline_benchmark.py`: single-process CLI benchmark
- `distributed_framework_benchmark.py`: distributed Slurm-task benchmark entrypoint
- `runners.py`: framework adapters and benchmark semantics
- `common.py`: corpus generation, vector stores, metric writing
- `plot_results.py`: plot generation
- `run_framework_pipeline.slurm`: distributed Slurm runner

## Backends

Embedding backends:
- `fake`
- `transformers`

Generation backends:
- `fake`
- `transformers`

Vector backends:
- `fake`
- `chroma`
- `faiss`

Notes:
- `faiss` is local per process and works well for thin-ingest and overlap-oriented comparisons.
- distributed `chroma` runs use a separate persistent Chroma path per Slurm rank.
- distributed `faiss` runs use a separate FAISS path per Slurm rank.

## Runtime Modes

Each framework reports a `runtime_mode`:
- `native`: package-backed adapter path was used
- `emulated`: local framework-style adapter path was used
- `distributed`: aggregated result from the distributed Slurm-task benchmark

Current environment in `/scratch/djy8hg/env/drc_rag_bench_env` may not have all framework packages installed. When packages are missing, the benchmark falls back to the emulated path.

## Installation

Recommended Python:
- `/scratch/djy8hg/env/drc_rag_bench_env/bin/python`

Example setup:

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
cd /project/bi_dsc_community/drc_rag
```

Optional packages:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/pip install langchain-core langgraph crewai autogen chromadb transformers torch faiss-cpu
```

## Outputs

Each completed run writes:
- `summary.csv`
- `full_summary.csv`
- `summary.json`

Fields in `summary.csv`:
- `framework`
- `documents_loaded`
- `generated_tokens`
- `load_s`
- `transform_s`
- `tokens_per_second`
- `embed_s`
- `upsert_s`
- `total_s`

`full_summary.csv` contains additional fields such as:
- `runtime_mode`
- `chunks`
- `generated_prompts`
- `generation_s`

## Single-Process CLI

Basic run:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m framework_rag_pipeline_benchmark.run_pipeline_benchmark \
  --benchmark-mode fair_parallelism \
  --vector-backend faiss \
  --faiss-path ./framework_rag_pipeline_benchmark/faiss_store \
  --data-dir ./framework_rag_pipeline_benchmark/synthetic_data \
  --nodes 100000 \
  --files 256 \
  --generation-samples 256 \
  --load-workers 16 \
  --transform-workers 16 \
  --async-workers 16 \
  --physical-workers 16 \
  --embed-workers 16 \
  --upsert-workers 16 \
  --embed-batch-size 64 \
  --upsert-batch-size 256 \
  --output-dir ./framework_rag_pipeline_benchmark/outputs
```

Useful single-process options:
- `--physical-workers N`
- `--framework-filter AAFLOW,AutoGen`
- `--repeat N`
- `--faiss-path PATH`
- `--chroma-path PATH`

`--framework-filter` and `--repeat` are currently supported by `run_pipeline_benchmark.py`.

## Distributed Slurm Benchmark

The Slurm runner launches the distributed benchmark with one Slurm task per physical worker.
Each rank processes its local file shard, writes rank-local results, and rank `0` aggregates the final metrics.

Runner:
- `drc_rag/framework_rag_pipeline_benchmark/run_framework_pipeline.slurm`

The script uses:
- `#SBATCH --ntasks-per-node=40`
- `#SBATCH --cpus-per-task=1`

You must provide enough nodes for:
- `required_nodes = ceil(PHYSICAL_WORKERS / CORES_PER_NODE)`

Example distributed FAISS overlap run:

```bash
cd /project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
sbatch -p parallel --nodes=2 \
  --export=ALL,PHYSICAL_WORKERS=64,CORES_PER_NODE=40,VECTOR_BACKEND=faiss,NODES=200000,FILES=256,GENERATION_SAMPLES=8 \
  run_framework_pipeline.slurm
```

Example distributed Chroma run:

```bash
cd /project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
sbatch -p parallel --nodes=2 \
  --export=ALL,PHYSICAL_WORKERS=64,CORES_PER_NODE=40,VECTOR_BACKEND=chroma,NODES=100000,FILES=256,GENERATION_SAMPLES=256 \
  run_framework_pipeline.slurm
```

## Current Slurm Defaults

General defaults in `run_framework_pipeline.slurm`:
- `NODES=3200`
- `FILES=200`
- `NODE_CHARS=900`
- `GENERATION_OUTPUT_TOKENS=64`
- `LOAD_WORKERS=16`
- `TRANSFORM_WORKERS=16`
- `ASYNC_WORKERS=16`
- `EMBED_BATCH_SIZE=32`
- `REPEAT=1`

Backend-specific defaults:

For `VECTOR_BACKEND=faiss`:
- `GENERATION_SAMPLES=32`
- `BENCHMARK_MODE=fair_parallelism_plus_overlap`
- `UPSERT_BATCH_SIZE=256`
- `AGENTIC_UPSERT_COALESCE_TARGET=256`

For `VECTOR_BACKEND=chroma`:
- `GENERATION_SAMPLES=200`
- `BENCHMARK_MODE=fair_parallelism_plus_overlap`
- `UPSERT_WORKERS=2`
- `UPSERT_BATCH_SIZE=2048`
- `AGENTIC_UPSERT_COALESCE_TARGET=4096`

## Current Semantics

### FAISS overlap profile

The current FAISS overlap benchmark is intentionally designed to make overlap and orchestration differences visible:
- `AAFLOW(AAFLOW)` uses native overlap between generation and streaming `embed -> upsert`
- non-Agentic frameworks execute the same stages with modeled per-batch dispatch overhead in `fair_parallelism_plus_overlap`

This is the profile used when the goal is to demonstrate a large `AAFLOW(AAFLOW)` advantage.

### Chroma profile

The current Chroma profile is storage-bound:
- distributed Chroma uses one persistent path per rank
- Chroma runs are dominated by upsert cost
- the tuning primarily reduces writer contention and increases batch size

## Plotting

Generate plots from a completed run:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python -m framework_rag_pipeline_benchmark.plot_results \
  --summary-csv ./framework_rag_pipeline_benchmark/slurm_runs/<job_id>/summary.csv \
  --output-dir ./framework_rag_pipeline_benchmark/slurm_runs/<job_id>/plots
```

Generated outputs:
- `stage_times.png`
- `throughput.png`

## Validated Runs

### Chroma run

Validated distributed Chroma run:
- `11130710`

Submit command:

```bash
cd /project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
sbatch -p parallel --nodes=2 \
  --export=ALL,PHYSICAL_WORKERS=64,CORES_PER_NODE=40,VECTOR_BACKEND=chroma,NODES=100000,FILES=256,GENERATION_SAMPLES=256 \
  run_framework_pipeline.slurm
```

Output folder:
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11130710/`

Observed summary:
- `AAFLOW(AAFLOW)`: `total_s=3.380`, `upsert_s=1.848`, `embed_s=0.276`
- `LangChain`: `total_s=3.697`, `upsert_s=2.136`, `embed_s=0.320`
- `CrewAI`: `total_s=3.987`, `upsert_s=2.434`, `embed_s=0.316`
- `LangGraph`: `total_s=4.583`, `upsert_s=3.036`, `embed_s=0.322`
- `AutoGen`: `total_s=4.732`, `upsert_s=3.200`, `embed_s=0.307`

### FAISS overlap run

Validated distributed FAISS overlap run:
- `11140904`

Submit command:

```bash
cd /project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
sbatch -p parallel --nodes=2 \
  --export=ALL,PHYSICAL_WORKERS=64,CORES_PER_NODE=40,VECTOR_BACKEND=faiss,NODES=200000,FILES=256,GENERATION_SAMPLES=8 \
  run_framework_pipeline.slurm
```

Output folder:
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11140904/`

Observed summary:
- `AAFLOW`: `total_s=0.875`, `embed_s=0.486`, `upsert_s=0.049`
- `AutoGen`: `total_s=1.614`, `embed_s=1.136`, `upsert_s=0.135`
- `LangGraph`: `total_s=1.614`, `embed_s=1.140`, `upsert_s=0.136`
- `CrewAI`: `total_s=1.625`, `embed_s=1.145`, `upsert_s=0.133`
- `LangChain`: `total_s=1.645`, `embed_s=1.149`, `upsert_s=0.140`

Improvement vs `LangChain`:
- `AAFLOW` is about `46.8%` faster in this overlap-oriented FAISS benchmark.

Latest confirmation rerun of the same configuration:
- `11787868`

Output folder:
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11787868/`

Observed summary:
- `AAFLOW`: `total_s=0.916`, `load_s=0.024`, `transform_s=0.015`, `embed_s=0.492`, `upsert_s=0.072`
- `LangGraph`: `total_s=1.641`, `embed_s=1.149`, `upsert_s=0.137`
- `AutoGen`: `total_s=1.647`, `embed_s=1.154`, `upsert_s=0.157`
- `CrewAI`: `total_s=1.653`, `embed_s=1.161`, `upsert_s=0.141`
- `LangChain`: `total_s=1.661`, `embed_s=1.155`, `upsert_s=0.161`

Arrow boundary rerun of the same configuration:
- `11978368`

Output folder:
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11978368/`

Observed summary:
- `AAFLOW+`: `total_s=0.880`, `load_s=0.011`, `transform_s=0.010`, `embed_s=0.491`, `upsert_s=0.056`
- `AAFLOW`: `total_s=0.893`, `load_s=0.012`, `transform_s=0.010`, `embed_s=0.491`, `upsert_s=0.063`
- `LangGraph`: `total_s=1.613`
- `AutoGen`: `total_s=1.613`
- `CrewAI`: `total_s=1.624`
- `LangChain`: `total_s=1.636`

Interpretation:
- `11787868` confirms that the `11140904` Slurm configuration is still valid.
- `11978368` validates `AAFLOW+`, where Arrow is used only at the embed/upsert boundary.
- `AAFLOW+` is the fastest framework in this profile, about `1.4%` faster than `AAFLOW`.
- `11140904` remains the best historical `AAFLOW` run, while `11978368` is the current `AAFLOW+` reference.

## Notes

- distributed Chroma and FAISS runs isolate each rank’s vector store path; they do not share one persistent store across ranks.
- the distributed benchmark aggregates max stage time across ranks.
- the single-process CLI supports `--framework-filter` and `--repeat`; the distributed Slurm path is intended for full benchmark runs.

## GPU HuggingFace Framework Benchmark

This directory also contains a GPU-based HuggingFace framework benchmark that mirrors the GPU RAG pipeline benchmark in `benchmark/`, but reports framework-level results for:

- `LangChain`
- `LangGraph`
- `CrewAI`
- `AutoGen`
- `AAFLOW`

The GPU benchmark uses real HuggingFace causal language models for both embedding and generation. Embeddings are produced by mean-pooling the model hidden states and normalizing the resulting vectors. FAISS `IndexFlatIP` is used as the local vector sink. The benchmark is designed to compare pipeline orchestration and execution strategy while keeping the model, corpus, generation settings, and baseline stage behavior fixed.

Important files:

- `distributed_hf_framework_benchmark.py`: distributed one-process-per-GPU benchmark entrypoint
- `run_hf_framework_benchmark.sbatch`: Slurm launcher for Llama3-8B and Mistral-7B GPU framework runs
- `slurm_runs_hf_framework/<job_id>/<model>_2gpu/benchmark.out`: tabular benchmark output
- `slurm_runs_hf_framework/<job_id>/<model>_2gpu/summary.csv`: machine-readable summary
- `slurm_runs_hf_framework/<job_id>/<model>_2gpu/summary.json`: JSON summary

### GPU Benchmark Design

Each Slurm rank owns one GPU and one corpus shard. For the final 2-GPU runs, each rank processes `16000` chunks, for `32000` total chunks. The same rank-local corpus is reused by all frameworks in a given run.

Measured stages:

- `Load(s)`: read rank-local text files
- `Transform(s)`: split documents into chunks
- `Embed(s)`: tokenize chunks, run the HF model hidden-state embedding path, mean-pool, normalize, and transfer vectors as needed
- `Upsert(s)`: insert vectors into FAISS
- `Generate(s)`: run batched text generation over sampled prompts
- `Total(s)`: end-to-end stage time excluding model load and AAFLOW compile setup

AAFLOW uses the final selected GPU optimization path:

- fixed semantic workload; no pre-embedded data
- `AAFLOW_EMBED_BATCH_SIZE=128`
- CPU tokenizer prefetch for AAFLOW embedding batches
- compiled AAFLOW embedding graph with default Torch compile options and CUDA graph support
- FlashAttention 2 enabled for model execution
- no TF32 forcing
- no deferred vector transfer in the final setting
- no token-budget batching in the final setting

The compile/setup cost is reported separately as `Setup(s)` and is not included in `Total(s)`. This keeps the measured total focused on steady-state pipeline execution.

### GPU Environment

Validated Python environment with FlashAttention:

```bash
module load gcc/14.2.0 cuda/13.0.2
/scratch/djy8hg/env/drc_rag_benchmarks_flashattn/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

Validated version:

```text
2.8.3.post1
```

Required runtime locations used in the final run:

```bash
PROJECT_ROOT=/scratch/djy8hg/workdir/AAFLOW
PYTHON_BIN=/scratch/djy8hg/env/drc_rag_benchmarks_flashattn/bin/python
HF_HOME=/scratch/djy8hg/huggingface
CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_16000x900
```

The final corpus contains `2` rank shards, `16000` chunks per rank, `64` files per rank, and `900` characters per chunk.

### Final GPU Slurm Command

Run Llama3-8B:

```bash
cd /scratch/djy8hg/workdir/AAFLOW
COMMON="GPUS=2,CHUNKS_PER_GPU=16000,FILES_PER_GPU=64,CHUNK_CHARS=900,GENERATION_SAMPLES_PER_GPU=64,EMBED_BATCH_SIZE=64,GENERATION_BATCH_SIZE=16,AAFLOW_EMBED_BATCH_SIZE=128,AAFLOW_TOKENIZER_PREFETCH=1,AAFLOW_DEFER_VECTOR_TRANSFER=0,AAFLOW_TF32=0,AAFLOW_COMPILE_MODE=default,AAFLOW_TOKEN_BUDGET=0,MAX_INPUT_TOKENS=128,MAX_NEW_TOKENS=32,CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_16000x900,ATTN_IMPLEMENTATION=flash_attention_2,DTYPE=bfloat16"
sbatch --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:a100:2 \
  --export=ALL,$COMMON,MODEL=llama3-8b \
  framework_rag_pipeline_benchmark/run_hf_framework_benchmark.sbatch
```

Run Mistral-7B:

```bash
cd /scratch/djy8hg/workdir/AAFLOW
COMMON="GPUS=2,CHUNKS_PER_GPU=16000,FILES_PER_GPU=64,CHUNK_CHARS=900,GENERATION_SAMPLES_PER_GPU=64,EMBED_BATCH_SIZE=64,GENERATION_BATCH_SIZE=16,AAFLOW_EMBED_BATCH_SIZE=128,AAFLOW_TOKENIZER_PREFETCH=1,AAFLOW_DEFER_VECTOR_TRANSFER=0,AAFLOW_TF32=0,AAFLOW_COMPILE_MODE=default,AAFLOW_TOKEN_BUDGET=0,MAX_INPUT_TOKENS=128,MAX_NEW_TOKENS=32,CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_16000x900,ATTN_IMPLEMENTATION=flash_attention_2,DTYPE=bfloat16"
sbatch --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:a100:2 \
  --export=ALL,$COMMON,MODEL=mistral-7b \
  framework_rag_pipeline_benchmark/run_hf_framework_benchmark.sbatch
```

The launcher requests the dedicated A100 partition and reservation:

```bash
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
```

### Final GPU Results

Final validated runs:

- Llama3-8B: job `15245798`
- Mistral-7B: job `15245799`

Llama3-8B, 2xA100, 32000 chunks:

| Framework | Embed(s) | Upsert(s) | Generate(s) | Total(s) | Chunks/s |
|---|---:|---:|---:|---:|---:|
| LangChain | 152.123 | 0.097 | 4.263 | 156.477 | 204.50 |
| LangGraph | 151.983 | 0.099 | 4.243 | 156.320 | 204.71 |
| CrewAI | 152.203 | 0.099 | 4.244 | 156.546 | 204.41 |
| AutoGen | 152.077 | 0.098 | 4.239 | 156.419 | 204.58 |
| AAFLOW | 124.306 | 0.540 | 4.237 | 128.543 | 248.94 |

AAFLOW is `17.77%` faster than the best Llama3-8B baseline, `LangGraph`.

Mistral-7B, 2xA100, 32000 chunks:

| Framework | Embed(s) | Upsert(s) | Generate(s) | Total(s) | Chunks/s |
|---|---:|---:|---:|---:|---:|
| LangChain | 151.140 | 0.113 | 4.112 | 155.425 | 205.89 |
| LangGraph | 151.037 | 0.098 | 4.103 | 155.282 | 206.08 |
| CrewAI | 151.373 | 0.099 | 4.104 | 155.614 | 205.64 |
| AutoGen | 151.239 | 0.101 | 4.101 | 155.489 | 205.80 |
| AAFLOW | 123.997 | 0.329 | 4.097 | 128.136 | 249.73 |

AAFLOW is `17.48%` faster than the best Mistral-7B baseline, `LangGraph`.

### Optimization Ablations


The dominant gain comes from reducing AAFLOW embedding time while keeping the same semantic workload and generation settings as the other frameworks.
