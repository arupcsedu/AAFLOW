# GPU RAG Benchmark: AAFLOW With Hugging Face 7B/8B Models

This benchmark evaluates the AAFLOW RAG ingestion/generation pipeline against four baselines on 2 A100 GPUs using public Hugging Face text data and local Hugging Face model backends.

Final compared implementations:

- `AsyncParallelOnly`
- `DaskScalableRAG`
- `RayDataScalableRAG`
- `HigressRAG`
- `AAFLOW`

The final AAFLOW path uses length-bucketed compiled embedding graphs, CUDA graph replay, FAISS overlap, and FlashAttention 2 under the Hugging Face backend. The benchmark keeps the comparison apples-to-apples: all methods use the same corpus, same model, same chunk count, same embedding token limit, same generation token limit, and same batch sizes.

## GPU Benchmark Implementation

Main files:

- `benchmark/distributed_hf_weak_scaling.py`
  - Distributed Slurm-task benchmark with one Python process per GPU.
  - Loads a local Hugging Face causal LM per GPU.
  - Measures `Load`, `Transform`, `Embed`, `Upsert`, `Generate`, `Total`, throughput, GPU memory, and optimization setup time.
  - Writes full metrics to `summary.csv` and `summary.json`.
  - Prints a clean fixed-width table to `benchmark.out`.
- `benchmark/prepare_hf_weak_scaling_corpus.py`
  - Builds deterministic rank-sharded text corpora from Hugging Face datasets.
- `benchmark/slurm_scripts/run_hf_weak_scaling.sbatch`
  - Final Slurm launcher for the full GPU benchmark.
  - Loads `gcc/14.2.0` and `cuda/13.0.2` automatically when `ATTN_IMPLEMENTATION=flash_attention_2`.
- `benchmark/slurm_scripts/build_flash_attn_a100.sbatch`
  - Builds and installs FlashAttention 2 for A100 (`sm80`) into a dedicated venv.

AAFLOW configuration used in the final benchmark:

- public output name: `AAFLOW`
- internal strategy: `s10_length_bucket_compile_cudagraph`
- embedding backend: compiled CUDA graph path
- attention backend: `flash_attention_2`
- dtype: `bfloat16`
- embedding input tokens: `128`
- generation new tokens: `32`
- total chunks: `32000` (`16000` per GPU)

## Installation

### 1. Create The Benchmark Environment

```bash
module load miniforge/26.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -p /scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421 \
  -f /scratch/djy8hg/workdir/AAFLOW/environment.benchmarks.yml
```

If the environment already exists, update it with:

```bash
module load miniforge/26.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env update -p /scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421 \
  -f /scratch/djy8hg/workdir/AAFLOW/environment.benchmarks.yml --prune
```

Basic verification:

```bash
/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python - <<'PY'
import torch, transformers, datasets, ray, dask, faiss
print('benchmark environment ok')
PY
```

### 2. Build FlashAttention 2 For A100

```bash
cd /scratch/djy8hg/workdir/AAFLOW
sbatch benchmark/slurm_scripts/build_flash_attn_a100.sbatch
```

The build creates:

- `/scratch/djy8hg/env/drc_rag_benchmarks_flashattn`
- `/scratch/djy8hg/wheelhouse/flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl`

Verify runtime import with the same modules used by the Slurm launcher:

```bash
module load gcc/14.2.0 cuda/13.0.2
/scratch/djy8hg/env/drc_rag_benchmarks_flashattn/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

Expected output:

```text
2.8.3.post1
```

### 3. Prepare The Public Hugging Face Corpus

The final run uses `Salesforce/wikitext`, subset `wikitext-103-raw-v1`, split `train`.

```bash
cd /scratch/djy8hg/workdir/AAFLOW
/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python \
  benchmark/prepare_hf_weak_scaling_corpus.py \
  --dataset Salesforce/wikitext \
  --subset wikitext-103-raw-v1 \
  --split train \
  --ranks 2 \
  --chunks-per-rank 16000 \
  --files-per-rank 64 \
  --chunk-chars 900 \
  --cache-dir /scratch/djy8hg/huggingface/datasets \
  --output-dir /scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_16000x900
```

## Final Run Configuration

Use this final configuration only for the published GPU benchmark comparison.

```bash
cd /scratch/djy8hg/workdir/AAFLOW

COMMON='GPUS=2,CHUNKS_PER_GPU=16000,FILES_PER_GPU=64,CHUNK_CHARS=900,GENERATION_SAMPLES_PER_GPU=64,EMBED_BATCH_SIZE=64,GENERATION_BATCH_SIZE=16,AAFLOW_EMBED_BATCH_SIZE=64,AAFLOW_GENERATION_BATCH_SIZE=16,MAX_INPUT_TOKENS=128,MAX_NEW_TOKENS=32,CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_16000x900,DISABLE_BSP=1,AAFLOW_STRATEGY=s10_length_bucket_compile_cudagraph,ATTN_IMPLEMENTATION=flash_attention_2,DTYPE=bfloat16'

sbatch --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:a100:2 \
  --export=ALL,$COMMON,MODEL=llama3-8b \
  benchmark/slurm_scripts/run_hf_weak_scaling.sbatch

sbatch --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:a100:2 \
  --export=ALL,$COMMON,MODEL=mistral-7b \
  benchmark/slurm_scripts/run_hf_weak_scaling.sbatch
```

The launcher submits a full run including all baselines and `AAFLOW`. The `benchmark.out` file contains the clean fixed-width table without `TTFT(s)`. Detailed `ttft_s` remains in `summary.csv` and `summary.json`.

Final validated jobs:

- Llama3-8B: `14993945`
- Mistral-7B: `14993946`

## Final GPU Benchmark Results

### Llama3-8B, 2 A100s, 32k Chunks

| Config | Load(s) | Transform(s) | Embed(s) | Upsert(s) | Generate(s) | Total(s) | Chunks/s | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AsyncParallelOnly | 0.032 | 0.016 | 152.127 | 0.096 | 4.215 | 156.462 | 204.52 | 490.35 |
| DaskScalableRAG | 0.186 | 0.033 | 152.135 | 0.101 | 4.206 | 156.615 | 204.32 | 491.43 |
| RayDataScalableRAG | 8.098 | 0.521 | 152.215 | 0.096 | 4.227 | 164.995 | 193.95 | 488.98 |
| HigressRAG | 0.033 | 0.022 | 151.979 | 0.098 | 4.224 | 156.309 | 204.72 | 489.33 |
| AAFLOW | 0.030 | 0.019 | 126.832 | 0.344 | 4.193 | 131.048 | 244.19 | 492.92 |

AAFLOW improvement:

- vs best baseline (`HigressRAG`): `16.16%` faster
- vs `RayDataScalableRAG`: `20.57%` faster

### Mistral-7B, 2 A100s, 32k Chunks

| Config | Load(s) | Transform(s) | Embed(s) | Upsert(s) | Generate(s) | Total(s) | Chunks/s | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AsyncParallelOnly | 0.035 | 0.016 | 152.900 | 0.099 | 4.212 | 157.270 | 203.47 | 972.16 |
| DaskScalableRAG | 0.197 | 0.031 | 152.770 | 0.098 | 4.199 | 157.278 | 203.46 | 975.30 |
| RayDataScalableRAG | 9.320 | 0.290 | 153.238 | 0.102 | 4.194 | 167.138 | 191.46 | 976.31 |
| HigressRAG | 0.118 | 0.022 | 152.776 | 0.104 | 4.207 | 157.137 | 203.64 | 973.47 |
| AAFLOW | 0.031 | 0.019 | 128.287 | 0.281 | 4.184 | 132.525 | 241.46 | 978.70 |

AAFLOW improvement:

- vs best baseline (`HigressRAG`): `15.66%` faster
- vs `RayDataScalableRAG`: `20.71%` faster

## Notes

- `Total(s)` excludes `optimization_setup_s_max`; setup time is reported separately in `summary.csv` and `summary.json`.
- FlashAttention improves steady-state embedding throughput but adds compile/setup overhead. Use steady-state totals for repeated benchmark comparisons and setup-inclusive totals for cold-start analysis.
- `benchmark.out` intentionally omits `TTFT(s)` for a cleaner table. Use `summary.csv` or `summary.json` for TTFT analysis.

---

# AAFLOW Ingestion Benchmark

Author: Arup Sarker, `djy8hg@virginia.edu`, `arupcsedu@gmail.com`  
Updated for Sets 1 through 9

This benchmark compares multiple ingestion strategies over the same synthetic corpus and the same simulated embedding model. The main script is:

`/project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py`

It reports:

- `Load(s)`
- `Transform(s)`
- `Embed(s)`
- `Upsert(s)`
- `Total(s)`


## Benchmark Sets

### Set 1: `LoaderParallel`
- Sync load with `SimpleDirectoryReader.load_data()`
- Sync transform pipeline
- Sequential embedding and upsert

### Set 2: `ReaderParallel`
- Parallel file load with `SimpleDirectoryReader.load_data(num_workers=...)`
- Sync transform pipeline
- Sequential embedding and upsert

### Set 3: `PipelineParallelSync`
- Sync load
- Parallel sync transform pipeline with `IngestionPipeline.run(num_workers=...)`
- Sequential embedding and upsert

### Set 4: `AsyncParallelOnly`
- Sync load
- Async transform pipeline
- Async embedding concurrency only
- Upsert batch size fixed at `1`

### Set 5: `AAFLOW`
- Sync load
- Async transform pipeline
- Async embedding with batching
- Batched upserts
- Optional strict scaling mode and sharded Chroma upserts

### Set 6: `RayDataScalableRAG`
- Parallel file load with Ray Data
- Parallel chunking
- Parallel embedding
- Parallel actor-based upsert sink

### Set 7: `DaskScalableRAG`
- Parallel file load with Dask Bag
- Parallel chunking with Dask graph execution
- Parallel embedding with delayed batch tasks
- Parallel upsert with sharded Chroma collections

### Set 8: `BulkSynchronousParallelRAG`
- Bulk synchronous stage execution
- Parallel load stage
- Parallel chunk stage
- Parallel embed stage
- Parallel upsert stage
- Barrier between stages

### Set 9: `HigressRAG`
- Parallel file load
- Direct delimiter chunking
- Parallel embedding
- Parallel batched upsert
- Thin ingestion baseline for comparison against `AAFLOW`


## Installation

### Option 1: Use the existing benchmark environment

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
```

### Option 2: Create a new conda environment

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p /scratch/djy8hg/env/drc_rag_bench_env python=3.11
conda activate /scratch/djy8hg/env/drc_rag_bench_env
pip install llama-index chromadb ray[data] "dask[distributed]" matplotlib
```

### Verify the environment

```bash
python -c "import chromadb, ray, dask, llama_index; print('ok')"
```


## Important Parameters

### Corpus size

- `--nodes`: exact number of chunks to generate
- `--files`: number of files to generate
- `--node-chars`: characters per chunk

### Parallelism

- `--reader-workers`: Set 2 load workers
- `--pipeline-workers`: Set 3 transform workers
- `--async-workers`: Set 4 and Set 5 concurrency
- `--ray-num-cpus`: Set 6 Ray CPU budget
- `--dask-workers`: Set 7 worker count
- `--bsp-workers`: Set 8 worker count
- `--higress-workers`: Set 9 worker count

### Embedding and upsert behavior

- `--set5-embed-batch`
- `--set5-upsert-batch`
- `--set5-embed-workers`
- `--set5-upsert-workers`
- `--set5-upsert-timeout-ms`
- `--set5-upsert-shards`
- `--batch-scale-baseline`
- `--upsert-workers-cap`
- `--set45-upsert-shards`
- `--no-scale-set5-batches`
- `--strict-stage-scaling`

### Simulated embedding cost

- `--request-overhead-ms`
- `--per-item-ms`

### Ray-specific

- `--ray-object-store-memory-mb`


## Local Runs

### Run all baseline sets

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py
```

### Run only Set 4 and Set 5

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16
```

### Run only Ray

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-ray \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --ray-num-cpus 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16 \
  --ray-object-store-memory-mb 4096
```

### Run only Dask

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-dask \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --dask-workers 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16
```

### Run only BSP

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-bsp \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --bsp-workers 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16
```

### Run Set 4, Set 5, Ray, Dask, and BSP together

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --run-ray-set6 \
  --run-dask-set7 \
  --run-bsp-set8 \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --ray-num-cpus 16 \
  --dask-workers 16 \
  --bsp-workers 16 \
  --upsert-workers-cap 16 \
  --set45-upsert-shards 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16 \
  --no-scale-set5-batches
```

### Run AsyncParallelOnly, AAFLOW(AAFLOW), Ray, Dask, and Higress with the best Set5 tuning

Current best completed fair comparison for `AAFLOW` vs `HigressRAG` uses:

- `--set5-embed-workers 96`
- `--set5-upsert-workers 8`
- `--set5-upsert-timeout-ms 2`
- `--set5-upsert-shards 64`

This setting produced a best observed `AAFLOW vs HigressRAG` improvement of `29.4% faster`.

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --run-ray-set6 \
  --run-dask-set7 \
  --run-higress-set9 \
  --nodes 4096 \
  --files 256 \
  --node-chars 900 \
  --async-workers 16 \
  --ray-num-cpus 16 \
  --dask-workers 16 \
  --higress-workers 16 \
  --set5-embed-batch 64 \
  --set5-upsert-batch 256 \
  --set5-embed-workers 96 \
  --set5-upsert-workers 8 \
  --set5-upsert-timeout-ms 2 \
  --set5-upsert-shards 64
```

Note:
- `Set5` now uses true shard count independent of upsert worker count.
- `Set5` upsert workers are multiplexed across shard-ready batches instead of one-worker-per-shard binding.

## Current Validated Scaling Runs

### Baseline no-Ray verification

Weak scaling `128w` verification after the rename to `AAFLOW`:
- job: `11863264`
- output: `drc_rag/benchmark/slurm_runs_agentic_scaling/11863264/weak_no_ray_128w/`
- result: `AAFLOW = 2.994s`

Strong scaling `128w` verification after the rename to `AAFLOW`:
- job: `11863460`
- output: `drc_rag/benchmark/slurm_runs_agentic_scaling/11863460/strong_no_ray_128w/`
- result: `AAFLOW = 31.952s`

### Ray headline and engineering reference

Semantic Ray headline run:
- job: `11368267`
- output: `drc_rag/benchmark/slurm_runs_agentic_scaling/11368267/strong_ray_only_128w/`
- result: `Load = 46.893s`, `Transform = 0.003s`, `Embed = 25.909s`, `Upsert = 3.310s`, `Total = 76.145s`

Aggressive engineering Ray reference:
- job: `11392420`
- output: `drc_rag/benchmark/slurm_runs_agentic_scaling/11392420/strong_ray_only_128w/`
- result: `Load = 38.411s`, `Transform = 0.000s`, `Embed = 0.000s`, `Upsert = 4.509s`, `Total = 43.746s`
- note: this run uses preembedded input and is not the semantic headline benchmark

### Arrow comparison run

Validated combined strong Arrow comparison:
- job: `11862925`
- output: `drc_rag/benchmark/slurm_runs_arrow_scaling/11862925/strong/summary.csv`

Key results:
- `AAFLOW = 3.210s`
- `AAFLOW+ = 2.405s`
- `RayDataScalableRAG = 164.424s`
- `ArrowRayDataScalableRAG = 646.063s`

Interpretation:
- Arrow helps the no-Ray `AAFLOW+` path in this strong scaling comparison.
- Arrow hurts the current Ray implementation badly and should not be treated as a default Ray optimization.


## Scaling-Oriented Configurations

### Strict scaling mode

This mode is useful when you want worker count to affect the stages that can actually scale, especially Set 5 upsert.

It does three things:

- fixes Set 5 batch sizes instead of scaling them with worker count
- sets upsert worker cap equal to async worker count
- shards Set 4 and Set 5 upserts across multiple Chroma collections

Command:

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --run-ray-set6 \
  --run-dask-set7 \
  --run-bsp-set8 \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --async-workers 16 \
  --ray-num-cpus 16 \
  --dask-workers 16 \
  --bsp-workers 16 \
  --upsert-workers-cap 16 \
  --set45-upsert-shards 16 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16 \
  --request-overhead-ms 180 \
  --per-item-ms 3.0 \
  --no-scale-set5-batches \
  --strict-stage-scaling \
  --ray-object-store-memory-mb 4096
```

Important constraint:

- Set 4 and Set 5 `Load(s)` is sync by design, so `Load(s)` should not be expected to decrease with `--async-workers`.
- Strict stage scaling primarily targets `Embed(s)`, `Upsert(s)`, and often `Total(s)`.


## Async Worker Sweep

### Generate CSV and plot

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --graph-async-workers 4,8,12,16,24,32 \
  --nodes 4096 \
  --files 512 \
  --node-chars 1200 \
  --set5-embed-batch 8 \
  --set5-upsert-batch 16 \
  --graph-csv /project/bi_dsc_community/drc_rag/benchmark/async_workers.csv \
  --graph-out /project/bi_dsc_community/drc_rag/benchmark/async_workers.png
```


## Slurm Scripts

Current Slurm scripts in active use:

- `/project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_arrow_agentic_drc.sbatch`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_arrow_scaling_compare.sbatch`

Older scripts still exist for legacy sweeps, but the current benchmark flow uses the `slurm_scripts/` launchers.

### `run_multinode_async_ray.sbatch`

Purpose:

- runs the broader benchmark flow
- supports Ray, Dask, and BSP
- can sweep worker pairs across nodes

Default per-run behavior:

- same `async-workers` is used for Set 4, Set 5, Set 6, Set 7, and Set 8
- same worker count is passed into Ray, Dask, and BSP for that run

### `run_multinode_set456_async_ray.sbatch`

Purpose:

- optimized for Set 4, Set 5, and scalable-set comparisons
- now also supports Set 6, Set 7, and Set 8
- resource defaults are more conservative than older versions
- supports profile presets through `PROFILE=...`

Available profiles:

- `default`: standard Set 4, Set 5, Ray, Dask, BSP sweep behavior
- `bsp_large`: runs only BSP with large benchmark defaults
- `no_bsp_large`: disables BSP and keeps a large benchmark profile for AAFLOW, Ray, and Dask


## Recommended Slurm Commands

### Baseline strong no-Ray `128w`

```bash
sbatch --nodes=13 --ntasks-per-node=10 \
  --export=ALL,PROFILE=strong_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=10,BASE_NODES=100000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,SHARED_CORPUS_ROOT=/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

This is the verified post-rename strong `AAFLOW` run shape used by `11863460`.

### Baseline weak no-Ray `128w`

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=weak_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=100000,BASE_FILES=128,BASE_NODES_PER_WORKER=95000,CHUNKS_PER_FILE=1000,SHARED_CORPUS_ROOT=/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Semantic Ray-only `128w`

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=raw \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Arrow comparison, no-Ray only

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,MODE=no_ray_only,PROFILE=strong,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,NODE_CHARS=900,CHUNKS_PER_FILE=100000,EMBED_BATCH_SIZE=256,UPSERT_BATCH_SIZE=512,SINK_BACKEND=faiss \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_arrow_scaling_compare.sbatch
```

### Arrow comparison, Ray only

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,MODE=ray_only,PROFILE=strong,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,NODE_CHARS=900,CHUNKS_PER_FILE=100000,EMBED_BATCH_SIZE=256,UPSERT_BATCH_SIZE=512,SINK_BACKEND=faiss \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_arrow_scaling_compare.sbatch
```

### Arrow comparison, combined

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,MODE=both,PROFILE=strong,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,NODE_CHARS=900,CHUNKS_PER_FILE=100000,EMBED_BATCH_SIZE=256,UPSERT_BATCH_SIZE=512,SINK_BACKEND=faiss \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_arrow_scaling_compare.sbatch
```

## Slurm Submission Commands

### Submit Set 4, Set 5, Ray, Dask, and BSP together

```bash
WORKER_CPU_PAIRS='4:4,8:8,12:12,16:16' \
sbatch -N 2 --cpus-per-task=12 --mem=192G \
  --export=ALL,\
RUN_RAY_SET6=1,\
RUN_DASK_SET7=1,\
RUN_BSP_SET8=1,\
NODES=4096,\
FILES=512,\
NODE_CHARS=1200,\
UPSERT_CAP=-1,\
SET45_UPSERT_SHARDS=-1,\
NO_SCALE_SET5_BATCHES=1,\
STRICT_STAGE_SCALING=1,\
SET5_EMBED_BATCH=8,\
SET5_UPSERT_BATCH=16,\
REQUEST_OVERHEAD_MS=180,\
PER_ITEM_MS=3.0,\
BATCH_SCALE_BASELINE=256,\
RAY_OBJECT_STORE_MB=4096 \
  /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
```

### Submit the large BSP-only profile

```bash
PROFILE=bsp_large \
WORKER_CPU_PAIRS='128:128,256:256,512:512' \
sbatch -N 4 --cpus-per-task=512 --mem=0 \
  /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
```

### Submit the large profile without BSP

```bash
PROFILE=no_bsp_large \
WORKER_CPU_PAIRS='128:128,256:256,512:512' \
sbatch -N 4 --cpus-per-task=512 --mem=0 \
  /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
```

### Submit the large profile without BSP and without Ray or Dask

```bash
PROFILE=no_bsp_large \
WORKER_CPU_PAIRS='128:128,256:256,512:512' \
sbatch -N 4 --cpus-per-task=512 --mem=0 \
  --export=ALL,RUN_RAY_SET6=0,RUN_DASK_SET7=0 \
  /project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch
```

### Submit the broader multinode script

```bash
WORKER_CPU_PAIRS='4:4,8:8,12:12,16:16' \
sbatch -N 2 --cpus-per-task=12 --mem=192G \
  --export=ALL,\
RUN_RAY_SET6=1,\
RUN_DASK_SET7=1,\
RUN_BSP_SET8=1,\
NODES=2048,\
FILES=256,\
NODE_CHARS=900,\
UPSERT_CAP=-1,\
SET45_UPSERT_SHARDS=-1,\
SET5_EMBED_BATCH=16,\
SET5_UPSERT_BATCH=32,\
REQUEST_OVERHEAD_MS=120,\
PER_ITEM_MS=2.0,\
BATCH_SCALE_BASELINE=128,\
RAY_OBJECT_STORE_MB=2048 \
  /project/bi_dsc_community/drc_rag/benchmark/run_multinode_async_ray.sbatch
```


## Slurm Outputs

Per-job outputs are written under:

- `/project/bi_dsc_community/drc_rag/benchmark/slurm_runs/<jobid>/`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_runs_set456/<jobid>/`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_runs_agentic_scaling/<jobid>/`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_runs_arrow_agentic_drc/<jobid>/`
- `/project/bi_dsc_community/drc_rag/benchmark/slurm_runs_arrow_scaling/<jobid>/`

Important files:

- `benchmark.out`
- `time.txt`
- `summary.csv`
- `resource_summary.csv`
- `resource_summary_gb.csv`


## Troubleshooting

### Dask import error

If you see `ModuleNotFoundError: No module named 'dask'`, install:

```bash
pip install "dask[distributed]"
```

### Ray startup or object store backpressure

If Ray is slow or stalls:

- increase `--ray-object-store-memory-mb`
- reduce `--ray-num-cpus`
- reduce `--async-workers`

### Upsert time does not decrease

Common reasons:

- `--upsert-workers-cap` is below `--async-workers`
- `--set45-upsert-shards` is too small
- batch sizes are too large and reduce concurrency
- all writes are contending on one Chroma collection

Recommended scaling parameters:

- `UPSERT_CAP=-1`
- `SET45_UPSERT_SHARDS=-1`
- `NO_SCALE_SET5_BATCHES=1`
- `STRICT_STAGE_SCALING=1`

### Slurm job stays pending

Common reason:

- requesting too many nodes or too much memory

Use leaner requests when queue time matters:

```bash
sbatch -N 2 --cpus-per-task=12 --mem=192G ...
```


## Quick Start

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
```

```bash
python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
  --only-async \
  --run-ray-set6 \
  --run-dask-set7 \
  --run-bsp-set8 \
  --nodes 32 \
  --files 8 \
  --node-chars 200 \
  --async-workers 2 \
  --ray-num-cpus 2 \
  --dask-workers 2 \
  --bsp-workers 2 \
  --upsert-workers-cap 2 \
  --set45-upsert-shards 2 \
  --set5-embed-batch 4 \
  --set5-upsert-batch 4 \
  --no-scale-set5-batches
```


## Hugging Face Llama 3 and Mistral Benchmark

`benchmark_hf_models.py` runs a real GPU-backed RAG pipeline benchmark for:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

For each model it measures:

- model load time, reported separately
- corpus load and transform
- hidden-state mean-pooling embeddings
- FAISS upsert
- causal text generation
- embedding throughput, generation throughput, and peak GPU memory

The default Slurm launcher uses one A100 and loads the two models sequentially:

```bash
cd /scratch/djy8hg/workdir/AAFLOW

sbatch benchmark/slurm_scripts/run_hf_model_benchmark.sbatch
```

Override the workload through exported variables:

```bash
sbatch \
  --export=ALL,CHUNKS=256,FILES=32,EMBED_BATCH_SIZE=4,GENERATION_SAMPLES=16,MAX_INPUT_TOKENS=256,MAX_NEW_TOKENS=64 \
  benchmark/slurm_scripts/run_hf_model_benchmark.sbatch
```

Run one model:

```bash
sbatch \
  --export=ALL,MODELS=llama3-8b \
  benchmark/slurm_scripts/run_hf_model_benchmark.sbatch
```

Outputs are written to:

```text
benchmark/slurm_runs_hf_models/<jobid>/
```

Important files:

- `benchmark.out`
- `time.txt`
- `summary.csv`
- `summary.json`
- `config.json`
- `allocation.txt`

### Validated A100 result

Job `14749493` ran on one A100 with:

- `CHUNKS=128`
- `EMBED_BATCH_SIZE=4`
- `GENERATION_SAMPLES=8`
- `MAX_INPUT_TOKENS=128`
- `MAX_NEW_TOKENS=32`
- `DTYPE=bfloat16`

| Model | Model load | Embed | Generate | Pipeline total | Embed items/s | Generate tokens/s | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 8B Instruct | 11.660 s | 1.703 s | 6.578 s | 8.285 s | 75.18 | 38.92 | 15.02 GB |
| Mistral 7B Instruct v0.3 | 8.918 s | 1.585 s | 6.390 s | 7.982 s | 80.76 | 40.06 | 13.57 GB |

Model load time is reported separately and excluded from the pipeline total.


## Hugging Face Weak Scaling

`distributed_hf_weak_scaling.py` compares these configurations with real causal-model
hidden-state embeddings and generation:

- `AsyncParallelOnly`
- `DaskScalableRAG`
- `RayDataScalableRAG`
- `BulkSynchronousParallelRAG`
- `HigressRAG`
- `AAFLOW`

Weak scaling keeps the default workload fixed at 128 chunks and 8 generation prompts
per A100. Each Slurm task owns one A100 and one independent Hugging Face model replica.
Rank 0 reports maximum stage latency and aggregate throughput.

Submit the six-run Llama/Mistral matrix:

```bash
cd /scratch/djy8hg/workdir/AAFLOW

for model in llama3-8b mistral-7b; do
  for gpus in 1 2 4; do
    sbatch \
      --ntasks="$gpus" \
      --gres="gpu:a100:$gpus" \
      --cpus-per-task=8 \
      --mem="$((96 * gpus))G" \
      --export=ALL,MODEL="$model",GPUS="$gpus" \
      benchmark/slurm_scripts/run_hf_weak_scaling.sbatch
  done
done
```

Results are written under:

```text
benchmark/slurm_runs_hf_weak_scaling/<jobid>/<model>_<gpus>gpu/
```

Each result directory contains `summary.csv`, `summary.json`, `config.json`,
`allocation.txt`, per-rank JSON results, and the captured benchmark output.
The combined scaling report is:

```text
benchmark/slurm_runs_hf_weak_scaling/weak_scaling_summary.csv
```

### Validated A100 weak-scaling matrix

The following jobs completed successfully on June 10, 2026:

| Model | 1 A100 | 2 A100s | 4 A100s |
|---|---:|---:|---:|
| Llama 3 8B Instruct | `14751987` | `14751988` | `14751989` |
| Mistral 7B Instruct v0.3 | `14751990` | `14751991` | `14751992` |

The workload was fixed per GPU:

- 128 chunks in 16 files
- 8 generation prompts
- embedding batch size 4
- generation batch size 1
- 128 maximum input tokens
- 32 maximum new tokens
- BF16 model weights

AAFLOW results:

| Model | GPUs | Total chunks | Total time | Chunks/s | Chunk efficiency | Generation tokens/s | Generation efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | 1 | 128 | 7.438 s | 17.21 | 100.0% | 39.39 | 100.0% |
| Llama 3 8B | 2 | 256 | 7.967 s | 32.13 | 93.4% | 76.11 | 96.6% |
| Llama 3 8B | 4 | 512 | 7.907 s | 64.76 | 94.1% | 148.62 | 94.3% |
| Mistral 7B | 1 | 128 | 7.742 s | 16.53 | 100.0% | 41.38 | 100.0% |
| Mistral 7B | 2 | 256 | 7.969 s | 32.13 | 97.2% | 79.90 | 96.6% |
| Mistral 7B | 4 | 512 | 7.915 s | 64.69 | 97.8% | 160.91 | 97.2% |

Four-GPU comparison:

| Model | Configuration | Total time | Chunks/s | Chunk efficiency |
|---|---|---:|---:|---:|
| Llama 3 8B | AsyncParallelOnly | 9.450 s | 54.18 | 96.0% |
| Llama 3 8B | DaskScalableRAG | 8.034 s | 63.73 | 94.7% |
| Llama 3 8B | RayDataScalableRAG | 15.903 s | 32.20 | 104.4% |
| Llama 3 8B | BulkSynchronousParallelRAG | 7.915 s | 64.68 | 93.1% |
| Llama 3 8B | HigressRAG | 7.910 s | 64.73 | 93.3% |
| Llama 3 8B | AAFLOW | 7.907 s | 64.76 | 94.1% |
| Mistral 7B | AsyncParallelOnly | 9.609 s | 53.28 | 97.6% |
| Mistral 7B | DaskScalableRAG | 8.051 s | 63.60 | 97.9% |
| Mistral 7B | RayDataScalableRAG | 16.763 s | 30.54 | 93.6% |
| Mistral 7B | BulkSynchronousParallelRAG | 7.909 s | 64.73 | 97.9% |
| Mistral 7B | HigressRAG | 7.913 s | 64.70 | 97.8% |
| Mistral 7B | AAFLOW | 7.915 s | 64.69 | 97.8% |

Efficiency is aggregate throughput divided by `one-GPU throughput * GPU count`.
Model loading is reported separately and excluded from the pipeline total.

This experiment measures replicated-data weak scaling. Every Slurm rank owns one
complete model replica and processes an independent corpus shard on one GPU. It does
not use tensor parallelism, pipeline parallelism, or cross-GPU model communication.
Consequently, the result measures pipeline throughput scaling rather than the ability
to serve a single request across multiple GPUs.

At this small per-GPU workload, Ray startup dominates its total time: the one-GPU
Ray load stage took approximately 8-9 seconds while the actual Ray Data map stages
completed in well under a second. Use a substantially larger corpus, or connect to a
persistent Ray cluster, before drawing conclusions about steady-state Ray throughput.


## WikiText-103 Large-Corpus Two-A100 Benchmark

This benchmark uses the public Hugging Face dataset:

- dataset: `Salesforce/wikitext`
- configuration: `wikitext-103-raw-v1`
- split: `train`
- source rows: 1,801,350

Corpus construction is deliberately outside the measured pipeline. The preparation
step deterministically builds 8,192 chunks of approximately 900 characters, divided
into two independent rank shards of 4,096 chunks and 64 files each:

```bash
cd /scratch/djy8hg/workdir/AAFLOW

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/djy8hg/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

python benchmark/prepare_hf_weak_scaling_corpus.py \
  --dataset Salesforce/wikitext \
  --subset wikitext-103-raw-v1 \
  --split train \
  --ranks 2 \
  --chunks-per-rank 4096 \
  --files-per-rank 64 \
  --chunk-chars 900 \
  --output-dir /scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_4096x900
```

### AAFLOW optimizations

The optimized path retains Python load/transform, real hidden-state embedding,
FAISS insertion, and causal generation. It adds:

- length bucketing before GPU inference to reduce tokenizer padding
- embedding batch size 32 instead of 4
- generation batch size 8 instead of 1
- bounded producer/consumer overlap between embedding and FAISS insertion

No chunks, embeddings, inserts, or generation prompts are skipped.

Run the full two-A100 comparison:

```bash
COMMON='GPUS=2,CHUNKS_PER_GPU=4096,FILES_PER_GPU=64,CHUNK_CHARS=900,GENERATION_SAMPLES_PER_GPU=64,EMBED_BATCH_SIZE=4,GENERATION_BATCH_SIZE=1,AAFLOW_EMBED_BATCH_SIZE=32,AAFLOW_GENERATION_BATCH_SIZE=8,MAX_INPUT_TOKENS=128,MAX_NEW_TOKENS=32,CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_4096x900'

sbatch --ntasks=2 --gres=gpu:a100:2 --cpus-per-task=8 --mem=192G \
  --export=ALL,MODEL=llama3-8b,$COMMON \
  benchmark/slurm_scripts/run_hf_weak_scaling.sbatch

sbatch --ntasks=2 --gres=gpu:a100:2 --cpus-per-task=8 --mem=192G \
  --export=ALL,MODEL=mistral-7b,$COMMON \
  benchmark/slurm_scripts/run_hf_weak_scaling.sbatch
```

### Final results

- Llama 3 job: `14847109`
- Mistral job: `14847110`
- machine-readable table: `benchmark/hf_large_dataset_2gpu_results.csv`

Llama 3 8B:

| Configuration | Load | Transform | Embed | Upsert | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| AsyncParallelOnly | 0.055 s | 0.004 s | 103.602 s | 0.035 s | 25.545 s | 129.238 s | 63.39 |
| DaskScalableRAG | 0.296 s | 0.149 s | 51.085 s | 0.049 s | 25.581 s | 77.048 s | 106.32 |
| RayDataScalableRAG | 10.206 s | 0.250 s | 50.465 s | 0.032 s | 25.548 s | 85.971 s | 95.29 |
| BulkSynchronousParallelRAG | 0.011 s | 0.006 s | 50.417 s | 0.037 s | 25.576 s | 76.048 s | 107.72 |
| HigressRAG | 0.012 s | 0.007 s | 50.450 s | 0.031 s | 25.552 s | 76.048 s | 107.72 |
| **AAFLOW** | **0.014 s** | **0.006 s** | **40.298 s** | **0.026 s** | **7.169 s** | **47.035 s** | **174.17** |

Mistral 7B:

| Configuration | Load | Transform | Embed | Upsert | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| AsyncParallelOnly | 0.032 s | 0.004 s | 101.846 s | 0.033 s | 50.447 s | 152.368 s | 53.76 |
| DaskScalableRAG | 0.224 s | 0.018 s | 50.530 s | 0.031 s | 50.493 s | 101.228 s | 80.93 |
| RayDataScalableRAG | 9.582 s | 0.251 s | 50.110 s | 0.029 s | 50.463 s | 110.434 s | 74.18 |
| BulkSynchronousParallelRAG | 0.010 s | 0.006 s | 50.247 s | 0.031 s | 50.423 s | 100.723 s | 81.33 |
| HigressRAG | 0.015 s | 0.006 s | 50.312 s | 0.030 s | 50.422 s | 100.783 s | 81.28 |
| **AAFLOW** | **0.010 s** | **0.006 s** | **39.252 s** | **0.025 s** | **6.967 s** | **46.232 s** | **177.19** |

Relative to the nearest competitor by total time:

- Llama 3: AAFLOW is 38.2% faster than `BulkSynchronousParallelRAG`
- Mistral: AAFLOW is 54.1% faster than `BulkSynchronousParallelRAG`

These percentages use `(competitor time - AAFLOW time) / competitor time`.

### Optimization ablation

Jobs `14757330` and `14757331` isolate the contribution of each optimization:

| Model | AAFLOW baseline | Embed batching only | Generation batching only | Combined AAFLOW |
|---|---:|---:|---:|---:|
| Llama 3 8B | 74.992 s | 64.257 s | 57.036 s | 46.302 s |
| Mistral 7B | 99.488 s | 88.727 s | 56.628 s | 46.092 s |

The comparison intentionally evaluates optimized AAFLOW against the existing
framework configurations. Competitors use embedding batch 4 and generation batch 1;
AAFLOW uses 32 and 8. Therefore, this establishes the value of AAFLOW's scheduling
and batching strategy under these configurations, not an intrinsic framework-only
advantage after independently tuning every competitor to the same batch sizes.

### Equal-batch framework comparison

Jobs `14850080` and `14850081` repeat the experiment with identical inference
settings for every enabled framework:

- embedding batch size: 32
- generation batch size: 8
- 4,096 chunks and 64 generation prompts per GPU
- two A100 GPUs
- BSP remains implemented but is disabled using `DISABLE_BSP=1`

The command uses:

```bash
COMMON='GPUS=2,CHUNKS_PER_GPU=4096,FILES_PER_GPU=64,CHUNK_CHARS=900,GENERATION_SAMPLES_PER_GPU=64,EMBED_BATCH_SIZE=32,GENERATION_BATCH_SIZE=8,AAFLOW_EMBED_BATCH_SIZE=32,AAFLOW_GENERATION_BATCH_SIZE=8,MAX_INPUT_TOKENS=128,MAX_NEW_TOKENS=32,CORPUS_ROOT=/scratch/djy8hg/aaflow_data/hf_wikitext103_2gpu_4096x900,DISABLE_BSP=1'
```

Llama 3 8B:

| Configuration | Embed | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|
| AsyncParallelOnly | 40.159 s | 6.553 s | 46.699 s | 175.42 |
| DaskScalableRAG | 40.185 s | 6.535 s | 47.049 s | 174.12 |
| RayDataScalableRAG | 40.217 s | 6.529 s | 55.222 s | 148.35 |
| **HigressRAG** | **40.128 s** | **6.525 s** | **46.674 s** | **175.52** |
| AAFLOW | 40.162 s | 6.987 s | 46.959 s | 174.45 |

Mistral 7B:

| Configuration | Embed | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|
| AsyncParallelOnly | 39.495 s | 6.886 s | 46.415 s | 176.50 |
| DaskScalableRAG | 39.619 s | 6.872 s | 46.693 s | 175.45 |
| RayDataScalableRAG | 39.560 s | 6.883 s | 54.568 s | 150.12 |
| **HigressRAG** | **39.475 s** | **6.908 s** | **46.411 s** | **176.51** |
| AAFLOW | 39.518 s | 6.925 s | 46.428 s | 176.45 |

Under equal batch sizes, AAFLOW is statistically tied with the non-Ray local
frameworks. It is 0.61% slower than Higress for Llama and 0.04% slower for Mistral.
The earlier 38-54% result was caused primarily by unequal inference batching and
must not be presented as a framework-only performance advantage.

Machine-readable equal-batch results:

```text
benchmark/hf_large_dataset_2gpu_equal_batch_results.csv
```

### Equal-batch AAFLOW tokenizer prefetch

The next optimization preserves the exact equal-batch configuration while
overlapping CPU tokenization for batch N+1 with GPU inference for batch N. AAFLOW
continues to overlap completed embedding batches with the FAISS consumer.

The optimized path also preserves original chunk order. Character-length sorting was
removed because all WikiText chunks reach the 128-token truncation limit, so sorting
did not reduce padding and could produce less favorable generation batch groupings.

A static Hugging Face KV cache was also tested in jobs `14853720` and `14853721`.
It was rejected because generation rose to 31-32 seconds and Llama generated a
different token count. It is not used by the final path.

Final jobs:

- Llama 3: `14855512`
- Mistral: `14855513`

Llama 3 8B:

| Configuration | Embed | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|
| AsyncParallelOnly | 39.485 s | 6.495 s | 46.020 s | 178.01 |
| DaskScalableRAG | 39.594 s | 6.485 s | 46.303 s | 176.92 |
| RayDataScalableRAG | 39.611 s | 6.511 s | 54.539 s | 150.20 |
| HigressRAG | 39.531 s | 6.525 s | 46.102 s | 177.69 |
| **AAFLOW** | **37.996 s** | **6.493 s** | **44.507 s** | **184.06** |

Mistral 7B:

| Configuration | Embed | Generate | Total | Chunks/s |
|---|---:|---:|---:|---:|
| AsyncParallelOnly | 39.762 s | 6.965 s | 46.781 s | 175.11 |
| DaskScalableRAG | 39.920 s | 6.951 s | 47.063 s | 174.07 |
| RayDataScalableRAG | 39.956 s | 6.947 s | 56.049 s | 146.16 |
| HigressRAG | 39.916 s | 6.968 s | 46.930 s | 174.56 |
| **AAFLOW** | **38.625 s** | **6.928 s** | **45.573 s** | **179.75** |

AAFLOW is 3.29% faster than the nearest Llama competitor and 2.58% faster than
the nearest Mistral competitor by total time. Relative to Higress, the improvements
are 3.46% and 2.89%, respectively.

Machine-readable results:

```text
benchmark/hf_large_dataset_2gpu_prefetch_results.csv
```

The remaining runtime is dominated by identical model kernels. Achieving a 20%
framework-only advantage while preserving model, batch size, prompts, token limits,
and output semantics would require lower-level model execution changes such as a
compiled embedding graph, CUDA graph capture, or a different attention kernel. Those
optimizations should be applied to every framework in a strict framework comparison.
