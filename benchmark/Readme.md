# LlamaIndex Ingestion Benchmark

Author: Arup Sarker, `djy8hg@virginia.edu`, `arupcsedu@gmail.com`  
Updated for Sets 1 through 8

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

### Set 5: `AgenticDRC`
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

### Embedding and upsert behavior

- `--set5-embed-batch`
- `--set5-upsert-batch`
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

There are two Slurm scripts:

- `/project/bi_dsc_community/drc_rag/benchmark/run_multinode_async_ray.sbatch`
- `/project/bi_dsc_community/drc_rag/benchmark/run_multinode_set456_async_ray.sbatch`

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
- `no_bsp_large`: disables BSP and keeps a large benchmark profile for AgenticDRC, Ray, and Dask


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
