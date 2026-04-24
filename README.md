# Agentic DRC(AAFLOW) : Scalable Patterns for Agentic AI Workflows

Agentic DRC(AAFLOW) is a unified distributed runtime for evaluating agentic retrieval pipelines under strong scaling, weak scaling, framework-level orchestration, and Higress-vs-Agentic comparisons.

This repository is organized around three main questions:

- How fast is the core Agentic DRC(AAFLOW) ingestion path compared with alternative implementations?
- How does a distributed Ray-based pipeline compare with the no-Ray path under the same synthetic workload?
- How does Agentic DRC(AAFLOW) compare with other framework orchestration layers and with Higress-style retrieval pipelines?

## Design And Architecture

Agentic DRC(AAFLOW) is built around a staged ingestion and retrieval model:

1. `Load`
2. `Transform`
3. `Embed`
4. `Upsert`

These stages are benchmarked independently and in aggregate.

### Core Design

- Synthetic corpora are generated deterministically so different backends see the same workload.
- The no-Ray path emphasizes a thin, batched, local execution model.
- The Ray path uses `ray.data` for distributed source processing and a FAISS sink for ingestion benchmarking.
- Framework and Higress benchmarks isolate orchestration effects from model quality.

### Main Subsystems

- `drc_rag/benchmark/`
  - Strong/weak scaling benchmarks for Agentic DRC, Ray-only, and no-Ray paths.
- `drc_rag/framework_rag_pipeline_benchmark/`
  - Framework-level benchmark for `AAFLOW`, `LangChain`, `LangGraph`, `CrewAI`, and `AutoGen`.
- `drc_rag/higress_agentic_benchmark/`
  - HigressRAG vs AgenticRAG comparison benchmark.

### Important Benchmark Semantics

- `raw`
  - Runtime transform and runtime embed are included.
- `prechunked`
  - Runtime chunk splitting is removed, runtime embed is still included.
- `preembedded`
  - Runtime transform and runtime embed are removed from the timed path.
  - This is an engineering throughput mode, not the headline semantic benchmark.

For semantic comparisons, use `raw`.

## Repository Layout

- `drc_rag/benchmark/benchmark_configs_1_to_5.py`
  - Main ingestion benchmark implementation.
- `drc_rag/benchmark/agentic_scaling_runner.py`
  - Scaling config launcher.
- `drc_rag/benchmark/distributed_agentic_scaling.py`
  - Distributed no-Ray aggregation path.
- `drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch`
  - Main Slurm launcher for strong/weak scaling.
- `drc_rag/framework_rag_pipeline_benchmark/run_framework_pipeline.slurm`
  - Framework benchmark Slurm launcher.
- `drc_rag/higress_agentic_benchmark/run_higress_benchmark.slurm`
  - Higress benchmark Slurm launcher.

## Installation

### Recommended Environment

Use the existing benchmark environment:

```bash
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/djy8hg/env/drc_rag_bench_env
cd /project/bi_dsc_community/drc_rag
```

Recommended Python:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python
```

### Optional Package Installation

If you need to rebuild the environment:

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/pip install \
  llama-index chromadb "ray[data]" "dask[distributed]" matplotlib \
  langchain-core langgraph crewai autogen transformers torch faiss-cpu
```

### Basic Verification

```bash
/scratch/djy8hg/env/drc_rag_bench_env/bin/python - <<'PY'
import chromadb, ray, dask
print("environment ok")
PY
```

## Main Experiments

### 0. Datasets and Input
The artifact uses Huggingface wikitext2 datasets and generated workloads. Workloads include small-scale and large-scale chunked corpora.
Use \texttt{download\_hf\_dataset.py} for both train and eval. If datasets is missing, execute pip install datasets.

To generate data/wikitext2_train

```bash
$ cd AAFLOW

$ python download_hf_dataset.py
  --dataset_name wikitext
  --subset_name wikitext-2-raw-v1
  --split train
  --output_dir data/wikitext2_train

```
To generate data/wikitext_eval

```bash

$ cd AAFLOW
$ python download_hf_dataset.py
  --dataset_name wikitext
  --subset_name wikitext-2-raw-v1
  --split test
  --output_dir data/wikitext_eval
```

To generate an optional smaller eval set
```bash

$ cd AAFLOW

$ python download_hf_dataset.py
  --dataset_name wikitext
  --subset_name wikitext-2-raw-v1
  --split test
  --max_samples 1000
  --output_dir data/wikitext_eval
```


### 1. Scaling Benchmark

Directory:

- `drc_rag/benchmark/`

Profiles:

- `strong_ray_only`
- `weak_ray_only`
- `strong_no_ray`
- `weak_no_ray`

Current semantic headline Ray result:

- job: `11368267`
- profile: `strong_ray_only`
- workers: `128`
- input format: `raw`
- total: `76.145s`

Reference:

- `drc_rag/benchmark/slurm_runs_agentic_scaling/11368267/strong_ray_only_128w/benchmark.out`

Current aggressive engineering Ray result:

- job: `11392420`
- profile: `strong_ray_only`
- input format: `preembedded`
- local staging enabled
- total: `43.746s`

Reference:

- `drc_rag/benchmark/slurm_runs_agentic_scaling/11392420/strong_ray_only_128w/benchmark.out`

Current best no-Ray jobs at common worker counts:

- `strong_no_ray_128w`: `11142392`
- `strong_no_ray_256w`: `11101202`
- `strong_no_ray_512w`: `11114125`
- `strong_no_ray_1024w`: `11101380`
- `weak_no_ray_128w`: `11149754`
- `weak_no_ray_256w`: `11115360`
- `weak_no_ray_512w`: `11116446`
- `weak_no_ray_1024w`: `11116450`

### 2. Framework Pipeline Benchmark

Directory:

- `drc_rag/framework_rag_pipeline_benchmark/`

Current validated FAISS overlap run:

- job: `11140904`

Latest confirmation rerun:

- job: `11787868`

Current AAFLOW+ reference run:

- job: `11978368`

Reference:

- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11140904/`
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11787868/`
- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11978368/`

### 3. Higress vs Agentic Benchmark

Directory:

- `drc_rag/higress_agentic_benchmark/`

Current validated distributed run:

- job: `11788136`

AAFLOW+ parity reference run:

- job: `11978502`

Previous validated reference:

- job: `11141981`

Reference:

- `drc_rag/higress_agentic_benchmark/slurm_runs/11788136/`
- `drc_rag/higress_agentic_benchmark/slurm_runs/11978502/`

## How To Run

### Scaling: Semantic Ray Run

This is the semantic headline path. It preserves runtime transform and runtime embed.

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=raw \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Scaling: Ray Prechunked Run

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=prechunked \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Scaling: Aggressive Engineering Ray Run

This is not the semantic headline result.

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=preembedded,LOCAL_CORPUS_STAGE=1,LOCAL_CORPUS_ROOT=/tmp/drc_rag_scaling_corpus_cache \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Scaling: No-Ray Strong Run

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000 \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Scaling: No-Ray Weak Run

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=weak_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES_PER_WORKER=128,BASE_FILES=4096,CHUNKS_PER_FILE=100000 \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

### Framework Benchmark

Validated FAISS overlap configuration:

```bash
cd /project/bi_dsc_community/drc_rag/framework_rag_pipeline_benchmark
sbatch -p parallel --nodes=2 \
  --export=ALL,PHYSICAL_WORKERS=64,CORES_PER_NODE=40,VECTOR_BACKEND=faiss,NODES=200000,FILES=256,GENERATION_SAMPLES=8 \
  run_framework_pipeline.slurm
```

### Higress Benchmark

Validated Higress vs Agentic configuration:

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

## Output Results

### Scaling Benchmark Outputs

Typical output directory:

- `drc_rag/benchmark/slurm_runs_agentic_scaling/<job_id>/`

Key files:

- `benchmark.out`
- `time.txt`
- `allocation.txt`
- `summary.csv`
- `summary.json`

Timing columns:

- `Load`
- `Transform`
- `Embed`
- `Upsert`
- `Total`

Notes:

- corpus preparation is excluded from benchmark stage timings
- local staging is intended to stay outside the timed benchmark path

### Framework Benchmark Outputs

Typical output directory:

- `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/<job_id>/`

Key files:

- `benchmark.out`
- `summary.csv`
- `full_summary.csv`
- `summary.json`
- `plots/`

### Higress Benchmark Outputs

Typical output directory:

- `drc_rag/higress_agentic_benchmark/slurm_runs/<job_id>/`

Key files:

- `benchmark.out`
- `summary.csv`
- `full_summary.csv`
- `summary.json`
- `time.txt`

## Result References

### Scaling

- Semantic Ray headline:
  - `drc_rag/benchmark/slurm_runs_agentic_scaling/11368267/strong_ray_only_128w/benchmark.out`
- Aggressive engineering Ray:
  - `drc_rag/benchmark/slurm_runs_agentic_scaling/11392420/strong_ray_only_128w/benchmark.out`

### Framework

- Validated FAISS overlap:
  - `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11140904/summary.csv`
- Confirmation rerun:
  - `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11787868/summary.csv`
- AAFLOW+ reference:
  - `drc_rag/framework_rag_pipeline_benchmark/slurm_runs/11978368/summary.csv`

### Higress

- Current validated run:
  - `drc_rag/higress_agentic_benchmark/slurm_runs/11788136/summary.csv`
- AAFLOW+ parity reference:
  - `drc_rag/higress_agentic_benchmark/slurm_runs/11978502/summary.csv`
- Previous validated run:
  - `drc_rag/higress_agentic_benchmark/slurm_runs/11141981/summary.csv`

## Operational Notes

- Use `raw` when transform and embed must remain part of the benchmark semantics.
- Use `prechunked` only when runtime chunk splitting can be excluded.
- Do not use `preembedded` as the semantic headline benchmark.
- For Ray experiments, the main remaining bottleneck in semantic runs is `Load`.
- For Higress distributed runs, the validated path uses `LLM_BACKEND=mock`; using `hf` changes startup cost materially.

## Quick Checks

Check a job:

```bash
squeue -j <job_id>
```

Inspect accounting:

```bash
sacct -j <job_id> --format=JobID,JobName,State,Elapsed
```
