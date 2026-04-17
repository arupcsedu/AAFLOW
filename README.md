# CyMEM
Agentic DRC transforms fragile, slow AI agents into robust, enterprise-grade systems. By replacing inefficient data pipelines with a high-speed, zero-copy architecture, it unifies data processing and agent orchestration to double end-to-end performance and ensure reliable, scalable execution.

## Benchmark
The main scaling benchmark lives under `drc_rag/benchmark/`.

Key files:
- `drc_rag/benchmark/benchmark_configs_1_to_5.py`
- `drc_rag/benchmark/agentic_scaling_runner.py`
- `drc_rag/benchmark/distributed_agentic_scaling.py`
- `drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch`

### Current Ray Configuration
Current validated Ray setup for strong scaling:
- profile: `strong_ray_only`
- workers: `128`
- nodes: `4`
- cores per node: `40`
- chunks: `10,000,000`
- files: `4096`
- embedder: `local-hash`
- sink: `faiss`

Current supported Ray input formats:
- `raw`: read raw `.txt` files and split on `DELIM`
- `prechunked`: read one chunk per line from cached `.txt` files
- `preembedded`: read cached `.npz` shards with `ids` and `embeddings`

### Best Current Ray Result
Best current semantic Ray result for `strong_ray_only / 128w / 10M` with runtime transform and runtime embedding preserved:
- job: `11368267`
- input format: `raw`
- total: `76.145s`

Stage split:
- `Load = 46.893s`
- `Transform = 0.003s`
- `Embed = 25.909s`
- `Upsert = 3.310s`

Reference output:
- `drc_rag/benchmark/slurm_runs_agentic_scaling/11368267/strong_ray_only_128w/benchmark.out`

Best prechunked Ray result:
- job: `11391379`
- input format: `prechunked`
- total: `75.878s`

Reference output:
- `drc_rag/benchmark/slurm_runs_agentic_scaling/11391379/strong_ray_only_128w/benchmark.out`

Aggressive engineering result, not the headline semantic benchmark:
- job: `11392420`
- input format: `preembedded`
- local corpus staging: enabled
- total: `43.746s`

Stage split:
- `Load = 38.411s`
- `Transform = 0.000s`
- `Embed = 0.000s`
- `Upsert = 4.509s`

Reference output:
- `drc_rag/benchmark/slurm_runs_agentic_scaling/11392420/strong_ray_only_128w/benchmark.out`

### Timing Semantics
Benchmark table columns:
- `Load`
- `Transform`
- `Embed`
- `Upsert`
- `Total`

Important timing rules:
- shared corpus preparation is excluded from benchmark stage timings
- if a run has to build a cached corpus first, prep time is reported separately as:
  - `CorpusPrep(s) [excluded from benchmark stages]: ...`
- local corpus staging is intended to be outside the benchmark timing as well

### Run Commands
Profiles:
- `strong_ray_only`: Ray-only strong-scaling benchmark
- `weak_ray_only`: Ray-only weak-scaling benchmark
- `strong_no_ray`: distributed no-Ray benchmark
- `weak_no_ray`: distributed no-Ray benchmark

Submit the current semantic raw-text Ray path:

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=raw \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

Fresh rerun of the `11368267` semantic configuration:
- job: `11759573`
- config:
  - `PROFILE=strong_ray_only`
  - `PHYSICAL_WORKERS=128`
  - `BASE_NODES=10000000`
  - `BASE_FILES=4096`
  - `CHUNKS_PER_FILE=100000`
  - `RAY_INPUT_FORMAT=raw`

Submit the current semantic prechunked Ray path:

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=prechunked \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

Submit the aggressive engineering path:

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_ray_only,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000,RAY_INPUT_FORMAT=preembedded,LOCAL_CORPUS_STAGE=1,LOCAL_CORPUS_ROOT=/tmp/drc_rag_scaling_corpus_cache \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

Submit the no-Ray strong-scaling path:

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=strong_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES=10000000,BASE_FILES=4096,CHUNKS_PER_FILE=100000 \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

Submit the no-Ray weak-scaling path:

```bash
sbatch --nodes=4 --ntasks-per-node=40 \
  --export=ALL,PROFILE=weak_no_ray,PHYSICAL_WORKERS=128,CORES_PER_NODE=40,BASE_NODES_PER_WORKER=128,BASE_FILES=4096,CHUNKS_PER_FILE=100000 \
  /project/bi_dsc_community/drc_rag/benchmark/slurm_scripts/run_agentic_scaling_strong_weak.sbatch
```

Check a job:

```bash
squeue -j <job_id>
```

### Notes
- use `raw` when transform and embed must remain part of the benchmark semantics
- `prechunked` removes runtime chunk splitting but keeps runtime embedding
- `preembedded` removes both runtime chunking and runtime embedding from the timed Ray path, so it should be treated as an aggressive engineering mode rather than the headline semantic benchmark
- first-time `prechunked` and `preembedded` runs can spend substantial excluded time building the shared cache.
- `local-hash` is a synthetic local embedder used for ingestion/scaling measurement, not a real model-serving benchmark.
