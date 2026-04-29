# Full Paper Experiment Matrix

This directory contains the generated full-paper matrix:

- 6 experiments
- 2 models: Mistral-7B-Instruct-v0.3 and Meta-Llama-3-8B-Instruct
- 3 backends: HF, vLLM, SGLang
- 36 YAML configs total

Each config is paired with a Slurm wrapper in:

```bash
stateful_agentic_algebra/slurm/paper_experiments/
```

Run one experiment:

```bash
sbatch stateful_agentic_algebra/slurm/paper_experiments/run_exp1_ttft_reduction_mistral_hf.sbatch
```

Run the entire 36-job matrix:

```bash
bash stateful_agentic_algebra/slurm/paper_experiments/submit_all.sh
```

Override partition/reservation without editing files:

```bash
SBATCH_EXTRA_ARGS="--partition=bii-gpu --reservation=bi_fox_dgx" \
  bash stateful_agentic_algebra/slurm/paper_experiments/submit_all.sh
```

Every successful run writes:

- `benchmark.out`
- `results.csv` or experiment-specific CSV
- `summary.out` where applicable
- `figures/*.png`
- `figures/*.pdf`
- `figures/*.svg`

Regenerate configs and wrappers after changing the matrix:

```bash
python stateful_agentic_algebra/scripts/setup_paper_experiments.py
```
