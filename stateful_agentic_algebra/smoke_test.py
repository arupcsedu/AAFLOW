"""End-to-end smoke test for Stateful Agentic Algebra.

Runs a small mock experiment that includes the dense prefill baseline and the
stateful runtime, writes JSON/CSV outputs, and generates publication-plot
artifacts. It is intentionally CPU/mock-safe and does not require optional
vLLM, SGLang, UCX, NCCL, or CUDA dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

from .experiment_runner import parse_args, run_experiment
from .plots import generate_all_plots


SMOKE_DIR = Path("runs/stateful/smoke")


def main() -> None:
    """Run the smoke workflow and exit by raising on failure."""

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args(
        [
            "--all-baselines",
            "--workload",
            "tree_of_thought",
            "--context-tokens",
            "1024",
            "--output-tokens",
            "32",
            "--num-agents",
            "2",
            "--branch-factor",
            "2",
            "--num-requests",
            "1",
            "--output-dir",
            str(SMOKE_DIR),
        ]
    )
    run_experiment(args)

    results_json = SMOKE_DIR / "results.json"
    results_csv = SMOKE_DIR / "results.csv"
    if not results_json.exists() or not results_csv.exists():
        raise RuntimeError("Smoke test did not create results.json and results.csv")

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    baselines = {row.get("baseline_name") for row in payload.get("results", [])}
    missing = {"dense_prefill", "ours_stateful"} - baselines
    if missing:
        raise RuntimeError(f"Smoke test missing required baseline rows: {sorted(missing)}")

    figure_dir = SMOKE_DIR / "figures"
    saved = generate_all_plots(results_csv, figure_dir)
    if not saved:
        raise RuntimeError("Smoke test did not generate any plot files")
    if not any(path.suffix == ".png" and path.exists() for path in saved):
        raise RuntimeError("Smoke test did not generate a PNG plot")

    print("STATEFUL AAFLOW SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
