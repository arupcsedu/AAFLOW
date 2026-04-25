import csv
import subprocess

from stateful_agentic_algebra.plots import PLOT_SPECS, generate_all_plots


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def write_results(path):
    fieldnames = [
        "ttft_sec",
        "total_latency_sec",
        "prefill_sec",
        "decode_sec",
        "transfer_sec",
        "resume_sec",
        "omega_sec",
        "throughput_tokens_per_sec",
        "kv_peak_bytes",
        "kv_reuse_ratio",
        "num_agents",
        "branch_factor",
        "context_tokens",
        "baseline_name",
        "workload_name",
        "num_requests",
    ]
    rows = [
        {
            "ttft_sec": 0.2,
            "total_latency_sec": 1.0,
            "prefill_sec": 0.4,
            "decode_sec": 0.2,
            "transfer_sec": 0.05,
            "resume_sec": 0.01,
            "omega_sec": 0.02,
            "throughput_tokens_per_sec": 100,
            "kv_peak_bytes": 1048576,
            "kv_reuse_ratio": 0.5,
            "num_agents": 2,
            "branch_factor": 2,
            "context_tokens": 1024,
            "baseline_name": "ours_stateful",
            "workload_name": "tree_of_thought",
            "num_requests": 1,
        },
        {
            "ttft_sec": 0.6,
            "total_latency_sec": 1.8,
            "prefill_sec": 0.9,
            "decode_sec": 0.2,
            "transfer_sec": 0.0,
            "resume_sec": 0.0,
            "omega_sec": 0.03,
            "throughput_tokens_per_sec": 80,
            "kv_peak_bytes": 4194304,
            "kv_reuse_ratio": 0.0,
            "num_agents": 4,
            "branch_factor": 4,
            "context_tokens": 4096,
            "baseline_name": "dense_prefill",
            "workload_name": "transfer_recompute_crossover",
            "num_requests": 4,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_all_plots_writes_three_formats_per_chart(tmp_path):
    results = tmp_path / "results.csv"
    output_dir = tmp_path / "figures"
    write_results(results)

    saved = generate_all_plots(results, output_dir)

    assert len(saved) == len(PLOT_SPECS) * 3
    for name, _title in PLOT_SPECS:
        for suffix in ("svg", "png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            assert path.exists()
            assert path.stat().st_size > 0


def test_plots_cli(tmp_path):
    results = tmp_path / "results.csv"
    output_dir = tmp_path / "figures"
    write_results(results)

    proc = subprocess.run(
        [PYTHON, "-m", "stateful_agentic_algebra.plots", "--results", str(results), "--output-dir", str(output_dir)],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 21 files" in proc.stdout
    assert (output_dir / "ttft_vs_context_length.png").exists()
