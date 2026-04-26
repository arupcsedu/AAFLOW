import csv
import subprocess

from stateful_agentic_algebra.plots import PLOT_SPECS, REAL_LLM_PLOT_SPECS, generate_all_plots, generate_real_llm_plots


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
            "baseline_name": "AAFLOW+",
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


def write_real_results(path):
    fieldnames = [
        "run_id",
        "model_id",
        "backend",
        "workload_name",
        "context_tokens",
        "output_tokens",
        "num_agents",
        "branch_factor",
        "num_prompts",
        "available",
        "skipped",
        "ttft_sec",
        "prefill_sec",
        "dense_prefill_sec",
        "stateful_prefill_sec",
        "transfer_sec",
        "t_transfer_sec",
        "t_recompute_sec",
        "bandwidth_name",
        "kv_total_bytes",
        "kv_bytes",
        "throughput_tokens_per_sec",
        "request_throughput_req_per_sec",
        "request_rate",
        "tpot_sec",
        "itl_sec",
        "exact_match",
        "exact_token_match_rate",
    ]
    rows = [
        {
            "run_id": "r1",
            "model_id": "gpt2",
            "backend": "hf",
            "workload_name": "AAFLOW+",
            "context_tokens": 1024,
            "output_tokens": 32,
            "num_agents": 2,
            "branch_factor": 2,
            "num_prompts": 4,
            "available": True,
            "skipped": False,
            "ttft_sec": 0.4,
            "prefill_sec": 0.4,
            "dense_prefill_sec": 1.6,
            "stateful_prefill_sec": 0.4,
            "transfer_sec": 0.05,
            "kv_total_bytes": 1048576,
            "throughput_tokens_per_sec": 100,
        },
        {
            "run_id": "r2",
            "model_id": "gpt2",
            "backend": "hf",
            "workload_name": "dense_prefill",
            "context_tokens": 1024,
            "output_tokens": 32,
            "num_agents": 2,
            "branch_factor": 2,
            "num_prompts": 4,
            "available": True,
            "skipped": False,
            "ttft_sec": 1.6,
            "prefill_sec": 1.6,
            "dense_prefill_sec": 1.6,
            "kv_total_bytes": 1048576,
        },
        {
            "run_id": "r3",
            "model_id": "gpt2",
            "backend": "vllm",
            "workload_name": "vllm_serve",
            "context_tokens": 1024,
            "output_tokens": 32,
            "num_agents": 2,
            "branch_factor": 2,
            "num_prompts": 4,
            "available": True,
            "skipped": False,
            "ttft_sec": 0.25,
            "request_rate": 4,
            "request_throughput_req_per_sec": 3.5,
            "throughput_tokens_per_sec": 112,
            "tpot_sec": 0.01,
            "itl_sec": 0.011,
        },
        {
            "run_id": "r4",
            "model_id": "gpt2",
            "context_tokens": 1024,
            "t_transfer_sec": 0.2,
            "t_recompute_sec": 0.8,
            "bandwidth_name": "100Gbps",
            "kv_bytes": 1048576,
            "available": True,
            "skipped": False,
        },
        {
            "run_id": "r5",
            "model_id": "gpt2",
            "exact_match": True,
            "exact_token_match_rate": 1.0,
            "available": True,
            "skipped": False,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_generate_real_llm_plots_writes_png_and_pdf(tmp_path):
    results = tmp_path / "real_results.csv"
    output_dir = tmp_path / "figures"
    write_real_results(results)

    saved = generate_real_llm_plots(results, output_dir)

    assert len(saved) == len(REAL_LLM_PLOT_SPECS) * 2
    for name, _title in REAL_LLM_PLOT_SPECS:
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            assert path.exists()
            assert path.stat().st_size > 0


def test_real_llm_plots_cli(tmp_path):
    results = tmp_path / "real_results.csv"
    output_dir = tmp_path / "figures"
    write_real_results(results)

    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.plots",
            "--results",
            str(results),
            "--output-dir",
            str(output_dir),
            "--real-llm",
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 16 files" in proc.stdout
    assert (output_dir / "real_ttft_vs_context_by_model.png").exists()
