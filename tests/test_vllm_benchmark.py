import json
import subprocess

import pytest

from stateful_agentic_algebra.vllm_benchmark import check_vllm_available, parse_vllm_results


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_parse_vllm_results_extracts_latency_metrics():
    raw = """
============ Serving Benchmark Result ============
Successful requests: 32
Benchmark duration (s): 8.0
Request throughput (req/s): 4.0
Output token throughput (tok/s): 512.0
Mean TTFT (ms): 123.4
Median TTFT (ms): 120.0
Mean TPOT (ms): 8.5
Mean ITL (ms): 8.7
Mean E2EL (ms): 1010.5
"""

    metrics = parse_vllm_results(raw)

    assert metrics["successful_requests"] == 32
    assert metrics["benchmark_duration_sec"] == 8.0
    assert metrics["ttft_sec"] == pytest.approx(0.1234)
    assert metrics["tpot_sec"] == pytest.approx(0.0085)
    assert metrics["itl_sec"] == pytest.approx(0.0087)
    assert metrics["e2el_sec"] == pytest.approx(1.0105)
    assert metrics["throughput_tokens_per_sec"] == 512.0
    assert metrics["request_throughput_req_per_sec"] == 4.0


def test_vllm_benchmark_cli_missing_vllm_skips_cleanly(tmp_path):
    if check_vllm_available():
        return
    output_dir = tmp_path / "vllm"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.vllm_benchmark",
            "--model-id",
            "gpt2",
            "--input-len",
            "16",
            "--output-len",
            "2",
            "--num-prompts",
            "1",
            "--request-rate",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=CWD,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["skipped"] is True
    assert "vLLM is not installed" in metrics["reason"]
    assert (output_dir / "config.json").exists()
    assert (output_dir / "vllm_stdout.log").exists()
    assert (output_dir / "vllm_stderr.log").exists()
    assert (output_dir / "vllm_bench_raw.txt").exists()


def test_vllm_benchmark_cli_require_vllm_returns_nonzero_when_missing(tmp_path):
    if check_vllm_available():
        return
    output_dir = tmp_path / "vllm_required"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.vllm_benchmark",
            "--model-id",
            "gpt2",
            "--input-len",
            "16",
            "--output-len",
            "2",
            "--num-prompts",
            "1",
            "--request-rate",
            "1",
            "--output-dir",
            str(output_dir),
            "--require-vllm",
        ],
        cwd=CWD,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert (output_dir / "metrics.json").exists()
