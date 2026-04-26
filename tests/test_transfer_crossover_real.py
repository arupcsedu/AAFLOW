import csv
import json
import subprocess

from stateful_agentic_algebra.transfer_crossover_real import (
    analyze_crossover,
    model_metadata,
    parse_bandwidths,
    parse_latencies,
)


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_model_metadata_estimates_kv_bytes():
    metadata = model_metadata("mistralai/Mistral-7B-Instruct-v0.3")

    assert metadata.hidden_size == 4096
    assert metadata.num_layers == 32
    assert metadata.kv_bytes(1024) > 0


def test_analyze_crossover_metadata_only():
    rows = analyze_crossover(
        model_id="gpt2",
        context_grid=[128, 256],
        output_tokens=8,
        bandwidths=parse_bandwidths("10Gbps,100Gbps"),
        latencies=parse_latencies("10us"),
        metadata_only=True,
    )

    assert len(rows) == 4
    assert all(row["kv_bytes"] > 0 for row in rows)
    assert all("benefit" in row for row in rows)
    assert {row["bandwidth_name"] for row in rows} == {"10Gbps", "100Gbps"}


def test_transfer_crossover_cli_writes_outputs(tmp_path):
    output_dir = tmp_path / "crossover"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.transfer_crossover_real",
            "--model-id",
            "gpt2",
            "--context-grid",
            "128,256",
            "--output-tokens",
            "8",
            "--bandwidths",
            "10Gbps,100Gbps",
            "--latencies",
            "10us",
            "--metadata-only",
            "--output-dir",
            str(output_dir),
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["rows"] == 4
    assert (output_dir / "crossover.csv").exists()
    assert (output_dir / "crossover.json").exists()
    assert (output_dir / "plot_transfer_vs_recompute.png").exists()
    assert (output_dir / "plot_transfer_vs_recompute.pdf").exists()
    with (output_dir / "crossover.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["measurement_source"] == "metadata_estimate"
