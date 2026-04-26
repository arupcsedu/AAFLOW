import csv
import json
import subprocess


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_multi_llm_runner_dry_run_outputs_matrix_files(tmp_path):
    output_dir = tmp_path / "multi"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.multi_llm_runner",
            "--models",
            "gpt2",
            "--backend",
            "hf,vllm",
            "--context-grid",
            "16,32",
            "--output-grid",
            "4",
            "--agent-grid",
            "2",
            "--branch-grid",
            "2",
            "--num-prompts",
            "2",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["rows"] == 6
    assert (output_dir / "results_raw.jsonl").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "summary_by_model.csv").exists()
    assert (output_dir / "config.json").exists()

    raw_rows = [json.loads(line) for line in (output_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw_rows) == 6
    workloads = {row["workload_name"] for row in raw_rows}
    assert {"ours_stateful", "dense_prefill", "vllm_serve"} <= workloads
    assert all(row["model_id"] == "gpt2" for row in raw_rows)

    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    ours = [row for row in rows if row["workload_name"] == "ours_stateful"][0]
    dense = [row for row in rows if row["workload_name"] == "dense_prefill"][0]
    assert float(ours["kv_reuse_ratio"]) > float(dense["kv_reuse_ratio"])


def test_multi_llm_runner_unknown_backend_is_skipped(tmp_path):
    output_dir = tmp_path / "multi_unknown"
    subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.multi_llm_runner",
            "--models",
            "gpt2",
            "--backend",
            "unknown",
            "--context-grid",
            "16",
            "--output-grid",
            "4",
            "--agent-grid",
            "2",
            "--branch-grid",
            "2",
            "--num-prompts",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    row = json.loads((output_dir / "results_raw.jsonl").read_text(encoding="utf-8").strip())
    assert row["available"] is False
    assert row["skipped"] is True
    assert "unsupported backend" in row["reason"]


def test_multi_llm_runner_accepts_yaml_config(tmp_path):
    config_path = tmp_path / "multi.yaml"
    output_dir = tmp_path / "configured"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  - gpt2",
                "backends:",
                "  - hf",
                "context_grid:",
                "  - 16",
                "output_grid:",
                "  - 4",
                "agent_grid:",
                "  - 2",
                "branch_grid:",
                "  - 2",
                "num_prompts: 1",
                "dry_run: true",
                f"output_dir: {output_dir}",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [PYTHON, "-m", "stateful_agentic_algebra.multi_llm_runner", "--config", str(config_path)],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["rows"] == 2
    assert (output_dir / "results.csv").exists()
