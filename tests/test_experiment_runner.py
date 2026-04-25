import csv
import json
import subprocess


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def run_cli(*args):
    return subprocess.run(
        [PYTHON, "-m", "stateful_agentic_algebra.experiment_runner", *args],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )


def test_list_baselines_includes_ours_and_optional_statuses():
    proc = run_cli("--list-baselines")

    payload = json.loads(proc.stdout)
    names = {item["name"] for item in payload["baselines"]}

    assert {
        "ours_stateful",
        "dense_prefill",
        "aaflow_text",
        "vllm_local_prefix",
        "sglang_prefix",
        "distserve_style",
    }.issubset(names)
    assert all("available" in item for item in payload["baselines"])


def test_single_ours_stateful_run_writes_required_outputs(tmp_path):
    output_dir = tmp_path / "single"

    run_cli(
        "--baseline",
        "ours_stateful",
        "--workload",
        "tree_of_thought",
        "--context-tokens",
        "64",
        "--output-tokens",
        "8",
        "--num-agents",
        "3",
        "--branch-factor",
        "2",
        "--num-requests",
        "2",
        "--output-dir",
        str(output_dir),
    )

    for name in ["results.json", "results.csv", "config.json", "skipped_baselines.json"]:
        assert (output_dir / name).exists()

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["results"]
    skipped = json.loads((output_dir / "skipped_baselines.json").read_text(encoding="utf-8"))["skipped_baselines"]

    assert len(results) == 1
    assert skipped == []
    assert results[0]["baseline_name"] == "ours_stateful"
    assert results[0]["workload_name"] == "tree_of_thought"
    assert results[0]["num_requests"] == 2

    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["baseline_name"] == "ours_stateful"


def test_all_baselines_all_workloads_small_sweep_skips_missing_optional(tmp_path):
    output_dir = tmp_path / "sweep"

    run_cli(
        "--all-baselines",
        "--all-workloads",
        "--context-grid",
        "16",
        "--agent-grid",
        "2",
        "--branch-grid",
        "2",
        "--num-requests",
        "1",
        "--output-dir",
        str(output_dir),
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["results"]
    skipped = json.loads((output_dir / "skipped_baselines.json").read_text(encoding="utf-8"))["skipped_baselines"]

    assert results
    assert {row["workload_name"] for row in results} == {
        "linear_handoff",
        "multi_agent_debate",
        "rag_shared_context",
        "transfer_recompute_crossover",
        "tree_of_thought",
    }
    assert all(row["baseline_name"] != "sglang_prefix" for row in results)
    assert all(item["baseline_name"] == "sglang_prefix" for item in skipped)
