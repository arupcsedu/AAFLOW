import csv
import json
import subprocess

from stateful_agentic_algebra.consistency_benchmark import (
    compare_outputs,
    first_divergence_position,
    levenshtein_distance,
    summarize_rows,
)


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_token_comparison_metrics_exact_and_divergent():
    exact = compare_outputs([1, 2, 3], [1, 2, 3], "abc", "abc")
    divergent = compare_outputs([1, 2, 3], [1, 4, 3, 5], "abc", "axc")

    assert exact.exact_match is True
    assert exact.first_divergence_position is None
    assert exact.normalized_edit_distance == 0.0
    assert divergent.exact_match is False
    assert divergent.first_divergence_position == 1
    assert divergent.exact_token_match_rate == 0.5
    assert divergent.normalized_edit_distance == 0.5


def test_edit_distance_helpers():
    assert first_divergence_position([1, 2], [1, 2]) is None
    assert first_divergence_position([1, 2], [1, 2, 3]) == 2
    assert levenshtein_distance([1, 2, 3], [1, 4, 3, 5]) == 2


def test_summary_rows():
    rows = [
        {"available": True, "skipped": False, "exact_match": True, "exact_token_match_rate": 1.0, "normalized_edit_distance": 0.0, "output_agreement_rate": 1.0},
        {"available": False, "skipped": True, "reason": "missing model"},
    ]

    summary = summarize_rows(rows)

    assert summary["num_prompts"] == 2
    assert summary["available_prompts"] == 1
    assert summary["skipped_prompts"] == 1
    assert summary["exact_match_rate"] == 1.0
    assert summary["skip_reasons"] == ["missing model"]


def test_consistency_cli_writes_outputs_when_model_unavailable(tmp_path):
    output_dir = tmp_path / "consistency"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.consistency_benchmark",
            "--model-id",
            "gpt2",
            "--context-tokens",
            "8",
            "--output-tokens",
            "2",
            "--num-prompts",
            "2",
            "--hf-local-files-only",
            "--output-dir",
            str(output_dir),
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["summary"]["num_prompts"] == 2
    assert (output_dir / "consistency.csv").exists()
    assert (output_dir / "consistency_summary.json").exists()
    with (output_dir / "consistency.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
