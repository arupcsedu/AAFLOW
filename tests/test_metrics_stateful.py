import csv
import json

from stateful_agentic_algebra.metrics_stateful import METRIC_FIELDS, StatefulMetricsRecorder, aggregate_runs


def test_metrics_recorder_summarizes_stateful_events():
    recorder = StatefulMetricsRecorder(
        run_id="run-test",
        baseline_name="baseline-a",
        workload_name="linear_handoff",
        seed=42,
    )

    recorder.record_event(
        "kv_materialize",
        0.2,
        {
            "kv_bytes": 1024,
            "num_agents": 2,
            "branch_factor": 1,
            "context_tokens": 128,
            "output_tokens": 16,
        },
    )
    recorder.record_event("kv_transfer", 0.05, {"transfer_bytes": 512})
    recorder.record_event("decode", 0.1)
    recorder.record_event("resume", 0.01)
    recorder.record_event("framework_overhead", 0.005)
    recorder.record_event("kv_fork", 0.0)
    recorder.record_event("kv_merge", 0.0)
    recorder.record_event("kv_evict", 0.0)
    recorder.record_event("outputs", 0.0, {"output_texts": ["A", "a", "B"]})
    recorder.mark_kv_reuse()

    summary = recorder.summarize()

    assert set(METRIC_FIELDS).issubset(summary)
    assert summary["prefill_sec"] == 0.2
    assert summary["decode_sec"] == 0.1
    assert summary["transfer_sec"] == 0.05
    assert summary["resume_sec"] == 0.01
    assert summary["omega_sec"] == 0.005
    assert summary["kv_total_bytes"] == 1024
    assert summary["kv_peak_bytes"] == 1024
    assert summary["kv_transferred_bytes"] == 512
    assert summary["transfer_count"] == 1
    assert summary["materialize_count"] == 1
    assert summary["fork_count"] == 1
    assert summary["merge_count"] == 1
    assert summary["evict_count"] == 1
    assert summary["output_agreement_rate"] == 2 / 3
    assert summary["baseline_name"] == "baseline-a"
    assert summary["workload_name"] == "linear_handoff"
    assert summary["run_id"] == "run-test"
    assert summary["seed"] == 42


def test_metrics_json_and_csv_are_written_under_run_dir(tmp_path):
    recorder = StatefulMetricsRecorder(run_id="run-io", run_dir=tmp_path / "runs" / "stateful" / "stamp")
    recorder.record_event("prefill", 0.1, {"context_tokens": 10, "output_tokens": 5})
    recorder.record_event("decode", 0.2)

    json_path = recorder.to_json()
    csv_path = recorder.to_csv()

    assert json_path == tmp_path / "runs" / "stateful" / "stamp" / "metrics.json"
    assert csv_path == tmp_path / "runs" / "stateful" / "stamp" / "metrics.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["prefill_sec"] == 0.1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-io"


def test_aggregate_runs_writes_one_csv_row_per_run(tmp_path):
    first = StatefulMetricsRecorder(run_id="run-a", run_dir=tmp_path / "run-a", workload_name="w")
    first.record_event("prefill", 0.1, {"output_tokens": 10})
    second = StatefulMetricsRecorder(run_id="run-b", run_dir=tmp_path / "run-b", workload_name="w")
    second.record_event("prefill", 0.2, {"output_tokens": 20})
    first.to_json()
    second.to_json()

    aggregate = aggregate_runs(tmp_path, tmp_path / "aggregate.json", tmp_path / "aggregate.csv")

    assert aggregate["count"] == 2
    with (tmp_path / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}


def test_output_agreement_rate_is_null_without_text_outputs():
    recorder = StatefulMetricsRecorder()
    recorder.record_event("decode", 0.1)

    assert recorder.summarize()["output_agreement_rate"] is None
