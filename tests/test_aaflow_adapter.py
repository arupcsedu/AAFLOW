import csv
import json

from stateful_agentic_algebra.aaflow_adapter import (
    export_in_aaflow_style,
    load_existing_metrics,
    run_existing_rag_agent_if_available,
)


def test_load_existing_metrics_from_aaflow_latency_files(tmp_path):
    latencies = {
        "context_build_total": {
            "count": 2,
            "total_sec": 0.5,
            "avg_ms": 250.0,
            "min_ms": 200.0,
            "max_ms": 300.0,
        }
    }
    (tmp_path / "latencies_rank0.json").write_text(json.dumps(latencies), encoding="utf-8")
    with (tmp_path / "throughput_rank0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "count", "avg_latency_s", "throughput_per_s"])
        writer.writeheader()
        writer.writerow({"level": "docs", "count": 4, "avg_latency_s": 0.25, "throughput_per_s": 4.0})

    payload = load_existing_metrics(tmp_path)

    assert payload["available"] is True
    assert payload["normalized"]["context_build_total.total_sec"] == 0.5
    assert payload["normalized"]["throughput.docs.throughput_per_s"] == 4.0


def test_load_existing_metrics_missing_path_is_standalone_safe(tmp_path):
    payload = load_existing_metrics(tmp_path / "missing")

    assert payload["available"] is False
    assert "does not exist" in payload["reason"]


def test_export_in_aaflow_style_writes_summary_and_latency_files(tmp_path):
    rows = [
        {
            "baseline_name": "ours_stateful",
            "workload_name": "tree_of_thought",
            "ttft_sec": 0.1,
            "total_latency_sec": 0.4,
            "prefill_sec": 0.2,
            "decode_sec": 0.1,
            "transfer_sec": 0.05,
            "resume_sec": 0.01,
            "omega_sec": 0.02,
        }
    ]

    paths = export_in_aaflow_style(rows, tmp_path, tag="stateful")

    for path in paths.values():
        assert path
    assert (tmp_path / "stateful_summary.csv").exists()
    assert (tmp_path / "stateful_summary.json").exists()
    assert (tmp_path / "latencies_stateful.csv").exists()
    assert (tmp_path / "latencies_stateful.json").exists()

    latency = json.loads((tmp_path / "latencies_stateful.json").read_text(encoding="utf-8"))
    assert latency["stateful.ttft"]["total_sec"] == 0.1


def test_run_existing_rag_agent_if_available_accepts_prebuilt_agent():
    class LLM:
        def generate(self, prompt, extra_context=None):
            return f"answer: {prompt} {extra_context}"

    class Agent:
        llm = LLM()

        def build_context(self, query):
            return "context", {"query": query}

    result = run_existing_rag_agent_if_available("q", agent=Agent())

    assert result["available"] is True
    assert "answer:" in result["answer"]


def test_run_existing_rag_agent_without_agent_is_standalone_safe():
    result = run_existing_rag_agent_if_available("q")

    assert result["available"] is False
    assert "No pre-built RagAgent" in result["reason"]
