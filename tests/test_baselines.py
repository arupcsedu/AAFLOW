import importlib.util

from stateful_agentic_algebra.baselines import (
    AAFLOWTextBaseline,
    DensePrefillBaseline,
    DistServeStyleBaseline,
    SGLangPrefixBaseline,
    VLLMLocalPrefixBaseline,
    WorkloadConfig,
    get_baselines,
    list_baselines,
)


EXPECTED_SCHEMA = {
    "ttft_sec",
    "total_latency_sec",
    "prefill_sec",
    "decode_sec",
    "transfer_sec",
    "resume_sec",
    "omega_sec",
    "kv_bytes",
    "reuse_ratio",
    "num_agents",
    "branch_factor",
    "context_tokens",
    "context_build_total_sec",
    "llm_generate_sec",
}


def test_all_baselines_are_listed_without_importing_optional_backends():
    records = list_baselines()
    names = {record["name"] for record in records}

    assert {
        "dense_prefill",
        "aaflow_text",
        "vllm_local_prefix",
        "sglang_prefix",
        "distserve_style",
    }.issubset(names)
    assert all("available" in record for record in records)


def test_baseline_metrics_use_identical_schema():
    cfg = WorkloadConfig(context_tokens=64, num_agents=3, branch_factor=3, output_tokens=8)
    baselines = [
        DensePrefillBaseline(),
        AAFLOWTextBaseline(),
        VLLMLocalPrefixBaseline(),
        DistServeStyleBaseline(),
    ]

    results = [baseline.run_workload(cfg) for baseline in baselines]

    assert all(EXPECTED_SCHEMA.issubset(result.metrics) for result in results)
    assert len({frozenset(result.metrics) for result in results}) == 1
    assert results[0].metrics["reuse_ratio"] == 0.0
    assert results[2].metrics["reuse_ratio"] > 0.0


def test_sglang_baseline_runs_with_simulated_fallback_when_unavailable():
    baseline = SGLangPrefixBaseline()
    result = baseline.run_workload({"context_tokens": 16, "num_agents": 2})

    assert result.skipped is False
    assert EXPECTED_SCHEMA.issubset(result.metrics)
    if importlib.util.find_spec("sglang") is None:
        assert result.available is False
        assert result.metadata["mode"] == "mock_sglang_prefix"
    else:
        assert result.available is True
        assert result.metadata["mode"] == "sglang_prefix"


def test_get_baselines_returns_adapter_instances():
    baselines = get_baselines()

    assert [baseline.name for baseline in baselines] == [
        "dense_prefill",
        "aaflow_text",
        "vllm_local_prefix",
        "sglang_prefix",
        "distserve_style",
    ]
