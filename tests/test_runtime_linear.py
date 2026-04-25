import json

from stateful_agentic_algebra import StatefulRuntime
from stateful_agentic_algebra.operators import EvictionRecord
from stateful_agentic_algebra.state_objects import KVStateStatus


def test_runtime_linear_handoff_runs_in_mock_mode():
    runtime = StatefulRuntime()

    result = runtime.run_linear_handoff("explain cache reuse", token_count=128)

    metrics = result["metrics"]
    assert json.loads(result["metrics_json"]) == metrics
    assert metrics["context_tokens"] == 128
    assert metrics["num_agents"] == 1
    assert metrics["branch_factor"] == 1
    assert metrics["kv_bytes"] > 0
    assert metrics["ttft_sec"] >= 0
    assert metrics["total_latency_sec"] >= metrics["ttft_sec"]
    assert metrics["prefill_sec"] > 0
    assert metrics["decode_sec"] > 0
    assert set(result["state_bindings"]) == {"kv_0", "kv_1"}
    assert isinstance(result["results"]["evict_0"], EvictionRecord)
    assert runtime.kv_manager.get(result["state_bindings"]["kv_1"]).metadata["status"] == KVStateStatus.EVICTED.value


def test_runtime_execute_graph_accepts_compiled_linear_graph():
    runtime = StatefulRuntime()
    graph = runtime.compiler.compile_linear_handoff()

    result = runtime.execute_graph(
        graph,
        initial_values={"prompt": "handoff", "token_count": 32},
        context={"context_tokens": 32, "num_agents": 1, "branch_factor": 1},
    )

    assert list(result["results"]) == ["materialize_0", "transfer_0", "generate_0", "evict_0"]
    assert "answer" in result["values"]
    assert result["metrics"]["context_tokens"] == 32
