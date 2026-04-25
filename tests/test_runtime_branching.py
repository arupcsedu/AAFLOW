from stateful_agentic_algebra import StatefulRuntime
from stateful_agentic_algebra.operators import MergeSummary


def test_runtime_branching_runs_all_branch_nodes():
    runtime = StatefulRuntime()

    result = runtime.run_branching("compare candidate answers", token_count=96, branch_count=3)

    metrics = result["metrics"]
    assert metrics["context_tokens"] == 96
    assert metrics["num_agents"] == 3
    assert metrics["branch_factor"] == 3
    assert metrics["kv_bytes"] > 0
    assert metrics["reuse_ratio"] > 0
    assert metrics["decode_sec"] > 0
    assert len([name for name in result["results"] if name.startswith("generate_branch_")]) == 3
    assert isinstance(result["results"]["merge_branches"], MergeSummary)
    assert "kv_merged" in result["state_bindings"]
    assert "eviction_record" in result["values"]


def test_runtime_branching_preserves_unique_branch_state_ids():
    runtime = StatefulRuntime()

    result = runtime.run_branching("branch state ids", token_count=48, branch_count=2)

    branch_symbols = ["kv_branch_0", "kv_branch_1", "kv_remote_branch_0", "kv_remote_branch_1"]
    concrete_ids = [result["state_bindings"][symbol] for symbol in branch_symbols]

    assert len(set(concrete_ids)) == len(concrete_ids)
    for state_id in concrete_ids:
        assert state_id in runtime.kv_manager.states
