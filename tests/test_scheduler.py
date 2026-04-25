from stateful_agentic_algebra.compiler import StatefulCompiler
from stateful_agentic_algebra.scheduler import CostModel, StateAwareScheduler
from stateful_agentic_algebra.state_objects import KVBlock, KVState


def make_state(nbytes, owner_node="source", token_end=100):
    return KVState(
        state_id="kv-test",
        model_id="model",
        tokenizer_id="tokenizer",
        model_config_hash="hash",
        position_encoding="rope",
        blocks=[
            KVBlock(
                block_id="b0",
                layer_id=0,
                token_start=0,
                token_end=token_end,
                key_shape=(token_end, 8),
                value_shape=(token_end, 8),
                dtype="float16",
                device="cuda:0",
                nbytes=nbytes,
            )
        ],
        owner_node=owner_node,
        owner_device="cuda:0",
    )


def test_transfer_chosen_for_long_context_and_high_bandwidth():
    scheduler = StateAwareScheduler(
        CostModel(
            bandwidth_bytes_per_sec=100_000_000_000,
            network_latency_sec=0.00001,
            prefill_time_per_token_sec=0.001,
            decode_time_per_token_sec=0.0001,
            resume_overhead_sec=0.0001,
            omega_text_sec=0.0001,
            omega_state_sec=0.0001,
            memory_weight=0.0,
        )
    )
    state = make_state(nbytes=10_000_000, owner_node="node-a")

    decision = scheduler.decide(state, token_count=10_000, target_node="node-b")

    assert decision.decision == "transfer"
    assert decision.estimated_memory_bytes == state.total_bytes()
    assert "transfer chosen" in decision.reason


def test_recompute_chosen_for_short_context_and_low_bandwidth():
    scheduler = StateAwareScheduler(
        CostModel(
            bandwidth_bytes_per_sec=1_000_000,
            network_latency_sec=0.5,
            prefill_time_per_token_sec=0.00001,
            decode_time_per_token_sec=0.0001,
            resume_overhead_sec=0.1,
            omega_text_sec=0.0001,
            omega_state_sec=0.1,
            memory_weight=0.0,
        )
    )
    state = make_state(nbytes=100_000_000, owner_node="node-a")

    decision = scheduler.decide(state, token_count=8, target_node="node-b")

    assert decision.decision == "recompute"
    assert decision.estimated_memory_bytes == 0
    assert "recompute chosen" in decision.reason


def test_local_reuse_chosen_when_state_already_on_target_node():
    scheduler = StateAwareScheduler(
        CostModel(
            bandwidth_bytes_per_sec=1,
            network_latency_sec=999.0,
            prefill_time_per_token_sec=999.0,
            decode_time_per_token_sec=0.0,
            resume_overhead_sec=0.01,
            omega_text_sec=999.0,
            omega_state_sec=0.02,
            memory_weight=0.0,
        )
    )
    state = make_state(nbytes=5000, owner_node="node-b")

    decision = scheduler.decide(state, token_count=8, target_node="node-b")

    assert decision.decision == "local_reuse"
    assert decision.estimated_memory_bytes == 5000
    assert "already resides" in decision.reason


def test_estimate_equations_are_exposed():
    scheduler = StateAwareScheduler(
        CostModel(
            bandwidth_bytes_per_sec=100,
            network_latency_sec=2.0,
            prefill_time_per_token_sec=0.5,
            decode_time_per_token_sec=0.25,
            resume_overhead_sec=0.0,
            omega_text_sec=0.0,
            omega_state_sec=0.0,
            memory_weight=0.0,
        )
    )
    state = make_state(nbytes=300, owner_node="a")

    assert scheduler.estimate_prefill(4) == 2.0
    assert scheduler.estimate_transfer(state) == 5.0
    assert scheduler.estimate_decode(8) == 2.0


def test_schedule_graph_returns_order_and_transfer_nodes():
    graph = StatefulCompiler().compile_linear_handoff()
    schedule = StateAwareScheduler().schedule_graph(graph)

    assert schedule["topological_order"][0] == "materialize_0"
    assert schedule["transfer_nodes"] == ["transfer_0"]
    assert schedule["state_edges"]
    assert schedule["data_edges"]

