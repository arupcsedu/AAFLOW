from stateful_agentic_algebra.operators import (
    EvictionRecord,
    KVEvictOperator,
    KVForkOperator,
    KVMergeOperator,
    KVMaterializeOperator,
    KVTransferOperator,
)
from stateful_agentic_algebra.state_objects import KVStateStatus, StateCompatibilityError
from stateful_agentic_algebra.transport import TransferResult


class FixedTransport:
    name = "fixed"

    def transfer(self, source, target, bytes_moved):
        return TransferResult(
            source=source,
            target=target,
            bytes_moved=bytes_moved,
            transfer_cost_ms=12.5,
            backend=self.name,
            simulated=True,
        )


def materialize_state(**overrides):
    kwargs = {
        "layer_count": 2,
        "head_count": 4,
        "hidden_size": 16,
        "dtype": "float16",
        "device": "cuda:0",
        "owner_node": "node-a",
        "owner_device": "cuda:0",
    }
    kwargs.update(overrides)
    op = KVMaterializeOperator(**kwargs)
    return op.execute(
        token_count=10,
        model_id="model-a",
        tokenizer_id="tok-a",
        model_config_hash="hash-a",
    )


def test_materialize_creates_expected_byte_sizes():
    state = materialize_state()

    # Per layer: key + value, token_count * hidden_size * dtype_bytes * 2.
    assert len(state.blocks) == 2
    assert state.blocks[0].nbytes == 10 * 16 * 2 * 2
    assert state.total_bytes() == 2 * 10 * 16 * 2 * 2
    assert state.blocks[0].key_shape == (4, 10, 4)
    assert state.blocks[0].value_shape == (4, 10, 4)
    assert state.metadata["status"] == KVStateStatus.MATERIALIZED.value


def test_transfer_updates_placement_and_records_latency_and_bytes():
    state = materialize_state()
    transferred = KVTransferOperator(transport=FixedTransport()).execute(
        state,
        source_node="node-a",
        target_node="node-b",
    )

    assert transferred.state_id != state.state_id
    assert transferred.owner_node == "node-b"
    assert transferred.owner_device == state.owner_device
    assert transferred.lineage[-1] == state.state_id
    assert transferred.metadata["source_node"] == "node-a"
    assert transferred.metadata["target_node"] == "node-b"
    assert transferred.metadata["transfer_latency_ms"] == 12.5
    assert transferred.metadata["transfer_bytes"] == state.total_bytes()
    assert transferred.metadata["transport_backend"] == "fixed"


def test_fork_creates_unique_state_ids_and_lineage():
    state = materialize_state()
    branches = KVForkOperator().execute(state, branch_count=3)

    ids = {branch.state_id for branch in branches}
    assert len(branches) == 3
    assert len(ids) == 3
    for idx, branch in enumerate(branches):
        assert branch.state_id != state.state_id
        assert branch.lineage[-1] == state.state_id
        assert branch.blocks == state.blocks
        assert branch.blocks is not state.blocks
        assert branch.metadata["branch_index"] == idx


def test_merge_rejects_incompatible_states():
    first = materialize_state()
    second = materialize_state()
    second.model_id = "other-model"

    try:
        KVMergeOperator().execute([first, second], merge_policy="segment_concat")
    except StateCompatibilityError as exc:
        assert "Incompatible KV states" in str(exc)
    else:
        raise AssertionError("expected StateCompatibilityError")


def test_merge_rejects_non_restricted_policy():
    state = materialize_state()

    try:
        KVMergeOperator().execute([state], merge_policy="unsafe_full_merge")
    except StateCompatibilityError as exc:
        assert "Unsupported restricted merge policy" in str(exc)
    else:
        raise AssertionError("expected StateCompatibilityError")


def test_evict_returns_record():
    state = materialize_state()
    record = KVEvictOperator().execute(state, reason="capacity")

    assert isinstance(record, EvictionRecord)
    assert record.state_id == state.state_id
    assert record.bytes_evicted == state.total_bytes()
    assert record.owner_node == "node-a"
    assert record.owner_device == "cuda:0"
    assert record.reason == "capacity"
    assert state.metadata["status"] == KVStateStatus.EVICTED.value
    assert state.metadata["eviction_record"]["bytes_evicted"] == state.total_bytes()

