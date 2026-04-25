import importlib.util
import json

from stateful_agentic_algebra.state_objects import KVBlock, KVState, KVStateStatus
from stateful_agentic_algebra.transport import (
    LocalFileTransport,
    MockTransport,
    Transport,
    TransportConfig,
    UCXTransport,
)


def make_state():
    return KVState(
        state_id="kv-source",
        model_id="model",
        tokenizer_id="tokenizer",
        model_config_hash="hash",
        position_encoding="rope",
        blocks=[
            KVBlock(
                block_id="b0",
                layer_id=0,
                token_start=0,
                token_end=10,
                key_shape=(10, 8),
                value_shape=(10, 8),
                dtype="float16",
                device="cuda:0",
                nbytes=1024,
                key_ref=object(),
                value_ref={"mock": "buffer"},
            )
        ],
        owner_node="node-a",
        owner_device="cuda:0",
    )


def test_mock_transport_send_receive_updates_placement_and_metrics():
    state = make_state()
    transport = MockTransport(
        TransportConfig(
            bandwidth_gbps=1.0,
            fixed_overhead_ms=1.0,
            sleep=False,
        )
    )

    moved = transport.send_state(state, "node-b")
    received = transport.receive_state(moved.state_id)

    assert moved.owner_node == "node-b"
    assert moved.owner_device == "cuda:0"
    assert moved.lineage[-1] == state.state_id
    assert moved.metadata["status"] == KVStateStatus.TRANSFERRED.value
    assert moved.metadata["source_node"] == "node-a"
    assert moved.metadata["target_node"] == "node-b"
    assert moved.metadata["transfer_bytes"] == state.total_bytes()
    assert moved.metadata["transfer_latency_sec"] == transport.estimate_transfer_time(state)
    assert received is moved
    assert transport.metrics.transfer_bytes == state.total_bytes()
    assert transport.metrics.transfer_count == 1
    assert transport.metrics.transfer_latency_sec > 0


def test_mock_transport_compatibility_transfer_api_records_metrics():
    transport = MockTransport(TransportConfig(bandwidth_gbps=1.0, fixed_overhead_ms=0.0))

    result = transport.transfer("a", "b", 2048)

    assert result.source == "a"
    assert result.target == "b"
    assert result.bytes_moved == 2048
    assert result.transfer_cost_ms > 0
    assert result.backend == "mock"
    assert transport.metrics.transfer_bytes == 2048
    assert transport.metrics.transfer_count == 1


def test_local_file_transport_writes_json_and_mock_buffers(tmp_path):
    state = make_state()
    transport = LocalFileTransport(
        TransportConfig(write_mock_buffers=True, root_dir=str(tmp_path), sleep=False),
    )

    moved = transport.send_state(state, "node-c")
    state_path = tmp_path / moved.state_id / "state.json"
    buffer_path = tmp_path / moved.state_id / "buffers" / "b0.bin"

    assert state_path.exists()
    assert buffer_path.exists()
    assert buffer_path.stat().st_size == 1024

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["owner_node"] == "node-c"
    assert payload["blocks"][0]["key_ref"] is None
    assert payload["blocks"][0]["value_ref"] is None

    received = transport.receive_state(moved.state_id)
    assert received.state_id == moved.state_id
    assert received.owner_node == "node-c"
    assert received.blocks[0].key_ref is None
    assert received.total_bytes() == moved.total_bytes()


def test_transport_auto_is_mock_safe():
    state = make_state()
    transport = Transport(TransportConfig(backend="auto", sleep=False))

    moved = transport.send_state(state, "node-d")

    assert moved.owner_node == "node-d"
    assert transport.metrics.transfer_count == 1


def test_ucx_transport_missing_dependency_raises_clear_runtime_error():
    if importlib.util.find_spec("ucp") is not None:
        return

    try:
        UCXTransport()
    except RuntimeError as exc:
        assert "UCXTransport requires" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when UCX is unavailable")

