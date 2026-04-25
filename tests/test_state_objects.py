import json

from stateful_agentic_algebra.state_objects import (
    KVBlock,
    KVState,
    StateCompatibilityError,
)


def make_block(block_id="b0", start=0, end=8, nbytes=1024, key_ref=None, value_ref=None):
    return KVBlock(
        block_id=block_id,
        layer_id=1,
        token_start=start,
        token_end=end,
        key_shape=(1, 2, end - start, 16),
        value_shape=(1, 2, end - start, 16),
        dtype="float16",
        device="cuda:0",
        nbytes=nbytes,
        key_ref=key_ref,
        value_ref=value_ref,
    )


def make_state(state_id="s0", **overrides):
    kwargs = {
        "state_id": state_id,
        "model_id": "llama-test",
        "tokenizer_id": "llama-tokenizer",
        "model_config_hash": "abc123",
        "position_encoding": "rope",
        "blocks": [
            make_block("b0", 4, 12, 100),
            make_block("b1", 12, 20, 200),
        ],
        "lineage": ["root"],
        "owner_node": "node-a",
        "owner_device": "cuda:0",
        "metadata": {"scenario": "unit"},
    }
    kwargs.update(overrides)
    return KVState(**kwargs)


def test_kv_block_allows_mock_references_but_json_omits_them():
    key_ref = object()
    value_ref = {"mock": "buffer"}
    block = make_block(key_ref=key_ref, value_ref=value_ref)

    assert block.key_ref is key_ref
    assert block.value_ref == value_ref

    payload = block.to_json_dict()
    assert payload["key_ref"] is None
    assert payload["value_ref"] is None
    assert payload["key_shape"] == [1, 2, 8, 16]

    restored = KVBlock.from_json_dict(payload)
    assert restored.key_shape == (1, 2, 8, 16)
    assert restored.value_shape == (1, 2, 8, 16)
    assert restored.key_ref is None
    assert restored.value_ref is None


def test_kv_state_total_bytes_and_token_span():
    state = make_state()

    assert state.total_bytes() == 300
    assert state.token_span() == (4, 20)


def test_empty_kv_state_token_span_is_zero_width():
    state = make_state(blocks=[])

    assert state.total_bytes() == 0
    assert state.token_span() == (0, 0)


def test_compatibility_checks_required_identity_fields():
    base = make_state()
    compatible = make_state("s1")

    assert base.is_compatible(compatible)

    assert not base.is_compatible(make_state("s2", model_id="other-model"))
    assert not base.is_compatible(make_state("s3", tokenizer_id="other-tokenizer"))
    assert not base.is_compatible(make_state("s4", model_config_hash="other-hash"))
    assert not base.is_compatible(make_state("s5", position_encoding="alibi"))


def test_state_compatibility_error_is_available_for_callers():
    try:
        raise StateCompatibilityError("incompatible")
    except StateCompatibilityError as exc:
        assert str(exc) == "incompatible"
    else:
        raise AssertionError("StateCompatibilityError was not raised")


def test_fork_preserves_metadata_and_blocks_with_new_lineage():
    state = make_state()
    child = state.fork("child-state")

    assert child.state_id == "child-state"
    assert child.model_id == state.model_id
    assert child.tokenizer_id == state.tokenizer_id
    assert child.model_config_hash == state.model_config_hash
    assert child.position_encoding == state.position_encoding
    assert child.blocks == state.blocks
    assert child.blocks is not state.blocks
    assert child.lineage == ["root", "s0"]
    assert child.owner_node == "node-a"
    assert child.owner_device == "cuda:0"
    assert child.metadata == state.metadata
    assert child.metadata is not state.metadata
    assert child.created_at >= state.created_at


def test_kv_state_json_round_trip_is_portable():
    state = make_state()
    payload = state.to_json_dict()

    encoded = json.dumps(payload)
    restored = KVState.from_json_dict(json.loads(encoded))

    assert restored == state
    assert restored.blocks[0].key_shape == (1, 2, 8, 16)
    assert restored.blocks[0].key_ref is None
    assert restored.blocks[0].value_ref is None
