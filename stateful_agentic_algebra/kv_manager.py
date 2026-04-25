"""KV state manager for Stateful Agentic Algebra.

Paper mapping:
  - KV state object: owns the registry of `KVState` instances.
  - KV materialize / transfer / fork / restricted merge / evict: lifecycle
    methods implement the state transformations used by operators.
  - Metrics: methods update transfer cost, recompute cost, memory, and reuse
    accounting through `StatefulMetricsRecorder`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .metrics_stateful import StatefulMetricsRecorder
from .state_objects import KVBlock, KVState, KVStateStatus, StateCompatibilityError


@dataclass
class KVManagerConfig:
    """Configuration for simulated KV memory accounting."""

    bytes_per_token: int = 2048
    default_owner_node: str = "local"
    default_owner_device: str = "cpu"
    model_id: str = "mock-model"
    tokenizer_id: str = "mock-tokenizer"
    model_config_hash: str = "mock-config"
    position_encoding: str = "absolute"


class KVStateManager:
    """In-memory KV state registry with algebra lifecycle operations."""

    def __init__(
        self,
        config: Optional[KVManagerConfig] = None,
        metrics: Optional[StatefulMetricsRecorder] = None,
    ) -> None:
        self.config = config or KVManagerConfig()
        self.metrics = metrics or StatefulMetricsRecorder()
        self.states: Dict[str, KVState] = {}

    def get(self, state_id: str) -> KVState:
        return self.states[state_id]

    def materialize(
        self,
        tokens: int,
        owner: Optional[str] = None,
        recompute_cost_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> KVState:
        start = time.perf_counter()
        block_id = f"block_{uuid.uuid4().hex[:12]}"
        block = KVBlock(
            block_id=block_id,
            layer_id=0,
            token_start=0,
            token_end=max(0, tokens),
            key_shape=(max(0, tokens),),
            value_shape=(max(0, tokens),),
            dtype="mock",
            device=owner or self.config.default_owner_device,
            nbytes=max(0, tokens) * self.config.bytes_per_token,
        )
        state = KVState(
            state_id=f"kv_{uuid.uuid4().hex[:12]}",
            model_id=str((metadata or {}).get("model_id", self.config.model_id)),
            tokenizer_id=str((metadata or {}).get("tokenizer_id", self.config.tokenizer_id)),
            model_config_hash=str((metadata or {}).get("model_config_hash", self.config.model_config_hash)),
            position_encoding=str((metadata or {}).get("position_encoding", self.config.position_encoding)),
            blocks=[block],
            owner_node=owner or self.config.default_owner_node,
            owner_device=self.config.default_owner_device,
            metadata={
                **(metadata or {}),
                "status": KVStateStatus.MATERIALIZED.value,
                "recompute_cost_ms": recompute_cost_ms,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        state.metadata["materialize_cost_ms"] = elapsed_ms
        self.states[state.state_id] = state
        self.metrics.mark_kv_materialized()
        self.metrics.observe("recompute_cost_ms", recompute_cost_ms)
        self.metrics.observe("kv_materialize_ms", elapsed_ms)
        self.metrics.update_live_memory(self.live_bytes())
        return state

    def transfer(self, state_id: str, target_owner: str, transfer_cost_ms: float = 0.0) -> KVState:
        source = self.get(state_id)
        transferred = source.fork(f"kv_{uuid.uuid4().hex[:12]}")
        transferred.owner_node = target_owner
        transferred.metadata.update(
            {
                "status": KVStateStatus.TRANSFERRED.value,
                "transferred_from": state_id,
                "transfer_cost_ms": transfer_cost_ms,
            }
        )
        self.states[transferred.state_id] = transferred
        source.metadata["reuse_count"] = int(source.metadata.get("reuse_count", 0)) + 1
        self.metrics.mark_kv_reuse()
        self.metrics.observe("transfer_cost_ms", transfer_cost_ms)
        self.metrics.update_live_memory(self.live_bytes())
        return transferred

    def fork(self, state_id: str, owner: Optional[str] = None) -> KVState:
        source = self.get(state_id)
        child = source.fork(f"kv_{uuid.uuid4().hex[:12]}")
        if owner is not None:
            child.owner_node = owner
        child.metadata["status"] = KVStateStatus.MATERIALIZED.value
        self.states[child.state_id] = child
        source.metadata["reuse_count"] = int(source.metadata.get("reuse_count", 0)) + 1
        self.metrics.mark_kv_reuse()
        self.metrics.update_live_memory(self.live_bytes())
        return child

    def restricted_merge(self, state_ids: Iterable[str], token_limit: Optional[int] = None) -> KVState:
        parents = [self.get(state_id) for state_id in state_ids]
        if not parents:
            return self.materialize(tokens=0, metadata={"merge": "empty"})
        first = parents[0]
        for parent in parents[1:]:
            if not first.is_compatible(parent):
                raise StateCompatibilityError(
                    f"Cannot merge incompatible KV states: {first.state_id} and {parent.state_id}"
                )
        selected_blocks = [block for parent in parents for block in parent.blocks]
        if token_limit is not None:
            remaining = max(0, token_limit)
            limited_blocks = []
            for block in selected_blocks:
                block_tokens = max(0, block.token_end - block.token_start)
                if remaining <= 0:
                    break
                if block_tokens <= remaining:
                    limited_blocks.append(block)
                    remaining -= block_tokens
                else:
                    limited_blocks.append(
                        KVBlock(
                            block_id=f"{block.block_id}_limited",
                            layer_id=block.layer_id,
                            token_start=block.token_start,
                            token_end=block.token_start + remaining,
                            key_shape=block.key_shape,
                            value_shape=block.value_shape,
                            dtype=block.dtype,
                            device=block.device,
                            nbytes=remaining * self.config.bytes_per_token,
                            key_ref=block.key_ref,
                            value_ref=block.value_ref,
                        )
                    )
                    remaining = 0
            selected_blocks = limited_blocks
        merged = KVState(
            state_id=f"kv_{uuid.uuid4().hex[:12]}",
            model_id=first.model_id,
            tokenizer_id=first.tokenizer_id,
            model_config_hash=first.model_config_hash,
            position_encoding=first.position_encoding,
            blocks=selected_blocks,
            lineage=[parent.state_id for parent in parents],
            owner_node=first.owner_node,
            owner_device=first.owner_device,
            metadata={
                "status": KVStateStatus.MERGED.value,
                "merge": "restricted",
                "token_limit": token_limit,
            },
        )
        self.states[merged.state_id] = merged
        for parent in parents:
            parent.metadata["reuse_count"] = int(parent.metadata.get("reuse_count", 0)) + 1
        self.metrics.mark_kv_reuse()
        self.metrics.update_live_memory(self.live_bytes())
        return merged

    def evict(self, state_id: str) -> KVState:
        state = self.get(state_id)
        state.metadata["status"] = KVStateStatus.EVICTED.value
        self.metrics.update_live_memory(self.live_bytes())
        return state

    def live_bytes(self) -> int:
        return sum(
            state.total_bytes()
            for state in self.states.values()
            if state.metadata.get("status") != KVStateStatus.EVICTED.value
        )
