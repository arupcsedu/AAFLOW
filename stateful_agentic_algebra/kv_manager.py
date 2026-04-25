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
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .metrics_stateful import StatefulMetricsRecorder
from .state_objects import KVState, KVStateStatus


@dataclass
class KVManagerConfig:
    """Configuration for simulated KV memory accounting."""

    bytes_per_token: int = 2048
    default_owner: str = "local"


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
        state = KVState(
            owner=owner or self.config.default_owner,
            tokens=tokens,
            bytes_size=max(0, tokens) * self.config.bytes_per_token,
            status=KVStateStatus.MATERIALIZED,
            metadata=metadata or {},
            recompute_cost_ms=recompute_cost_ms,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        state.materialize_cost_ms = elapsed_ms
        self.states[state.state_id] = state
        self.metrics.mark_kv_materialized()
        self.metrics.observe("recompute_cost_ms", recompute_cost_ms)
        self.metrics.observe("kv_materialize_ms", elapsed_ms)
        self.metrics.update_live_memory(self.live_bytes())
        return state

    def transfer(self, state_id: str, target_owner: str, transfer_cost_ms: float = 0.0) -> KVState:
        source = self.get(state_id)
        transferred = source.fork(owner=target_owner, metadata={"transferred_from": state_id})
        transferred.transfer_cost_ms = transfer_cost_ms
        transferred.touch(KVStateStatus.TRANSFERRED)
        self.states[transferred.state_id] = transferred
        source.reuse_count += 1
        self.metrics.mark_kv_reuse()
        self.metrics.observe("transfer_cost_ms", transfer_cost_ms)
        self.metrics.update_live_memory(self.live_bytes())
        return transferred

    def fork(self, state_id: str, owner: Optional[str] = None) -> KVState:
        source = self.get(state_id)
        child = source.fork(owner=owner)
        self.states[child.state_id] = child
        source.reuse_count += 1
        self.metrics.mark_kv_reuse()
        self.metrics.update_live_memory(self.live_bytes())
        return child

    def restricted_merge(self, state_ids: Iterable[str], token_limit: Optional[int] = None) -> KVState:
        parents = [self.get(state_id) for state_id in state_ids]
        if not parents:
            return self.materialize(tokens=0, metadata={"merge": "empty"})
        total_tokens = sum(parent.tokens for parent in parents)
        if token_limit is not None:
            total_tokens = min(total_tokens, max(0, token_limit))
        merged = KVState(
            parent_ids=[parent.state_id for parent in parents],
            owner=parents[0].owner,
            tokens=total_tokens,
            bytes_size=total_tokens * self.config.bytes_per_token,
            status=KVStateStatus.MERGED,
            metadata={"merge": "restricted", "token_limit": token_limit},
        )
        self.states[merged.state_id] = merged
        for parent in parents:
            parent.reuse_count += 1
        self.metrics.mark_kv_reuse()
        self.metrics.update_live_memory(self.live_bytes())
        return merged

    def evict(self, state_id: str) -> KVState:
        state = self.get(state_id)
        state.touch(KVStateStatus.EVICTED)
        self.metrics.update_live_memory(self.live_bytes())
        return state

    def live_bytes(self) -> int:
        return sum(
            state.bytes_size
            for state in self.states.values()
            if state.status != KVStateStatus.EVICTED
        )

