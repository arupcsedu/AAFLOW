"""State and plan objects for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: `OperatorSpec`, `OperatorType`, and
    `WorkflowState` describe algebra nodes and their state dependencies.
  - KV state object: `KVState` is the explicit materialized KV-cache object.
  - KV materialize / transfer / fork / restricted merge / evict: represented by
    `OperatorType` values and consumed by operators/runtime.
  - Metrics fields: per-state accounting stores TTFT, transfer cost, recompute
    cost, throughput, memory bytes, reuse count, and framework overhead Omega.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class KVStateStatus(str, Enum):
    """Lifecycle state for an explicit KV object."""

    PLANNED = "planned"
    MATERIALIZED = "materialized"
    TRANSFERRED = "transferred"
    EVICTED = "evicted"
    MERGED = "merged"


class OperatorType(str, Enum):
    """Algebra operator kinds used in the paper abstraction."""

    RETRIEVE = "retrieve"
    GENERATE = "generate"
    MATERIALIZE = "materialize"
    TRANSFER = "transfer"
    FORK = "fork"
    RESTRICTED_MERGE = "restricted_merge"
    EVICT = "evict"
    TOOL = "tool"
    NOOP = "noop"


@dataclass
class KVState:
    """Explicit KV state object with lineage and metric counters."""

    state_id: str = field(default_factory=lambda: f"kv_{uuid.uuid4().hex[:12]}")
    parent_ids: List[str] = field(default_factory=list)
    owner: str = "local"
    tokens: int = 0
    bytes_size: int = 0
    status: KVStateStatus = KVStateStatus.PLANNED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reuse_count: int = 0
    materialize_cost_ms: float = 0.0
    transfer_cost_ms: float = 0.0
    recompute_cost_ms: float = 0.0
    ttft_ms: float = 0.0
    framework_overhead_ms: float = 0.0

    def touch(self, status: Optional[KVStateStatus] = None) -> None:
        """Update the modification timestamp and optionally status."""

        if status is not None:
            self.status = status
        self.updated_at = time.time()

    def fork(self, owner: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "KVState":
        """Create a child KV object that shares this state as lineage."""

        child = KVState(
            parent_ids=[self.state_id],
            owner=owner or self.owner,
            tokens=self.tokens,
            bytes_size=self.bytes_size,
            status=KVStateStatus.MATERIALIZED,
            metadata={**self.metadata, **(metadata or {})},
        )
        child.framework_overhead_ms = self.framework_overhead_ms
        return child

    def reuse_ratio(self) -> float:
        """Return a simple reuse ratio normalized by lineage size."""

        denom = max(1, len(self.parent_ids) + 1)
        return self.reuse_count / denom


@dataclass
class OperatorSpec:
    """Serializable operator description used by compiler and scheduler."""

    name: str
    op_type: OperatorType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Set[str] = field(default_factory=set)


@dataclass
class WorkflowState:
    """Mutable state shared by algebra operators during one workflow run."""

    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}")
    kv_states: Dict[str, KVState] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_trace(self, event: str, **fields: Any) -> None:
        """Append a timestamped trace event."""

        self.trace.append({"ts": time.time(), "event": event, **fields})

    def live_kv_bytes(self) -> int:
        """Return bytes held by non-evicted KV states."""

        return sum(
            state.bytes_size
            for state in self.kv_states.values()
            if state.status != KVStateStatus.EVICTED
        )

