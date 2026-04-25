"""State objects for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: `OperatorSpec`, `OperatorType`, and
    `WorkflowState` describe typed algebra nodes and workflow state.
  - KV state object: `KVBlock` and `KVState` represent explicit KV-cache state
    without requiring real model tensors.
  - KV materialize / transfer / fork / restricted merge / evict: lifecycle
    operations in `kv_manager.py` operate on these serializable state objects.
  - Metrics: state metadata can carry TTFT, transfer cost, recompute cost,
    throughput, memory, reuse ratio, and framework overhead Omega annotations.

`key_ref` and `value_ref` intentionally accept any object. They may point to
torch tensors, numpy arrays, shared-memory buffers, remote object references, or
mock buffers. Serialization omits those references because first-pass state
metadata must be portable even when real KV tensors are absent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StateCompatibilityError(Exception):
    """Raised when two KV states cannot be safely reused or merged."""


class KVStateStatus(str, Enum):
    """Lightweight lifecycle label used by the package runtime."""

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
class KVBlock:
    """One layer/token interval of KV-cache state.

    The object stores metadata and optional opaque references only. Real KV
    tensors are not required for simulation, planning, or JSON round-tripping.
    """

    block_id: str
    layer_id: int
    token_start: int
    token_end: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    dtype: str
    device: str
    nbytes: int
    key_ref: Optional[Any] = None
    value_ref: Optional[Any] = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata representation."""

        return {
            "block_id": self.block_id,
            "layer_id": self.layer_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "dtype": self.dtype,
            "device": self.device,
            "nbytes": self.nbytes,
            "key_ref": None,
            "value_ref": None,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "KVBlock":
        """Create a block from a JSON dictionary."""

        return cls(
            block_id=str(data["block_id"]),
            layer_id=int(data["layer_id"]),
            token_start=int(data["token_start"]),
            token_end=int(data["token_end"]),
            key_shape=tuple(int(x) for x in data["key_shape"]),
            value_shape=tuple(int(x) for x in data["value_shape"]),
            dtype=str(data["dtype"]),
            device=str(data["device"]),
            nbytes=int(data["nbytes"]),
            key_ref=None,
            value_ref=None,
        )


@dataclass
class KVState:
    """Explicit KV-cache state with compatibility metadata and lineage."""

    state_id: str
    model_id: str
    tokenizer_id: str
    model_config_hash: str
    position_encoding: str
    blocks: list[KVBlock] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
    owner_node: str = "local"
    owner_device: str = "cpu"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_bytes(self) -> int:
        """Return total bytes across all KV blocks."""

        return sum(block.nbytes for block in self.blocks)

    def token_span(self) -> tuple[int, int]:
        """Return the half-open token interval covered by this state."""

        if not self.blocks:
            return (0, 0)
        return (
            min(block.token_start for block in self.blocks),
            max(block.token_end for block in self.blocks),
        )

    def is_compatible(self, other: "KVState") -> bool:
        """Check whether another state can share/reuse this state's KV cache."""

        return (
            self.model_id == other.model_id
            and self.tokenizer_id == other.tokenizer_id
            and self.model_config_hash == other.model_config_hash
            and self.position_encoding == other.position_encoding
        )

    def fork(self, new_state_id: str) -> "KVState":
        """Create a new state id with the same blocks and parent lineage."""

        return KVState(
            state_id=new_state_id,
            model_id=self.model_id,
            tokenizer_id=self.tokenizer_id,
            model_config_hash=self.model_config_hash,
            position_encoding=self.position_encoding,
            blocks=list(self.blocks),
            lineage=[*self.lineage, self.state_id],
            owner_node=self.owner_node,
            owner_device=self.owner_device,
            created_at=time.time(),
            metadata=dict(self.metadata),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state dictionary."""

        return {
            "state_id": self.state_id,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_config_hash": self.model_config_hash,
            "position_encoding": self.position_encoding,
            "blocks": [block.to_json_dict() for block in self.blocks],
            "lineage": list(self.lineage),
            "owner_node": self.owner_node,
            "owner_device": self.owner_device,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "KVState":
        """Create a state from a JSON dictionary."""

        return cls(
            state_id=str(data["state_id"]),
            model_id=str(data["model_id"]),
            tokenizer_id=str(data["tokenizer_id"]),
            model_config_hash=str(data["model_config_hash"]),
            position_encoding=str(data["position_encoding"]),
            blocks=[KVBlock.from_json_dict(item) for item in data.get("blocks", [])],
            lineage=[str(item) for item in data.get("lineage", [])],
            owner_node=str(data.get("owner_node", "local")),
            owner_device=str(data.get("owner_device", "cpu")),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OperatorSpec:
    """Serializable operator description used by compiler and scheduler."""

    name: str
    op_type: OperatorType
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: set[str] = field(default_factory=set)


@dataclass
class WorkflowState:
    """Mutable state shared by algebra operators during one workflow run."""

    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}")
    kv_states: dict[str, KVState] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def add_trace(self, event: str, **fields: Any) -> None:
        """Append a timestamped trace event."""

        self.trace.append({"ts": time.time(), "event": event, **fields})

    def live_kv_bytes(self) -> int:
        """Return bytes held by non-evicted KV states."""

        return sum(
            state.total_bytes()
            for state in self.kv_states.values()
            if state.metadata.get("status") != KVStateStatus.EVICTED.value
        )
