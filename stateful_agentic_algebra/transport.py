"""Transport interfaces for moving KV state.

Paper mapping:
  - KV transfer: transports implement `send_state`, `receive_state`, and
    `estimate_transfer_time` for explicit KV movement.
  - Metrics: each transport records `transfer_bytes`, `transfer_latency_sec`,
    and `transfer_count`.
  - Optional accelerators: `UCXTransport` imports UCX only inside the class and
    raises a clear `RuntimeError` when UCX is not available.

The first implementation moves metadata, not real tensors. `key_ref` and
`value_ref` may point to torch tensors, numpy arrays, remote handles, or mock
buffers, but JSON transport serializes only portable state metadata unless
mock binary output is requested.
"""

from __future__ import annotations

import importlib.util
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .state_objects import KVState, KVStateStatus


def optional_available(module_name: str) -> bool:
    """Return True when an optional runtime module can be imported."""

    return importlib.util.find_spec(module_name) is not None


@dataclass
class TransferMetrics:
    """Aggregate transport metrics."""

    transfer_bytes: int = 0
    transfer_latency_sec: float = 0.0
    transfer_count: int = 0

    def observe(self, bytes_moved: int, latency_sec: float) -> None:
        self.transfer_bytes += int(bytes_moved)
        self.transfer_latency_sec += float(latency_sec)
        self.transfer_count += 1

    def to_json_dict(self) -> dict[str, float]:
        return {
            "transfer_bytes": float(self.transfer_bytes),
            "transfer_latency_sec": self.transfer_latency_sec,
            "transfer_count": float(self.transfer_count),
        }


@dataclass
class TransferResult:
    """Result of a KV transfer operation."""

    source: str
    target: str
    bytes_moved: int
    transfer_cost_ms: float
    backend: str
    simulated: bool = True
    state_id: Optional[str] = None


@dataclass
class TransportConfig:
    """Transport configuration with safe defaults."""

    backend: str = "auto"
    bandwidth_gbps: float = 25.0
    fixed_overhead_ms: float = 0.05
    sleep: bool = False
    root_dir: str = "runs/stateful_transport"
    write_mock_buffers: bool = False


class AbstractTransport(ABC):
    """Abstract transport API for KV state movement."""

    name = "abstract"

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self.config = config or TransportConfig()
        self.metrics = TransferMetrics()
        self._received: dict[str, KVState] = {}

    @abstractmethod
    def send_state(self, kv_state: KVState, target_node: str) -> KVState:
        """Send a state to a target node and return the transferred state."""

    @abstractmethod
    def receive_state(self, state_id: str) -> KVState:
        """Receive or load a transferred state by id."""

    @abstractmethod
    def estimate_transfer_time(self, kv_state: KVState) -> float:
        """Estimate transfer time in seconds."""

    def transfer(self, source: str, target: str, bytes_moved: int) -> TransferResult:
        """Compatibility API used by older operator code."""

        latency_sec = self._estimate_bytes_transfer_time(bytes_moved)
        self._maybe_sleep(latency_sec)
        self.metrics.observe(bytes_moved, latency_sec)
        return TransferResult(
            source=source,
            target=target,
            bytes_moved=bytes_moved,
            transfer_cost_ms=latency_sec * 1000.0,
            backend=self.name,
            simulated=True,
        )

    def _estimate_bytes_transfer_time(self, bytes_moved: int) -> float:
        bandwidth_bytes_per_sec = max(self.config.bandwidth_gbps, 1e-12) * 1e9 / 8.0
        return self.config.fixed_overhead_ms / 1000.0 + max(0, int(bytes_moved)) / bandwidth_bytes_per_sec

    def _maybe_sleep(self, latency_sec: float) -> None:
        if self.config.sleep and latency_sec > 0:
            time.sleep(latency_sec)

    def _transferred_copy(self, kv_state: KVState, target_node: str, latency_sec: float) -> KVState:
        moved = kv_state.fork(f"kv_{uuid.uuid4().hex[:12]}")
        moved.owner_node = target_node
        moved.metadata.update(
            {
                "status": KVStateStatus.TRANSFERRED.value,
                "source_node": kv_state.owner_node,
                "target_node": target_node,
                "transfer_bytes": kv_state.total_bytes(),
                "transfer_latency_sec": latency_sec,
                "transport_backend": self.name,
            }
        )
        return moved


class MockTransport(AbstractTransport):
    """Simulation transport.

    It can sleep for the simulated latency, or only record the latency. It
    updates placement metadata and keeps transferred states in memory.
    """

    name = "mock"

    def send_state(self, kv_state: KVState, target_node: str) -> KVState:
        latency_sec = self.estimate_transfer_time(kv_state)
        self._maybe_sleep(latency_sec)
        moved = self._transferred_copy(kv_state, target_node, latency_sec)
        self.metrics.observe(kv_state.total_bytes(), latency_sec)
        self._received[moved.state_id] = moved
        return moved

    def receive_state(self, state_id: str) -> KVState:
        try:
            return self._received[state_id]
        except KeyError as exc:
            raise KeyError(f"State {state_id!r} has not been sent through {self.name} transport") from exc

    def estimate_transfer_time(self, kv_state: KVState) -> float:
        return self._estimate_bytes_transfer_time(kv_state.total_bytes())


class LocalFileTransport(MockTransport):
    """Transport that writes KVState metadata to JSON files.

    This is useful for reproducibility tests and offline inspection. It can
    optionally write mock binary files sized like each KV block.
    """

    name = "local_file"

    def __init__(self, config: Optional[TransportConfig] = None, root_dir: Optional[str | Path] = None) -> None:
        super().__init__(config)
        self.root_dir = Path(root_dir or self.config.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def send_state(self, kv_state: KVState, target_node: str) -> KVState:
        moved = super().send_state(kv_state, target_node)
        state_dir = self.root_dir / moved.state_id
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(json.dumps(moved.to_json_dict(), indent=2), encoding="utf-8")
        if self.config.write_mock_buffers:
            buffers_dir = state_dir / "buffers"
            buffers_dir.mkdir(parents=True, exist_ok=True)
            for block in moved.blocks:
                (buffers_dir / f"{block.block_id}.bin").write_bytes(b"\0" * max(0, block.nbytes))
        moved.metadata["local_file_path"] = str(state_dir / "state.json")
        (state_dir / "state.json").write_text(json.dumps(moved.to_json_dict(), indent=2), encoding="utf-8")
        self._received[moved.state_id] = moved
        return moved

    def receive_state(self, state_id: str) -> KVState:
        path = self.root_dir / state_id / "state.json"
        if not path.exists():
            return super().receive_state(state_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = KVState.from_json_dict(payload)
        self._received[state_id] = state
        return state


class UCXTransport(AbstractTransport):
    """Optional UCX transport placeholder.

    UCX libraries are imported only inside the class. If UCX is unavailable, a
    clear RuntimeError is raised and tests can continue using MockTransport.
    """

    name = "ucx"

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        super().__init__(config)
        try:
            import ucp  # type: ignore
        except Exception as exc:
            raise RuntimeError("UCXTransport requires the optional 'ucp' package.") from exc
        self.ucp = ucp

    def send_state(self, kv_state: KVState, target_node: str) -> KVState:
        # Metadata-only placeholder until real UCX endpoints are wired in.
        latency_sec = self.estimate_transfer_time(kv_state)
        moved = self._transferred_copy(kv_state, target_node, latency_sec)
        self.metrics.observe(kv_state.total_bytes(), latency_sec)
        self._received[moved.state_id] = moved
        return moved

    def receive_state(self, state_id: str) -> KVState:
        try:
            return self._received[state_id]
        except KeyError as exc:
            raise KeyError(f"State {state_id!r} has not been received by UCXTransport") from exc

    def estimate_transfer_time(self, kv_state: KVState) -> float:
        return self._estimate_bytes_transfer_time(kv_state.total_bytes())


class Transport(MockTransport):
    """Auto-selecting transport facade with mock-safe fallback."""

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        config = config or TransportConfig()
        requested = config.backend
        if requested == "local_file":
            # Rebind this instance to local-file behavior by copying state from
            # a LocalFileTransport. This keeps `Transport(...)` construction
            # backwards compatible while preserving the concrete class API.
            local = LocalFileTransport(config)
            self.__class__ = LocalFileTransport
            self.__dict__ = local.__dict__
            return
        if requested == "ucx":
            ucx = UCXTransport(config)
            self.__class__ = UCXTransport
            self.__dict__ = ucx.__dict__
            return
        super().__init__(config)
        if requested != "auto":
            self.name = requested
        elif optional_available("ucp"):
            # Do not instantiate UCX implicitly. Auto remains mock-safe and
            # labels availability in the name for diagnostics only.
            self.name = "mock"
        else:
            self.name = "mock"

