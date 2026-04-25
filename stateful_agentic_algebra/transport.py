"""Transport abstraction for KV state movement.

Paper mapping:
  - KV transfer: `Transport` estimates or performs state movement.
  - Transfer cost metric: every transfer returns a measured/simulated cost.
  - Robust imports: UCX, NCCL, CUDA, vLLM, and SGLang are optional; missing
    libraries select `MockTransport`.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Optional


def optional_available(module_name: str) -> bool:
    """Return True when an optional runtime module can be imported."""

    return importlib.util.find_spec(module_name) is not None


@dataclass
class TransferResult:
    """Result of a KV transfer operation."""

    source: str
    target: str
    bytes_moved: int
    transfer_cost_ms: float
    backend: str
    simulated: bool = True


@dataclass
class TransportConfig:
    """Transport configuration with safe defaults."""

    backend: str = "auto"
    bandwidth_gbps: float = 25.0
    fixed_overhead_ms: float = 0.05


class MockTransport:
    """Simulation transport used when UCX/NCCL/CUDA paths are unavailable."""

    name = "mock"

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self.config = config or TransportConfig()

    def transfer(self, source: str, target: str, bytes_moved: int) -> TransferResult:
        bandwidth_bytes_per_ms = max(self.config.bandwidth_gbps, 1e-9) * 1e9 / 8.0 / 1000.0
        simulated_ms = self.config.fixed_overhead_ms + bytes_moved / bandwidth_bytes_per_ms
        time.sleep(min(simulated_ms / 1000.0, 0.001))
        return TransferResult(source, target, bytes_moved, simulated_ms, self.name, simulated=True)


class Transport(MockTransport):
    """Auto-selecting transport facade.

    The class currently simulates all data movement, but exposes which backend
    would be selected when optional libraries are available.
    """

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        super().__init__(config)
        requested = self.config.backend
        if requested != "auto":
            self.name = requested
        elif optional_available("ucp"):
            self.name = "ucx"
        elif optional_available("nccl"):
            self.name = "nccl"
        else:
            self.name = "mock"

