"""Metrics for Stateful Agentic Algebra experiments.

Paper mapping:
  - TTFT: observed as `ttft_ms`.
  - Transfer cost: observed as `transfer_cost_ms`.
  - Recompute cost: observed as `recompute_cost_ms`.
  - Throughput: derived from completed operations per elapsed second.
  - Memory: tracked as live KV bytes.
  - Reuse ratio: computed from KV reuse hits vs materializations.
  - Framework overhead Omega: observed as `framework_overhead_ms` and reported
    as `omega_ms`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class MetricSeries:
    """In-memory metric series with basic aggregate helpers."""

    count: int = 0
    total_ms: float = 0.0
    samples_ms: List[float] = field(default_factory=list)

    def observe(self, value_ms: float, store_sample: bool = True) -> None:
        self.count += 1
        self.total_ms += value_ms
        if store_sample:
            self.samples_ms.append(value_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class StatefulMetricsRecorder:
    """Recorder specialized for stateful KV and operator metrics."""

    series: Dict[str, MetricSeries] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    completed_ops: int = 0
    kv_reuse_hits: int = 0
    kv_materializations: int = 0
    peak_live_kv_bytes: int = 0

    def observe(self, name: str, value_ms: float, store_sample: bool = True) -> None:
        if name not in self.series:
            self.series[name] = MetricSeries()
        self.series[name].observe(value_ms, store_sample=store_sample)

    @contextmanager
    def span(self, name: str, store_sample: bool = True) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe(name, elapsed_ms, store_sample=store_sample)

    def mark_completed_op(self) -> None:
        self.completed_ops += 1

    def mark_kv_reuse(self) -> None:
        self.kv_reuse_hits += 1

    def mark_kv_materialized(self) -> None:
        self.kv_materializations += 1

    def update_live_memory(self, live_bytes: int) -> None:
        self.peak_live_kv_bytes = max(self.peak_live_kv_bytes, live_bytes)

    def throughput_ops_per_s(self, now: Optional[float] = None) -> float:
        end = now if now is not None else time.perf_counter()
        elapsed = max(end - self.started_at, 1e-9)
        return self.completed_ops / elapsed

    def reuse_ratio(self) -> float:
        denom = max(1, self.kv_reuse_hits + self.kv_materializations)
        return self.kv_reuse_hits / denom

    def summary(self) -> Dict[str, float]:
        out = {
            "throughput_ops_per_s": self.throughput_ops_per_s(),
            "peak_live_kv_bytes": float(self.peak_live_kv_bytes),
            "reuse_ratio": self.reuse_ratio(),
            "completed_ops": float(self.completed_ops),
            "kv_reuse_hits": float(self.kv_reuse_hits),
            "kv_materializations": float(self.kv_materializations),
        }
        for name, entry in self.series.items():
            key = "omega_ms" if name == "framework_overhead_ms" else name
            out[f"{key}_avg"] = entry.avg_ms
            out[f"{key}_total"] = entry.total_ms
            out[f"{key}_count"] = float(entry.count)
        return out

