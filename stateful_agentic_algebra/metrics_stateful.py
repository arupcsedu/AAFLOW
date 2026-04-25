"""Metrics for Stateful Agentic Algebra experiments.

Paper mapping:
  - TTFT, total latency, prefill, decode, transfer, resume, and Omega are
    recorded as explicit run-level timing fields.
  - KV state object lifecycle is counted through materialize, fork, merge,
    transfer, and evict events, with total/peak/transferred byte accounting.
  - Throughput, memory, reuse ratio, workload shape, baseline identity, run id,
    and seed are exported in one row per run for JSON/CSV aggregation.
  - Output agreement is computed when multiple baseline/agent text outputs are
    supplied; otherwise it remains null.

The recorder preserves the package's earlier `observe`, `span`, `mark_*`, and
`summary` APIs while adding the requested run-level collector methods.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


METRIC_FIELDS = [
    "ttft_sec",
    "total_latency_sec",
    "prefill_sec",
    "decode_sec",
    "transfer_sec",
    "resume_sec",
    "omega_sec",
    "throughput_tokens_per_sec",
    "kv_total_bytes",
    "kv_peak_bytes",
    "kv_transferred_bytes",
    "kv_reuse_ratio",
    "transfer_count",
    "materialize_count",
    "fork_count",
    "merge_count",
    "evict_count",
    "num_agents",
    "branch_factor",
    "context_tokens",
    "output_tokens",
    "baseline_name",
    "workload_name",
    "run_id",
    "seed",
    "output_agreement_rate",
]


@dataclass
class MetricSeries:
    """In-memory metric series with basic aggregate helpers."""

    count: int = 0
    total_ms: float = 0.0
    samples_ms: List[float] = field(default_factory=list)

    def observe(self, value_ms: float, store_sample: bool = True) -> None:
        self.count += 1
        self.total_ms += float(value_ms)
        if store_sample:
            self.samples_ms.append(float(value_ms))

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class MetricEvent:
    """One metric event captured during a run."""

    name: str
    duration_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StatefulMetricsRecorder:
    """Run-level metrics collector for stateful agentic experiments."""

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    baseline_name: Optional[str] = None
    workload_name: Optional[str] = None
    seed: Optional[int] = None
    run_dir: Optional[str | Path] = None
    series: Dict[str, MetricSeries] = field(default_factory=dict)
    events: list[MetricEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    completed_ops: int = 0
    kv_reuse_hits: int = 0
    kv_materializations: int = 0
    peak_live_kv_bytes: int = 0
    current_live_kv_bytes: int = 0

    def __post_init__(self) -> None:
        if self.run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = Path("runs") / "stateful" / stamp
        else:
            self.run_dir = Path(self.run_dir)

    def record_event(
        self,
        name: str,
        duration: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a timed event with optional metrics metadata.

        Durations are seconds. Metadata may include workload shape fields,
        byte counts, token counts, text outputs, or explicit override fields
        matching `METRIC_FIELDS`.
        """

        payload = dict(metadata or {})
        event = MetricEvent(name=name, duration_sec=float(duration or 0.0), metadata=payload)
        self.events.append(event)
        self._apply_event_to_compat_state(event)

    def observe(self, name: str, value_ms: float, store_sample: bool = True) -> None:
        """Compatibility API: observe a millisecond-valued metric series."""

        if name not in self.series:
            self.series[name] = MetricSeries()
        self.series[name].observe(value_ms, store_sample=store_sample)

    @contextmanager
    def span(self, name: str, store_sample: bool = True) -> Iterator[None]:
        """Compatibility API: time a block and store milliseconds."""

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
        self.current_live_kv_bytes = max(0, int(live_bytes))
        self.peak_live_kv_bytes = max(self.peak_live_kv_bytes, self.current_live_kv_bytes)

    def throughput_ops_per_s(self, now: Optional[float] = None) -> float:
        end = now if now is not None else time.perf_counter()
        elapsed = max(end - self.started_at, 1e-9)
        return self.completed_ops / elapsed

    def reuse_ratio(self) -> float:
        denom = max(1, self.kv_reuse_hits + self.kv_materializations)
        return self.kv_reuse_hits / denom

    def summarize(self) -> dict[str, Any]:
        """Return one run-level metrics row."""

        explicit: dict[str, Any] = {}
        counters = Counter()
        totals = {
            "ttft_sec": 0.0,
            "total_latency_sec": 0.0,
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "transfer_sec": 0.0,
            "resume_sec": 0.0,
            "omega_sec": 0.0,
        }
        kv_total_bytes = 0
        kv_peak_bytes = self.peak_live_kv_bytes
        kv_transferred_bytes = 0
        output_texts: list[str] = []

        for event in self.events:
            metadata = event.metadata
            for field_name in METRIC_FIELDS:
                if field_name in metadata:
                    explicit[field_name] = metadata[field_name]

            metric_key = _duration_metric_key(event.name)
            if metric_key is not None:
                if metric_key == "ttft_sec":
                    totals[metric_key] = max(totals[metric_key], event.duration_sec)
                else:
                    totals[metric_key] += event.duration_sec

            lifecycle = _lifecycle_counter_key(event.name)
            if lifecycle is not None:
                counters[lifecycle] += 1

            bytes_value = _first_int(metadata, "kv_bytes", "bytes", "nbytes", "total_bytes")
            if bytes_value is not None and event.name.lower() in {"materialize", "kv_materialize", "prefill"}:
                kv_total_bytes += bytes_value
                kv_peak_bytes = max(kv_peak_bytes, kv_total_bytes)
            transfer_bytes = _first_int(metadata, "kv_transferred_bytes", "transfer_bytes", "bytes_moved")
            if transfer_bytes is not None and "transfer" in event.name.lower():
                kv_transferred_bytes += transfer_bytes
            live_bytes = _first_int(metadata, "kv_peak_bytes", "live_bytes", "kv_live_bytes")
            if live_bytes is not None:
                kv_peak_bytes = max(kv_peak_bytes, live_bytes)

            text = metadata.get("output_text")
            if isinstance(text, str):
                output_texts.append(text)
            texts = metadata.get("output_texts")
            if isinstance(texts, list):
                output_texts.extend(str(item) for item in texts if item is not None)

        self._fold_series_into_totals(totals)

        output_tokens = int(explicit.get("output_tokens", _sum_metadata_ints(self.events, "output_tokens")))
        total_latency = float(explicit.get("total_latency_sec", totals["total_latency_sec"]))
        if total_latency <= 0:
            total_latency = (
                totals["prefill_sec"]
                + totals["decode_sec"]
                + totals["transfer_sec"]
                + totals["resume_sec"]
                + totals["omega_sec"]
            )
        throughput = explicit.get("throughput_tokens_per_sec")
        if throughput is None:
            throughput = output_tokens / total_latency if total_latency > 0 else 0.0

        materialize_count = int(explicit.get("materialize_count", counters["materialize_count"] or self.kv_materializations))
        reuse_ratio = explicit.get("kv_reuse_ratio")
        if reuse_ratio is None:
            denom = max(1, self.kv_reuse_hits + materialize_count)
            reuse_ratio = self.kv_reuse_hits / denom

        row: dict[str, Any] = {
            "ttft_sec": float(explicit.get("ttft_sec", totals["ttft_sec"])),
            "total_latency_sec": float(total_latency),
            "prefill_sec": float(explicit.get("prefill_sec", totals["prefill_sec"])),
            "decode_sec": float(explicit.get("decode_sec", totals["decode_sec"])),
            "transfer_sec": float(explicit.get("transfer_sec", totals["transfer_sec"])),
            "resume_sec": float(explicit.get("resume_sec", totals["resume_sec"])),
            "omega_sec": float(explicit.get("omega_sec", totals["omega_sec"])),
            "throughput_tokens_per_sec": float(throughput),
            "kv_total_bytes": int(explicit.get("kv_total_bytes", kv_total_bytes or self.current_live_kv_bytes)),
            "kv_peak_bytes": int(explicit.get("kv_peak_bytes", kv_peak_bytes)),
            "kv_transferred_bytes": int(explicit.get("kv_transferred_bytes", kv_transferred_bytes)),
            "kv_reuse_ratio": float(reuse_ratio),
            "transfer_count": int(explicit.get("transfer_count", counters["transfer_count"])),
            "materialize_count": materialize_count,
            "fork_count": int(explicit.get("fork_count", counters["fork_count"])),
            "merge_count": int(explicit.get("merge_count", counters["merge_count"])),
            "evict_count": int(explicit.get("evict_count", counters["evict_count"])),
            "num_agents": int(explicit.get("num_agents", 0)),
            "branch_factor": int(explicit.get("branch_factor", 0)),
            "context_tokens": int(explicit.get("context_tokens", 0)),
            "output_tokens": output_tokens,
            "baseline_name": explicit.get("baseline_name", self.baseline_name),
            "workload_name": explicit.get("workload_name", self.workload_name),
            "run_id": str(explicit.get("run_id", self.run_id)),
            "seed": explicit.get("seed", self.seed),
            "output_agreement_rate": explicit.get("output_agreement_rate", _agreement_rate(output_texts)),
        }
        return row

    def summary(self) -> Dict[str, Any]:
        """Compatibility summary plus the new run-level schema."""

        out = self.summarize()
        out.update(
            {
                "throughput_ops_per_s": self.throughput_ops_per_s(),
                "peak_live_kv_bytes": float(self.peak_live_kv_bytes),
                "reuse_ratio": self.reuse_ratio(),
                "completed_ops": float(self.completed_ops),
                "kv_reuse_hits": float(self.kv_reuse_hits),
                "kv_materializations": float(self.kv_materializations),
            }
        )
        for name, entry in self.series.items():
            key = "omega_ms" if name == "framework_overhead_ms" else name
            out[f"{key}_avg"] = entry.avg_ms
            out[f"{key}_total"] = entry.total_ms
            out[f"{key}_count"] = float(entry.count)
        return out

    def to_json(self, path: Optional[str | Path] = None) -> Path:
        """Write run metrics JSON and return the output path."""

        output = self._resolve_output_path(path, "metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summarize(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    def to_csv(self, path: Optional[str | Path] = None) -> Path:
        """Write one-row run metrics CSV and return the output path."""

        output = self._resolve_output_path(path, "metrics.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        row = self.summarize()
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerow({field: row.get(field) for field in METRIC_FIELDS})
        return output

    @staticmethod
    def aggregate_runs(input_dir: str | Path, output_json: str | Path, output_csv: str | Path) -> dict[str, Any]:
        """Aggregate run JSON files under `input_dir` into JSON and CSV outputs."""

        root = Path(input_dir)
        rows = []
        for path in sorted(root.rglob("*.json")):
            if Path(output_json).resolve() == path.resolve():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            row = _coerce_metric_row(payload)
            if row is not None:
                rows.append(row)

        json_path = Path(output_json)
        csv_path = Path(output_csv)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate = {"runs": rows, "count": len(rows)}
        json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in METRIC_FIELDS})
        return aggregate

    def _resolve_output_path(self, path: Optional[str | Path], default_name: str) -> Path:
        if path is None:
            return Path(self.run_dir or Path("runs") / "stateful" / self.run_id) / default_name
        output = Path(path)
        if output.suffix:
            return output
        return output / default_name

    def _apply_event_to_compat_state(self, event: MetricEvent) -> None:
        name = event.name.lower()
        metadata = event.metadata
        if "reuse" in name:
            self.mark_kv_reuse()
        if name in {"materialize", "kv_materialize"}:
            self.mark_kv_materialized()
        if "completed_op" in name:
            self.mark_completed_op()
        live_bytes = _first_int(metadata, "live_bytes", "kv_live_bytes", "kv_peak_bytes")
        if live_bytes is not None:
            self.update_live_memory(live_bytes)

    def _fold_series_into_totals(self, totals: dict[str, float]) -> None:
        series_map = {
            "ttft_ms": "ttft_sec",
            "prefill_ms": "prefill_sec",
            "recompute_cost_ms": "prefill_sec",
            "decode_ms": "decode_sec",
            "transfer_cost_ms": "transfer_sec",
            "resume_ms": "resume_sec",
            "framework_overhead_ms": "omega_sec",
        }
        for name, key in series_map.items():
            entry = self.series.get(name)
            if entry is None:
                continue
            value_sec = entry.total_ms / 1000.0
            if key == "ttft_sec":
                totals[key] = max(totals[key], value_sec)
            else:
                totals[key] += value_sec


def aggregate_runs(input_dir: str | Path, output_json: str | Path, output_csv: str | Path) -> dict[str, Any]:
    """Module-level convenience wrapper for run aggregation."""

    return StatefulMetricsRecorder.aggregate_runs(input_dir, output_json, output_csv)


def _duration_metric_key(name: str) -> Optional[str]:
    normalized = name.lower().replace("-", "_").replace(".", "_")
    if normalized in {"ttft", "time_to_first_token"}:
        return "ttft_sec"
    if normalized in {"total_latency", "latency", "run", "request"}:
        return "total_latency_sec"
    if normalized in {"prefill", "materialize", "kv_materialize", "recompute"}:
        return "prefill_sec"
    if normalized in {"decode", "generate", "llm_generate"}:
        return "decode_sec"
    if normalized in {"transfer", "kv_transfer"}:
        return "transfer_sec"
    if normalized in {"resume", "kv_resume"}:
        return "resume_sec"
    if normalized in {"omega", "framework_overhead", "framework_overhead_ms"}:
        return "omega_sec"
    return None


def _lifecycle_counter_key(name: str) -> Optional[str]:
    normalized = name.lower().replace("-", "_").replace(".", "_")
    if normalized in {"transfer", "kv_transfer"}:
        return "transfer_count"
    if normalized in {"materialize", "kv_materialize"}:
        return "materialize_count"
    if normalized in {"fork", "kv_fork"}:
        return "fork_count"
    if normalized in {"merge", "kv_merge", "restricted_merge"}:
        return "merge_count"
    if normalized in {"evict", "kv_evict"}:
        return "evict_count"
    return None


def _first_int(metadata: dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return max(0, int(value))
    return None


def _sum_metadata_ints(events: list[MetricEvent], key: str) -> int:
    total = 0
    for event in events:
        value = event.metadata.get(key)
        if value is not None:
            total += int(value)
    return total


def _agreement_rate(texts: list[str]) -> Optional[float]:
    if len(texts) < 2:
        return None
    normalized = [text.strip().lower() for text in texts]
    if not normalized:
        return None
    counts = Counter(normalized)
    return max(counts.values()) / len(normalized)


def _coerce_metric_row(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        payload = payload["metrics"]
    if "runs" in payload:
        return None
    if not any(field in payload for field in METRIC_FIELDS):
        return None
    return {field: payload.get(field) for field in METRIC_FIELDS}
