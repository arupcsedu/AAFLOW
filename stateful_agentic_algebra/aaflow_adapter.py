"""Optional adapter for existing AAFLOW metrics and RAG agents.

This module is deliberately import-light. It only imports top-level AAFLOW
modules such as `metrics` or `agents` inside functions, so the stateful package
continues to run standalone when Torch, Transformers, datasets, FAISS, or other
AAFLOW dependencies are unavailable.

Supported AAFLOW output conventions:
  - `latencies*.json`: `metrics.MetricsRecorder.dump_json()` dictionary keyed
    by metric name with `count`, `total_sec`, `avg_ms`, `min_ms`, `max_ms`.
  - `latencies*.csv`: same metric table with columns `name,count,total_sec,...`.
  - `throughput*.json/csv`: output from `export_throughput_json/csv`.
  - benchmark `summary.csv/json` and `full_summary.csv` files.
"""

from __future__ import annotations

import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Optional


def load_existing_metrics(path: str | Path | None = None) -> dict[str, Any]:
    """Load existing AAFLOW metrics or the live global recorder if available.

    Parameters
    ----------
    path:
        Optional file or directory. Directories are scanned recursively for
        common AAFLOW metric files.

    Returns
    -------
    dict
        A standalone payload. Import/file failures are returned as
        `available=False` with a reason instead of raising.
    """

    if path is None:
        try:
            metrics_mod = importlib.import_module("metrics")
            recorder = getattr(metrics_mod, "metrics")
            return {
                "available": True,
                "source": "metrics.metrics",
                "metrics": recorder.summary(),
                "normalized": _normalize_latency_summary(recorder.summary()),
            }
        except Exception as exc:
            return _unavailable(f"Could not import live AAFLOW metrics: {exc}")

    root = Path(path)
    if not root.exists():
        return _unavailable(f"Metrics path does not exist: {root}")

    files = [root] if root.is_file() else sorted(_iter_metric_files(root))
    loaded = []
    for file_path in files:
        parsed = _load_metric_file(file_path)
        if parsed is not None:
            loaded.append(parsed)

    return {
        "available": bool(loaded),
        "source": str(root),
        "files": loaded,
        "normalized": _combine_normalized([item.get("normalized", {}) for item in loaded]),
        "reason": "" if loaded else f"No supported AAFLOW metric files found under {root}",
    }


def run_existing_rag_agent_if_available(
    query: str = "What is the answer?",
    agent: Any = None,
    build_agent_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run an existing `RagAgent` object when one is already available.

    The adapter intentionally does not construct a full AAFLOW pipeline by
    default because that may require local corpora, model downloads, and heavy
    dependencies. Callers can pass a pre-built agent, or pass `build_agent_kwargs`
    with a `builder` callable that returns one.
    """

    start = time.perf_counter()
    try:
        if agent is None and build_agent_kwargs:
            builder = build_agent_kwargs.get("builder")
            if callable(builder):
                agent = builder(**{k: v for k, v in build_agent_kwargs.items() if k != "builder"})
        if agent is None:
            return _unavailable("No pre-built RagAgent or builder callable was provided")

        if hasattr(agent, "generate_answer"):
            answer, debug = agent.generate_answer(query)
        elif hasattr(agent, "build_context") and hasattr(agent, "llm"):
            context, debug = agent.build_context(query)
            answer = agent.llm.generate(prompt=query, extra_context=context)
        else:
            return _unavailable("Provided object does not look like an AAFLOW RagAgent")

        metrics_payload = load_existing_metrics()
        return {
            "available": True,
            "query": query,
            "answer": answer,
            "debug": debug,
            "elapsed_sec": time.perf_counter() - start,
            "aaflow_metrics": metrics_payload,
        }
    except Exception as exc:
        return _unavailable(f"AAFLOW RagAgent execution failed: {exc}")


def export_in_aaflow_style(
    stateful_results: list[dict[str, Any]] | dict[str, Any],
    output_dir: str | Path,
    tag: str = "stateful",
) -> dict[str, str]:
    """Export stateful results using familiar AAFLOW output conventions.

    Writes:
      - `{tag}_summary.csv`
      - `{tag}_summary.json`
      - `latencies_{tag}.csv`
      - `latencies_{tag}.json`
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _extract_rows(stateful_results)

    summary_csv = out / f"{tag}_summary.csv"
    summary_json = out / f"{tag}_summary.json"
    latency_csv = out / f"latencies_{tag}.csv"
    latency_json = out / f"latencies_{tag}.json"

    _write_summary_csv(summary_csv, rows)
    summary_json.write_text(json.dumps({"results": rows}, indent=2, sort_keys=True), encoding="utf-8")

    latency_summary = _rows_to_latency_summary(rows)
    latency_json.write_text(json.dumps(latency_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_latency_csv(latency_csv, latency_summary)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "latency_csv": str(latency_csv),
        "latency_json": str(latency_json),
    }


def _iter_metric_files(root: Path):
    patterns = [
        "latencies*.json",
        "latencies*.csv",
        "throughput*.json",
        "throughput*.csv",
        "summary.json",
        "summary.csv",
        "full_summary.csv",
        "llm_generation_stats*.json",
    ]
    seen = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path not in seen:
                seen.add(path)
                yield path


def _load_metric_file(path: Path) -> Optional[dict[str, Any]]:
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                payload = list(csv.DictReader(handle))
        else:
            return None
    except Exception as exc:
        return {"path": str(path), "available": False, "reason": str(exc), "normalized": {}}

    return {
        "path": str(path),
        "available": True,
        "payload": payload,
        "normalized": _normalize_payload(path, payload),
    }


def _normalize_payload(path: Path, payload: Any) -> dict[str, float]:
    name = path.name
    if name.startswith("latencies") and isinstance(payload, dict):
        return _normalize_latency_summary(payload)
    if name.startswith("latencies") and isinstance(payload, list):
        return _normalize_latency_rows(payload)
    if name.startswith("throughput"):
        return _normalize_throughput(payload)
    if "summary" in name:
        return {"summary_rows": float(len(payload) if isinstance(payload, list) else 1)}
    if name.startswith("llm_generation_stats") and isinstance(payload, dict):
        return _numeric_dict(payload, prefix="llm")
    return {}


def _normalize_latency_summary(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric_name, entry in summary.items():
        if isinstance(entry, dict):
            if "total_sec" in entry:
                out[f"{metric_name}.total_sec"] = float(entry["total_sec"])
            if "avg_ms" in entry:
                out[f"{metric_name}.avg_ms"] = float(entry["avg_ms"])
            if "count" in entry:
                out[f"{metric_name}.count"] = float(entry["count"])
    return out


def _normalize_latency_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        summary[str(name)] = row
    return _normalize_latency_summary(summary)


def _normalize_throughput(payload: Any) -> dict[str, float]:
    out = {}
    if isinstance(payload, dict):
        for level, stats in payload.items():
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        out[f"throughput.{level}.{key}"] = float(value)
    elif isinstance(payload, list):
        for row in payload:
            level = row.get("level")
            for key in ("count", "avg_latency_s", "throughput_per_s"):
                value = row.get(key)
                if level and value not in (None, ""):
                    out[f"throughput.{level}.{key}"] = float(value)
    return out


def _numeric_dict(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            out[f"{prefix}.{key}"] = float(value)
    return out


def _combine_normalized(items: list[dict[str, Any]]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                combined[key] = combined.get(key, 0.0) + float(value)
    return combined


def _extract_rows(stateful_results: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(stateful_results, dict):
        if isinstance(stateful_results.get("results"), list):
            return [dict(row) for row in stateful_results["results"]]
        if isinstance(stateful_results.get("metrics"), dict):
            return [dict(stateful_results["metrics"])]
        return [dict(stateful_results)]
    return [dict(row) for row in stateful_results]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rows_to_latency_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    mapping = {
        "ttft_sec": "stateful.ttft",
        "total_latency_sec": "stateful.total_latency",
        "prefill_sec": "stateful.prefill",
        "decode_sec": "stateful.decode",
        "transfer_sec": "stateful.transfer",
        "resume_sec": "stateful.resume",
        "omega_sec": "stateful.omega",
    }
    summary = {}
    for source_key, metric_name in mapping.items():
        values = [float(row[source_key]) for row in rows if row.get(source_key) not in (None, "")]
        if not values:
            continue
        total = sum(values)
        summary[metric_name] = {
            "count": len(values),
            "total_sec": total,
            "avg_ms": (total / len(values)) * 1000.0,
            "min_ms": min(values) * 1000.0,
            "max_ms": max(values) * 1000.0,
        }
    return summary


def _write_latency_csv(path: Path, summary: dict[str, dict[str, float]]) -> None:
    fieldnames = ["name", "count", "total_sec", "avg_ms", "min_ms", "max_ms"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in summary.items():
            writer.writerow({"name": name, **stats})


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "metrics": {}, "normalized": {}}
