"""Transfer/recompute crossover analysis using real or metadata KV sizes.

The module compares dense recomputation against local KV reuse and simulated
KV transfer over RDMA/Ethernet-like bandwidths:

    T_transfer = kv_bytes / bandwidth + latency
    T_recompute = measured_prefill_sec

When HuggingFace loading succeeds, KV bytes and prefill timing come from the
real model. Otherwise, metadata estimates are used from common model family
specs so the analysis still runs for gated or not-yet-installed models.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .hf_kv_backend import HFBackendConfig, HFKVBackend


DEFAULT_BANDWIDTHS = {
    "10Gbps": 10e9 / 8,
    "25Gbps": 25e9 / 8,
    "100Gbps": 100e9 / 8,
    "200Gbps": 200e9 / 8,
    "400Gbps": 400e9 / 8,
}
DEFAULT_LATENCIES_SEC = {
    "ethernet_100us": 100e-6,
    "rdma_10us": 10e-6,
}


@dataclass
class ModelKVMetadata:
    """Minimal metadata needed to estimate KV bytes."""

    hidden_size: int
    num_layers: int
    num_key_value_heads: int
    num_attention_heads: int
    dtype_bytes: int = 2
    source: str = "metadata_estimate"

    @property
    def head_dim(self) -> int:
        return max(1, self.hidden_size // max(1, self.num_attention_heads))

    def kv_bytes(self, context_tokens: int) -> int:
        return int(max(1, context_tokens) * self.num_layers * 2 * self.num_key_value_heads * self.head_dim * self.dtype_bytes)


@dataclass
class Measurement:
    """KV/prefill measurement for one model/context."""

    model_id: str
    context_tokens: int
    output_tokens: int
    kv_bytes: int
    prefill_sec: float
    decode_sec: float = 0.0
    source: str = "metadata_estimate"
    reason: str = ""


def analyze_crossover(
    model_id: str,
    context_grid: list[int],
    output_tokens: int,
    bandwidths: dict[str, float],
    latencies: dict[str, float],
    *,
    metadata_only: bool = False,
    device: str = "auto",
    local_files_only: bool = False,
    run_file_transfer: bool = False,
) -> list[dict[str, Any]]:
    """Return crossover rows for every context/bandwidth/latency pair."""

    rows: list[dict[str, Any]] = []
    for context_tokens in context_grid:
        measurement = measure_or_estimate(
            model_id=model_id,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            metadata_only=metadata_only,
            device=device,
            local_files_only=local_files_only,
        )
        local_reuse_sec = measurement.decode_sec
        file_transfer_sec = _measure_file_transfer(measurement.kv_bytes) if run_file_transfer else None
        for bandwidth_name, bandwidth in bandwidths.items():
            for latency_name, latency_sec in latencies.items():
                transfer_sec = measurement.kv_bytes / max(float(bandwidth), 1.0) + float(latency_sec)
                rows.append(
                    {
                        "model_id": model_id,
                        "context_tokens": int(context_tokens),
                        "output_tokens": int(output_tokens),
                        "kv_bytes": int(measurement.kv_bytes),
                        "kv_mib": measurement.kv_bytes / (1024 * 1024),
                        "prefill_sec": measurement.prefill_sec,
                        "decode_sec": measurement.decode_sec,
                        "t_recompute_sec": measurement.prefill_sec,
                        "t_local_reuse_sec": local_reuse_sec,
                        "bandwidth_name": bandwidth_name,
                        "bandwidth_bytes_per_sec": float(bandwidth),
                        "latency_name": latency_name,
                        "latency_sec": float(latency_sec),
                        "t_transfer_sec": transfer_sec,
                        "benefit": transfer_sec < measurement.prefill_sec,
                        "speedup_vs_recompute": (measurement.prefill_sec / transfer_sec) if transfer_sec > 0 else 0.0,
                        "file_transfer_sec": file_transfer_sec,
                        "measurement_source": measurement.source,
                        "reason": measurement.reason,
                    }
                )
    return rows


def measure_or_estimate(
    *,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    metadata_only: bool = False,
    device: str = "auto",
    local_files_only: bool = False,
) -> Measurement:
    """Use HF real measurement when possible, else metadata estimate."""

    if not metadata_only:
        try:
            backend = HFKVBackend(
                HFBackendConfig(
                    model_id=model_id,
                    tokenizer_id=model_id,
                    device=device,
                    local_files_only=local_files_only,
                )
            )
            prompt = backend.build_prompt(context_tokens)
            result = backend.measure(prompt, context_tokens=context_tokens, output_tokens=output_tokens)
            return Measurement(
                model_id=model_id,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                kv_bytes=int(result.metrics.get("kv_total_bytes", result.kv_state.total_bytes())),
                prefill_sec=float(result.metrics.get("prefill_sec", 0.0)),
                decode_sec=float(result.metrics.get("decode_sec", 0.0)),
                source="hf_real",
            )
        except Exception as exc:
            estimate = estimate_measurement(model_id, context_tokens, output_tokens)
            estimate.reason = f"HF measurement unavailable: {exc}"
            return estimate
    return estimate_measurement(model_id, context_tokens, output_tokens)


def estimate_measurement(model_id: str, context_tokens: int, output_tokens: int) -> Measurement:
    """Estimate KV bytes and prefill timing from known model-family metadata."""

    metadata = model_metadata(model_id)
    kv_bytes = metadata.kv_bytes(context_tokens)
    # Conservative synthetic prefill model: larger hidden/layer models pay more.
    model_scale = metadata.num_layers * metadata.hidden_size / (12 * 768)
    prefill_sec = max(1e-6, context_tokens * 0.0002 * model_scale)
    decode_sec = max(0.0, output_tokens * 0.00005 * model_scale)
    return Measurement(
        model_id=model_id,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        kv_bytes=kv_bytes,
        prefill_sec=prefill_sec,
        decode_sec=decode_sec,
        source=metadata.source,
    )


def model_metadata(model_id: str) -> ModelKVMetadata:
    """Return approximate KV metadata for common benchmark models."""

    lowered = model_id.lower()
    if "distilgpt2" in lowered:
        return ModelKVMetadata(hidden_size=768, num_layers=6, num_key_value_heads=12, num_attention_heads=12, dtype_bytes=4)
    if lowered == "gpt2" or lowered.endswith("/gpt2"):
        return ModelKVMetadata(hidden_size=768, num_layers=12, num_key_value_heads=12, num_attention_heads=12, dtype_bytes=4)
    if "mistral-7b" in lowered:
        return ModelKVMetadata(hidden_size=4096, num_layers=32, num_key_value_heads=8, num_attention_heads=32)
    if "qwen2.5-7b" in lowered or "qwen2-7b" in lowered:
        return ModelKVMetadata(hidden_size=3584, num_layers=28, num_key_value_heads=4, num_attention_heads=28)
    if "qwen2.5-32b" in lowered or "qwen2-32b" in lowered:
        return ModelKVMetadata(hidden_size=5120, num_layers=64, num_key_value_heads=8, num_attention_heads=40)
    if "llama-3-8b" in lowered or "meta-llama-3-8b" in lowered:
        return ModelKVMetadata(hidden_size=4096, num_layers=32, num_key_value_heads=8, num_attention_heads=32)
    if "llama-2-7b" in lowered or "llama-7b" in lowered:
        return ModelKVMetadata(hidden_size=4096, num_layers=32, num_key_value_heads=32, num_attention_heads=32)
    if "70b" in lowered:
        return ModelKVMetadata(hidden_size=8192, num_layers=80, num_key_value_heads=8, num_attention_heads=64)
    return ModelKVMetadata(hidden_size=4096, num_layers=32, num_key_value_heads=8, num_attention_heads=32)


def write_outputs(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Write CSV, JSON, and plots."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "crossover.csv", rows)
    (out / "crossover.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    try:
        _plot_transfer_vs_recompute(rows, out / "plot_transfer_vs_recompute")
    except Exception as exc:
        logs = out / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "plots.err").write_text(f"plot generation failed: {exc}\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_id",
        "context_tokens",
        "output_tokens",
        "kv_bytes",
        "kv_mib",
        "prefill_sec",
        "decode_sec",
        "t_recompute_sec",
        "t_local_reuse_sec",
        "bandwidth_name",
        "bandwidth_bytes_per_sec",
        "latency_name",
        "latency_sec",
        "t_transfer_sec",
        "benefit",
        "speedup_vs_recompute",
        "file_transfer_sec",
        "measurement_source",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_transfer_vs_recompute(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    contexts = sorted({int(row["context_tokens"]) for row in rows})
    recompute = {
        int(row["context_tokens"]): float(row["t_recompute_sec"])
        for row in rows
    }
    ax.plot(contexts, [recompute[x] for x in contexts], marker="s", linewidth=2.2, label="dense recompute")

    series: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (str(row["bandwidth_name"]), str(row["latency_name"]))
        series.setdefault(key, {})[int(row["context_tokens"])] = float(row["t_transfer_sec"])
    for (bandwidth_name, latency_name), values in sorted(series.items()):
        label = f"{bandwidth_name} {latency_name}"
        ax.plot(contexts, [values[x] for x in contexts], marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Time (s)")
    ax.set_title("KV Transfer vs Dense Recompute")
    ax.grid(True, axis="both", color="#D7DCE2", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def _measure_file_transfer(nbytes: int) -> float:
    """Measure a local file copy of a sparse mock buffer."""

    with tempfile.TemporaryDirectory(prefix="saa_transfer_") as tmp:
        source = Path(tmp) / "source.bin"
        target = Path(tmp) / "target.bin"
        with source.open("wb") as handle:
            handle.truncate(max(0, int(nbytes)))
        start = time.perf_counter()
        shutil.copyfile(source, target)
        return time.perf_counter() - start


def parse_bandwidths(raw: str) -> dict[str, float]:
    if not raw:
        return dict(DEFAULT_BANDWIDTHS)
    parsed: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            parsed[name.strip()] = _bandwidth_to_bytes_per_sec(value.strip())
        else:
            parsed[item] = _bandwidth_to_bytes_per_sec(item)
    return parsed or dict(DEFAULT_BANDWIDTHS)


def parse_latencies(raw: str) -> dict[str, float]:
    if not raw:
        return dict(DEFAULT_LATENCIES_SEC)
    parsed: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            parsed[name.strip()] = _latency_to_sec(value.strip())
        else:
            parsed[item] = _latency_to_sec(item)
    return parsed or dict(DEFAULT_LATENCIES_SEC)


def _bandwidth_to_bytes_per_sec(value: str) -> float:
    text = value.strip().lower()
    multiplier = 1.0
    if text.endswith("gbps"):
        multiplier = 1e9 / 8
        text = text[:-4]
    elif text.endswith("mbps"):
        multiplier = 1e6 / 8
        text = text[:-4]
    elif text.endswith("gb/s"):
        multiplier = 1e9
        text = text[:-4]
    elif text.endswith("mb/s"):
        multiplier = 1e6
        text = text[:-4]
    return float(text) * multiplier


def _latency_to_sec(value: str) -> float:
    text = value.strip().lower()
    if text.endswith("us"):
        return float(text[:-2]) * 1e-6
    if text.endswith("ms"):
        return float(text[:-2]) * 1e-3
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def _parse_grid(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real/model-metadata KV transfer crossover")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-grid", default="1024,4096,8192,16384,32768")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--bandwidths", default="", help="Comma list like 10Gbps,25Gbps or name:100Gbps")
    parser.add_argument("--latencies", default="", help="Comma list like 10us,100us or rdma:10us")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--real-file-transfer", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    rows = analyze_crossover(
        model_id=args.model_id,
        context_grid=_parse_grid(args.context_grid),
        output_tokens=args.output_tokens,
        bandwidths=parse_bandwidths(args.bandwidths),
        latencies=parse_latencies(args.latencies),
        metadata_only=args.metadata_only,
        device=args.device,
        local_files_only=args.hf_local_files_only,
        run_file_transfer=args.real_file_transfer,
    )
    write_outputs(rows, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
