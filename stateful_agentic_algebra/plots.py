"""Publication-ready plots for Stateful Agentic Algebra experiments.

Paper mapping:
  - TTFT, total latency, transfer/recompute crossover, memory footprint,
    throughput, framework overhead Omega, and KV reuse ratio are plotted from
    `experiment_runner.py` result rows.
  - Real LLM figures consume `multi_llm_runner.py`, `vllm_benchmark.py`,
    `transfer_crossover_real.py`, or `consistency_benchmark.py` CSV rows and
    plot model-grouped TTFT, speedup, KV memory, vLLM serving, TPOT/ITL, and
    consistency metrics.
  - Each chart is a separate matplotlib figure and is saved as SVG, PNG, and
    PDF for both synthetic and real LLM plots.

This module uses matplotlib only. It does not require seaborn or pandas.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence


PLOT_SPECS = [
    ("ttft_vs_context_length", "TTFT vs Context Length"),
    ("total_latency_vs_agents", "Total Latency vs Number of Agents"),
    ("transfer_recompute_crossover", "Transfer vs Recompute Crossover"),
    ("kv_memory_vs_branch_factor", "KV Memory Footprint vs Branch Factor"),
    ("throughput_vs_num_requests", "Throughput vs Number of Requests"),
    ("omega_by_baseline", "Framework Overhead Omega by Baseline"),
    ("kv_reuse_ratio_by_workload", "KV Reuse Ratio by Workload"),
]

REAL_LLM_PLOT_SPECS = [
    ("real_ttft_vs_context_by_baseline", "TTFT vs Context Length by Baseline"),
    ("real_ttft_vs_context_by_model", "TTFT vs Context Length by Model"),
    ("real_ttft_speedup_over_dense", "TTFT Speedup over Dense Prefill"),
    ("real_total_latency_vs_agents_by_baseline", "Total Latency vs Number of Agents by Baseline"),
    ("real_speedup_vs_agents", "Speedup vs Number of Agents"),
    ("real_transfer_recompute_crossover", "Transfer/Recompute Crossover by Model and Bandwidth"),
    ("real_kv_memory_vs_context", "KV Memory Footprint vs Context Length"),
    ("real_kv_memory_vs_branch", "KV Memory Footprint vs Branch Factor"),
    ("real_throughput_by_baseline", "Throughput by Baseline"),
    ("real_omega_by_baseline", "Framework Overhead Omega by Baseline"),
    ("real_vllm_throughput_vs_request_rate", "vLLM Throughput vs Request Rate"),
    ("real_tpot_vs_context", "TPOT vs Context Length"),
    ("real_itl_vs_context", "ITL vs Context Length"),
    ("real_consistency_exact_match_by_model", "Consistency Exact-Match Rate by Model"),
]


def write_metrics_csv(path: str | Path, rows: Iterable[Dict[str, float]]) -> None:
    """Write metric dictionaries to CSV."""

    row_list = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in row_list for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load experiment result rows from CSV."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def generate_all_plots(results: str | Path | Sequence[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Generate all requested figures and return saved paths."""

    rows = load_results(results) if isinstance(results, (str, Path)) else list(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt = _matplotlib()
    saved: list[Path] = []
    plotters: list[tuple[str, Callable[[Any, list[dict[str, Any]], Path], list[Path]]]] = [
        ("ttft_vs_context_length", _plot_ttft_vs_context),
        ("total_latency_vs_agents", _plot_latency_vs_agents),
        ("transfer_recompute_crossover", _plot_transfer_recompute),
        ("kv_memory_vs_branch_factor", _plot_memory_vs_branch),
        ("throughput_vs_num_requests", _plot_throughput_vs_requests),
        ("omega_by_baseline", _plot_omega_by_baseline),
        ("kv_reuse_ratio_by_workload", _plot_reuse_by_workload),
    ]
    for name, plotter in plotters:
        saved.extend(plotter(plt, rows, out / name))
    return saved


def generate_real_llm_plots(
    results: str | Path | Sequence[dict[str, Any]],
    output_dir: str | Path,
    plot_names: Sequence[str] | None = None,
) -> list[Path]:
    """Generate real-model benchmark figures and return saved paths.

    The input may be a `multi_llm_runner.py` results CSV, a
    `transfer_crossover_real.py` crossover CSV, or a `consistency_benchmark.py`
    consistency CSV. Missing columns produce a clearly labeled empty figure
    instead of an exception so partial sweeps remain plottable.
    """

    rows = load_results(results) if isinstance(results, (str, Path)) else list(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt = _matplotlib()
    saved: list[Path] = []
    plotters: list[tuple[str, Callable[[Any, list[dict[str, Any]], Path], list[Path]]]] = [
        ("real_ttft_vs_context_by_baseline", _plot_real_ttft_vs_context_by_baseline),
        ("real_ttft_vs_context_by_model", _plot_real_ttft_vs_context_by_model),
        ("real_ttft_speedup_over_dense", _plot_real_ttft_speedup_over_dense),
        ("real_total_latency_vs_agents_by_baseline", _plot_real_total_latency_vs_agents_by_baseline),
        ("real_speedup_vs_agents", _plot_real_speedup_vs_agents),
        ("real_transfer_recompute_crossover", _plot_real_transfer_recompute_crossover),
        ("real_kv_memory_vs_context", _plot_real_kv_memory_vs_context),
        ("real_kv_memory_vs_branch", _plot_real_kv_memory_vs_branch),
        ("real_throughput_by_baseline", _plot_real_throughput_by_baseline),
        ("real_omega_by_baseline", _plot_real_omega_by_baseline),
        ("real_vllm_throughput_vs_request_rate", _plot_real_vllm_throughput_vs_request_rate),
        ("real_tpot_vs_context", _plot_real_tpot_vs_context),
        ("real_itl_vs_context", _plot_real_itl_vs_context),
        ("real_consistency_exact_match_by_model", _plot_real_consistency_exact_match_by_model),
    ]
    wanted = set(plot_names or [])
    for name, plotter in plotters:
        if wanted and name not in wanted:
            continue
        saved.extend(plotter(plt, rows, out / name))
    return saved


def plot_metric_bar(path: str | Path, metrics: Dict[str, float], title: str = "Stateful Metrics") -> bool:
    """Compatibility helper: plot a simple metric bar chart."""

    try:
        plt = _matplotlib()
    except Exception:
        return False

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics)
    values = [metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(keys, values, color="#4477AA")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return True


def _plot_ttft_vs_context(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="context_tokens", y_key="ttft_sec", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Context Length (tokens)", "TTFT (s)", "TTFT vs Context Length")
    return _save_all(plt, fig, stem)


def _plot_latency_vs_agents(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="num_agents", y_key="total_latency_sec", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Number of Agents", "Total Latency (s)", "Total Latency vs Number of Agents")
    return _save_all(plt, fig, stem)


def _plot_transfer_recompute(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    crossover = [
        row for row in rows if row.get("workload_name") == "transfer_recompute_crossover"
    ] or rows
    grouped_transfer = _group_xy(crossover, x_key="context_tokens", y_key="transfer_sec", series_key="baseline")
    grouped_prefill = _group_xy(crossover, x_key="context_tokens", y_key="prefill_sec", series_key="baseline")
    for label, points in grouped_transfer.items():
        x, y = _sorted_points(points)
        ax.plot(x, y, marker="o", linewidth=2.0, label=f"{label} transfer")
    for label, points in grouped_prefill.items():
        x, y = _sorted_points(points)
        ax.plot(x, y, marker="s", linestyle="--", linewidth=1.8, label=f"{label} recompute")
    _finish_xy(ax, "Context Length (tokens)", "Cost (s)", "Transfer vs Recompute Crossover")
    return _save_all(plt, fig, stem)


def _plot_memory_vs_branch(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="branch_factor", y_key="kv_peak_bytes", series_key="baseline", y_scale=1.0 / (1024 * 1024))
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Branch Factor", "Peak KV Memory (MiB)", "KV Memory Footprint vs Branch Factor")
    return _save_all(plt, fig, stem)


def _plot_throughput_vs_requests(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="num_requests", y_key="throughput_tokens_per_sec", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Number of Requests", "Throughput (tokens/s)", "Throughput vs Number of Requests")
    return _save_all(plt, fig, stem)


def _plot_omega_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt, width=8.5, height=4.8)
    grouped = _group_values(rows, group_key="baseline", value_key="omega_sec")
    labels, values = _bar_values(grouped)
    ax.bar(labels, values, color=_palette(len(labels)))
    ax.set_ylabel("Framework Overhead Omega (s)")
    ax.set_title("Framework Overhead Omega by Baseline")
    _style_axis(ax)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_all(plt, fig, stem)


def _plot_reuse_by_workload(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt, width=9.0, height=4.8)
    grouped = _group_values(rows, group_key="workload_name", value_key="kv_reuse_ratio")
    labels, values = _bar_values(grouped)
    ax.bar(labels, values, color=_palette(len(labels)))
    ax.set_ylabel("KV Reuse Ratio")
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    ax.set_title("KV Reuse Ratio by Workload")
    _style_axis(ax)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_all(plt, fig, stem)


def _plot_real_ttft_vs_context_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    source = [
        row
        for row in rows
        if _row_usable(row)
        and _number(row.get("ttft_sec")) is not None
        and _baseline_label(row)
    ]
    model_ids = {str(row.get("model_id") or "unknown") for row in source}
    normalized = []
    for row in source:
        copied = dict(row)
        baseline = _baseline_label(row) or "unknown"
        if len(model_ids) > 1:
            copied["_baseline_label"] = f"{_short_label(str(row.get('model_id') or 'unknown'))} / {baseline}"
        else:
            copied["_baseline_label"] = baseline
        normalized.append(copied)
    grouped = _group_xy(normalized, x_key="context_tokens", y_key="ttft_sec", series_key="_baseline_label")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "TTFT (s)", "TTFT vs Context Length by Baseline")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_ttft_vs_context_by_model(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    source = _real_primary_rows(rows)
    grouped = _group_xy(source, x_key="context_tokens", y_key="ttft_sec", series_key="model_id")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "TTFT (s)", "TTFT vs Context Length by Model")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_ttft_speedup_over_dense(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _real_speedup_points(rows)
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "Speedup (dense/stateful)", "TTFT Speedup over Dense Prefill")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_total_latency_vs_agents_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    source = [row for row in _real_baseline_rows(rows) if _number(row.get("total_latency_sec")) is not None]
    grouped = _group_xy(source, x_key="num_agents", y_key="total_latency_sec", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Number of Agents", "Total Latency (s)", "Total Latency vs Number of Agents")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_speedup_vs_agents(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _real_speedup_by_x(rows, x_key="num_agents", metric_key="total_latency_sec")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Number of Agents", "Speedup (baseline / AAFLOW+)", "Speedup vs Number of Agents")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_transfer_recompute_crossover(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    if any(row.get("bandwidth_name") for row in rows):
        transfer = _group_xy_with_label(
            rows,
            x_key="context_tokens",
            y_key="t_transfer_sec",
            label_keys=("model_id", "bandwidth_name"),
            separator=" / ",
        )
        recompute = _group_xy(rows, x_key="context_tokens", y_key="t_recompute_sec", series_key="model_id")
    else:
        baseline_rows = _real_baseline_rows(rows)
        transfer = _group_xy(baseline_rows, x_key="context_tokens", y_key="transfer_sec", series_key="baseline")
        recompute = _group_xy(baseline_rows, x_key="context_tokens", y_key="dense_prefill_sec", series_key="baseline")

    colors = _palette(max(1, len(transfer)))
    for idx, (label, points) in enumerate(transfer.items()):
        x, y = _sorted_points(points)
        ax.plot(x, y, marker="o", linewidth=1.8, color=colors[idx % len(colors)], label=f"{label} transfer")
    for label, points in recompute.items():
        x, y = _sorted_points(points)
        ax.plot(x, y, marker="s", linestyle="--", linewidth=1.8, label=f"{label} recompute")
    grouped = {**transfer, **{f"{key} recompute": value for key, value in recompute.items()}}
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "Time (s)", "Transfer/Recompute Crossover by Model and Bandwidth")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_kv_memory_vs_context(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    normalized = []
    for row in _real_baseline_rows(rows) or _real_primary_rows(rows) or rows:
        memory = _effective_kv_memory_bytes(row)
        context = _number(row.get("context_tokens"))
        if memory is None or context is None:
            continue
        copied = dict(row)
        copied["_kv_memory_mib"] = memory / (1024 * 1024)
        normalized.append(copied)
    grouped = _group_xy(normalized, x_key="context_tokens", y_key="_kv_memory_mib", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "KV Memory (MiB)", "KV Memory Footprint vs Context Length")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_kv_memory_vs_branch(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    normalized = []
    for row in _real_baseline_rows(rows) or rows:
        memory = _effective_kv_memory_bytes(row)
        branch = _number(row.get("branch_factor"))
        if memory is None or branch is None:
            continue
        copied = dict(row)
        copied["_kv_memory_gib"] = memory / (1024**3)
        normalized.append(copied)
    grouped = _group_xy(normalized, x_key="branch_factor", y_key="_kv_memory_gib", series_key="baseline")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Branch Factor", "Peak KV Memory (GiB)", "KV Memory Footprint vs Branch Factor")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_throughput_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_values(_real_baseline_rows(rows), group_key="baseline", value_key="throughput_tokens_per_sec")
    labels, values = _bar_values(grouped)
    if labels:
        ax.bar(labels, values, color=_palette(len(labels)))
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput by Baseline")
    _style_axis(ax)
    if not labels:
        _annotate_no_data(ax)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_png_pdf(plt, fig, stem)


def _plot_real_omega_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_values(_real_baseline_rows(rows), group_key="baseline", value_key="omega_sec")
    labels, values = _bar_values(grouped)
    if labels:
        ax.bar(labels, values, color=_palette(len(labels)))
    ax.set_ylabel("Framework Overhead Omega (s)")
    ax.set_title("Framework Overhead Omega by Baseline")
    _style_axis(ax)
    if not labels:
        _annotate_no_data(ax)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_png_pdf(plt, fig, stem)


def _plot_real_vllm_throughput_vs_request_rate(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    normalized = []
    for row in rows:
        if str(row.get("backend", "")).lower() != "vllm" and row.get("workload_name") != "vllm_serve":
            continue
        x_value = _first_number(row, ("request_rate_configured_rps", "traffic_request_rate", "request_rate", "num_prompts"))
        y_value = _first_number(row, ("request_throughput_req_per_sec", "throughput_tokens_per_sec"))
        if x_value is None or y_value is None:
            continue
        copied = dict(row)
        copied["_request_rate"] = x_value
        copied["_throughput"] = y_value
        normalized.append(copied)
    grouped = _group_xy(normalized, x_key="_request_rate", y_key="_throughput", series_key="model_id")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Configured Request Rate or Prompt Count", "Throughput", "vLLM Throughput vs Request Rate")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_tpot_vs_context(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    source = [row for row in rows if _number(row.get("tpot_sec")) is not None]
    grouped = _group_xy(source, x_key="context_tokens", y_key="tpot_sec", series_key="model_id")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "TPOT (s/token)", "TPOT vs Context Length")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_itl_vs_context(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    source = [row for row in rows if _number(row.get("itl_sec")) is not None]
    grouped = _group_xy(source, x_key="context_tokens", y_key="itl_sec", series_key="model_id")
    _plot_lines(ax, grouped)
    _finish_real_xy(ax, grouped, "Context Length (tokens)", "ITL (s/token)", "ITL vs Context Length")
    return _save_png_pdf(plt, fig, stem)


def _plot_real_consistency_exact_match_by_model(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt, width=8.5, height=4.8)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _consistency_value(row)
        if value is None:
            continue
        grouped[str(row.get("model_id") or "unknown")].append(value)
    labels, values = _bar_values(dict(sorted(grouped.items())))
    if labels:
        ax.bar(labels, values, color=_palette(len(labels)))
    ax.set_ylabel("Exact-Match Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Consistency Exact-Match Rate by Model")
    _style_axis(ax)
    if not labels:
        _annotate_no_data(ax)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_png_pdf(plt, fig, stem)


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _figure(plt: Any, width: float = 7.2, height: float = 4.6) -> tuple[Any, Any]:
    return plt.subplots(figsize=(width, height))


def _finish_xy(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")
    ax.figure.tight_layout()


def _finish_real_xy(
    ax: Any,
    grouped: dict[str, list[tuple[float, float]]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")
    if not grouped:
        _annotate_no_data(ax)
    ax.figure.tight_layout()


def _style_axis(ax: Any) -> None:
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.8)
    ax.grid(True, axis="x", color="#EEF1F4", linewidth=0.5)
    ax.set_axisbelow(True)


def _plot_lines(ax: Any, grouped: dict[str, list[tuple[float, float]]]) -> None:
    colors = _palette(len(grouped))
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    offsets = _series_x_offsets(grouped)
    for idx, (label, points) in enumerate(grouped.items()):
        x, y = _sorted_points(points)
        offset = offsets.get(label, 0.0)
        x_plot = [value + offset for value in x]
        ax.plot(
            x_plot,
            y,
            marker=markers[idx % len(markers)],
            markersize=5.5,
            linewidth=2.0,
            linestyle=linestyles[idx % len(linestyles)],
            color=colors[idx],
            label=label,
            alpha=0.95,
        )


def _annotate_no_data(ax: Any) -> None:
    ax.text(
        0.5,
        0.5,
        "No matching data in results CSV",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#667085",
    )


def _group_xy(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    y_scale: float = 1.0,
) -> dict[str, list[tuple[float, float]]]:
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        x = _number(row.get(x_key))
        y = _number(row.get(y_key))
        if x is None or y is None:
            continue
        label = _series_label(row, series_key)
        if not label:
            continue
        buckets[(label, x)].append(y * y_scale)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (label, x), values in buckets.items():
        grouped[label].append((x, sum(values) / len(values)))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _group_xy_with_label(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    label_keys: tuple[str, ...],
    separator: str = " ",
    y_scale: float = 1.0,
) -> dict[str, list[tuple[float, float]]]:
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        x = _number(row.get(x_key))
        y = _number(row.get(y_key))
        if x is None or y is None:
            continue
        labels = [str(row.get(key) or "unknown") for key in label_keys]
        label = separator.join(labels)
        buckets[(label, x)].append(y * y_scale)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (label, x), values in buckets.items():
        grouped[label].append((x, sum(values) / len(values)))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _group_values(rows: list[dict[str, Any]], *, group_key: str, value_key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(value_key))
        if value is None:
            continue
        label = _series_label(row, group_key)
        if not label:
            continue
        grouped[label].append(value)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _bar_values(grouped: dict[str, list[float]]) -> tuple[list[str], list[float]]:
    labels = list(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) if grouped[label] else 0.0 for label in labels]
    return labels, values


def _sorted_points(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    ordered = sorted(points, key=lambda item: item[0])
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _series_x_offsets(grouped: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    """Return tiny visual x-offsets so coincident baseline curves remain visible."""

    if len(grouped) <= 1:
        return {}
    all_x = sorted({x for points in grouped.values() for x, _y in points})
    if len(all_x) > 1:
        min_gap = min(b - a for a, b in zip(all_x, all_x[1:]) if b > a)
        step = min_gap * 0.008
    else:
        step = max(abs(all_x[0]) * 0.01, 0.02) if all_x else 0.0
    if step == 0.0:
        return {}

    labels = list(grouped)
    center = (len(labels) - 1) / 2.0
    offsets = {label: (idx - center) * step for idx, label in enumerate(labels)}
    return offsets


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _first_number(row: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"", "0", "false", "no", "n"}


def _row_usable(row: dict[str, Any]) -> bool:
    available = row.get("available")
    skipped = row.get("skipped")
    return (available is None or _truthy(available)) and (skipped is None or _falsey(skipped))


def _real_primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _row_usable(row)
        and _number(row.get("ttft_sec")) is not None
        and (
            row.get("workload_name") in {"AAFLOW+", "aaflow_plus", "vllm_serve"}
            or str(row.get("backend", "")).lower() == "vllm"
            or not row.get("workload_name")
        )
    ]
    return primary or [row for row in rows if _row_usable(row)]


def _real_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _row_usable(row)
        and _baseline_label(row)
        and _number(row.get("context_tokens")) is not None
    ]


BASELINE_WORKLOAD_NAMES = {
    "AAFLOW+",
    "aaflow_plus",
    "dense_prefill",
    "aaflow_text",
    "vllm_local_prefix",
    "sglang_prefix",
    "kvcomm_prefix",
    "distserve_style",
}


def _baseline_label(row: dict[str, Any]) -> str:
    baseline = str(row.get("baseline_name") or "").strip()
    if baseline:
        return baseline
    workload = str(row.get("workload_name") or "").strip()
    if workload in BASELINE_WORKLOAD_NAMES:
        return workload
    return ""


def _effective_kv_memory_bytes(row: dict[str, Any]) -> float | None:
    """Estimate peak KV footprint consistently for plotted baseline rows.

    Older result files stored the measured prefix KV size in `kv_peak_bytes` for
    every baseline. For memory-efficiency figures we need the logical peak
    footprint under each baseline's sharing model, so derive it from KV size,
    agent count, and branch factor when a baseline label is present.
    """

    kv_bytes = _first_number(row, ("kv_total_bytes", "kv_peak_bytes", "kv_bytes"))
    if kv_bytes is None or kv_bytes <= 0:
        return None
    agents = max(1, int(_number(row.get("num_agents")) or 1))
    branch = max(1, int(_number(row.get("branch_factor")) or 1))
    branch_instances = max(1, int(_number(row.get("branch_instances")) or (agents * branch)))
    suffix_fraction = 0.15
    baseline = _baseline_label(row)
    if baseline == "AAFLOW+":
        return kv_bytes * (1.0 + max(0, branch_instances - 1) * suffix_fraction)
    if baseline in {"dense_prefill", "aaflow_text", "distserve_style"}:
        return kv_bytes * branch_instances
    if baseline in {"vllm_local_prefix", "sglang_prefix", "kvcomm_prefix"}:
        return kv_bytes * agents * (1.0 + max(0, branch - 1) * suffix_fraction)
    return kv_bytes


def _series_label(row: dict[str, Any], series_key: str) -> str:
    if series_key == "baseline":
        return _baseline_label(row)
    return str(row.get(series_key) or "").strip()


def _real_speedup_points(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    dense_by_key: dict[tuple[str, str, float, float, float, float], dict[str, Any]] = {}
    stateful_rows: list[dict[str, Any]] = []
    for row in rows:
        if not _row_usable(row):
            continue
        context = _number(row.get("context_tokens"))
        if context is None:
            continue
        key = (
            str(row.get("model_id") or "unknown"),
            str(row.get("backend") or "hf"),
            context,
            _number(row.get("output_tokens")) or 0.0,
            _number(row.get("num_agents")) or 0.0,
            _number(row.get("branch_factor")) or 0.0,
        )
        workload = str(row.get("workload_name") or "")
        if workload == "dense_prefill":
            dense_by_key[key] = row
        elif workload in {"AAFLOW+", "aaflow_plus", ""}:
            stateful_rows.append(row)

    for row in stateful_rows:
        context = _number(row.get("context_tokens"))
        if context is None:
            continue
        key = (
            str(row.get("model_id") or "unknown"),
            str(row.get("backend") or "hf"),
            context,
            _number(row.get("output_tokens")) or 0.0,
            _number(row.get("num_agents")) or 0.0,
            _number(row.get("branch_factor")) or 0.0,
        )
        dense_row = dense_by_key.get(key, {})
        dense = _first_number(dense_row, ("dense_prefill_sec", "prefill_sec", "ttft_sec"))
        stateful = _first_number(row, ("stateful_prefill_sec", "prefill_sec", "ttft_sec"))
        if dense is None:
            dense = _first_number(row, ("dense_prefill_sec",))
        if stateful is None:
            stateful = _first_number(row, ("stateful_prefill_sec",))
        if dense is None or stateful is None or stateful <= 0:
            continue
        grouped[str(row.get("model_id") or "unknown")].append((context, dense / stateful))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _real_speedup_by_x(rows: list[dict[str, Any]], *, x_key: str, metric_key: str) -> dict[str, list[tuple[float, float]]]:
    refs: dict[tuple[str, str, float, float, float, float], float] = {}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not _row_usable(row):
            continue
        metric = _number(row.get(metric_key))
        x_value = _number(row.get(x_key))
        if metric is None or metric <= 0 or x_value is None:
            continue
        key = (
            str(row.get("model_id") or "unknown"),
            _backend_family(str(row.get("backend") or "")),
            _number(row.get("context_tokens")) or 0.0,
            _number(row.get("output_tokens")) or 0.0,
            _number(row.get("branch_factor")) or 0.0,
            x_value,
        )
        if _baseline_label(row) == "AAFLOW+":
            refs[key] = metric
        candidates.append(row)

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in candidates:
        x_value = _number(row.get(x_key))
        metric = _number(row.get(metric_key))
        if x_value is None or metric is None:
            continue
        key = (
            str(row.get("model_id") or "unknown"),
            _backend_family(str(row.get("backend") or "")),
            _number(row.get("context_tokens")) or 0.0,
            _number(row.get("output_tokens")) or 0.0,
            _number(row.get("branch_factor")) or 0.0,
            x_value,
        )
        ref = refs.get(key)
        label = _baseline_label(row)
        if ref is None or ref <= 0 or not label:
            continue
        grouped[label].append((x_value, metric / ref))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _backend_family(backend: str) -> str:
    text = backend.lower()
    if text.startswith("hf"):
        return "hf"
    if text.startswith("vllm"):
        return "vllm"
    if text.startswith("sglang"):
        return "sglang"
    return text


def _consistency_value(row: dict[str, Any]) -> float | None:
    for key in ("exact_match_rate", "mean_exact_token_match_rate", "exact_token_match_rate", "output_agreement_rate"):
        value = _number(row.get(key))
        if value is not None:
            return max(0.0, min(1.0, value))
    exact = row.get("exact_match")
    if exact in (None, ""):
        return None
    return 1.0 if _truthy(exact) else 0.0


def _palette(n: int) -> list[str]:
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]
    if n <= len(colors):
        return colors[:n]
    return [colors[idx % len(colors)] for idx in range(n)]


def _short_label(value: str) -> str:
    return value.rsplit("/", 1)[-1] if "/" in value else value


def _save_all(plt: Any, fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in ("svg", "png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def _save_png_pdf(plt: Any, fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stateful Agentic Algebra plots")
    parser.add_argument("--results", required=True, help="Path to experiment results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures")
    parser.add_argument("--real-llm", action="store_true", help="Generate real LLM benchmark figures")
    parser.add_argument(
        "--plot-names",
        default="",
        help="Comma-separated real-LLM plot function names to generate; defaults to all plots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot_names = [item.strip() for item in args.plot_names.split(",") if item.strip()] or None
    saved = (
        generate_real_llm_plots(args.results, args.output_dir, plot_names=plot_names)
        if args.real_llm
        else generate_all_plots(args.results, args.output_dir)
    )
    print(f"wrote {len(saved)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
