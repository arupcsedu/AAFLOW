"""Publication-ready plots for Stateful Agentic Algebra experiments.

Paper mapping:
  - TTFT, total latency, transfer/recompute crossover, memory footprint,
    throughput, framework overhead Omega, and KV reuse ratio are plotted from
    `experiment_runner.py` result rows.
  - Each chart is a separate matplotlib figure and is saved as SVG, PNG, and
    PDF for paper, slide, and web workflows.

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
    grouped = _group_xy(rows, x_key="context_tokens", y_key="ttft_sec", series_key="baseline_name")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Context Length (tokens)", "TTFT (s)", "TTFT vs Context Length")
    return _save_all(plt, fig, stem)


def _plot_latency_vs_agents(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="num_agents", y_key="total_latency_sec", series_key="baseline_name")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Number of Agents", "Total Latency (s)", "Total Latency vs Number of Agents")
    return _save_all(plt, fig, stem)


def _plot_transfer_recompute(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    crossover = [
        row for row in rows if row.get("workload_name") == "transfer_recompute_crossover"
    ] or rows
    grouped_transfer = _group_xy(crossover, x_key="context_tokens", y_key="transfer_sec", series_key="baseline_name")
    grouped_prefill = _group_xy(crossover, x_key="context_tokens", y_key="prefill_sec", series_key="baseline_name")
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
    grouped = _group_xy(rows, x_key="branch_factor", y_key="kv_peak_bytes", series_key="baseline_name", y_scale=1.0 / (1024 * 1024))
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Branch Factor", "Peak KV Memory (MiB)", "KV Memory Footprint vs Branch Factor")
    return _save_all(plt, fig, stem)


def _plot_throughput_vs_requests(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt)
    grouped = _group_xy(rows, x_key="num_requests", y_key="throughput_tokens_per_sec", series_key="baseline_name")
    _plot_lines(ax, grouped)
    _finish_xy(ax, "Number of Requests", "Throughput (tokens/s)", "Throughput vs Number of Requests")
    return _save_all(plt, fig, stem)


def _plot_omega_by_baseline(plt: Any, rows: list[dict[str, Any]], stem: Path) -> list[Path]:
    fig, ax = _figure(plt, width=8.5, height=4.8)
    grouped = _group_values(rows, group_key="baseline_name", value_key="omega_sec")
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


def _style_axis(ax: Any) -> None:
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.8)
    ax.grid(True, axis="x", color="#EEF1F4", linewidth=0.5)
    ax.set_axisbelow(True)


def _plot_lines(ax: Any, grouped: dict[str, list[tuple[float, float]]]) -> None:
    colors = _palette(len(grouped))
    for idx, (label, points) in enumerate(grouped.items()):
        x, y = _sorted_points(points)
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors[idx], label=label)


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
        label = str(row.get(series_key) or "unknown")
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
        grouped[str(row.get(group_key) or "unknown")].append(value)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _bar_values(grouped: dict[str, list[float]]) -> tuple[list[str], list[float]]:
    labels = list(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) if grouped[label] else 0.0 for label in labels]
    return labels, values


def _sorted_points(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    ordered = sorted(points, key=lambda item: item[0])
    return [item[0] for item in ordered], [item[1] for item in ordered]


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


def _palette(n: int) -> list[str]:
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]
    if n <= len(colors):
        return colors[:n]
    return [colors[idx % len(colors)] for idx in range(n)]


def _save_all(plt: Any, fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in ("svg", "png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stateful Agentic Algebra plots")
    parser.add_argument("--results", required=True, help="Path to experiment results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    saved = generate_all_plots(args.results, args.output_dir)
    print(f"wrote {len(saved)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
