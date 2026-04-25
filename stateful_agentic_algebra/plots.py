"""Plot helpers for Stateful Agentic Algebra experiments.

Paper mapping:
  - Metrics visualization: plots/summaries cover TTFT, transfer cost,
    recompute cost, throughput, memory, reuse ratio, and framework overhead
    Omega.
  - Robust imports: matplotlib is optional; CSV output remains available when
    plotting libraries are absent.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


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


def plot_metric_bar(path: str | Path, metrics: Dict[str, float], title: str = "Stateful Metrics") -> bool:
    """Plot a simple bar chart if matplotlib is installed; return success."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics)
    values = [metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(keys, values)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True

