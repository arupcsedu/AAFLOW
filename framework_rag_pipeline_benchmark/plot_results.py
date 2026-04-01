#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STAGE_COLUMNS = ["load_s", "transform_s", "embed_s", "upsert_s", "total_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot framework RAG pipeline benchmark outputs")
    p.add_argument("--summary-csv", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "summary.csv"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "plots"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_df = df[["framework", *STAGE_COLUMNS]].set_index("framework")
    ax = stage_df.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Seconds")
    ax.set_title("RAG Pipeline Stage Times by Framework")
    ax.legend(title="Stage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    stage_path = output_dir / "stage_times.png"
    plt.savefig(stage_path, dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df["framework"], df["tokens_per_second"], color="#4C78A8")
    ax1.set_ylabel("Tokens / second")
    ax1.set_title("Generation Throughput by Framework")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    throughput_path = output_dir / "throughput.png"
    plt.savefig(throughput_path, dpi=200)
    plt.close()

    print(f"Wrote {stage_path}")
    print(f"Wrote {throughput_path}")


if __name__ == "__main__":
    main()
