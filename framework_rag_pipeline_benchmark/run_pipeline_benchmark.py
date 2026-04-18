#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .common import BenchmarkConfig, ensure_synthetic_corpus, median_metrics, write_metrics
from .runners import RUNNERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RAG pipeline stages across framework orchestrators")
    p.add_argument("--benchmark-mode", choices=["default", "fair_parallelism", "fair_parallelism_plus_overlap"], default="default")
    p.add_argument("--embedding-backend", choices=["fake", "transformers"], default="fake")
    p.add_argument("--generation-backend", choices=["fake", "transformers"], default="fake")
    p.add_argument("--vector-backend", choices=["fake", "chroma", "faiss"], default="fake")
    p.add_argument("--generation-cost-mode", choices=["linear", "fixed"], default="linear")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--generation-model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--chroma-path", type=str, default=None)
    p.add_argument("--faiss-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "synthetic_data"))
    p.add_argument("--file-glob", type=str, default="*.txt")
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--files", type=int, default=64)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunk-tokens", type=int, default=180)
    p.add_argument("--chunk-overlap", type=int, default=20)
    p.add_argument("--generation-samples", type=int, default=128)
    p.add_argument("--generation-output-tokens", type=int, default=64)
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--transform-workers", type=int, default=8)
    p.add_argument("--async-workers", type=int, default=16)
    p.add_argument("--physical-workers", type=int, default=None)
    p.add_argument("--embed-workers", type=int, default=None)
    p.add_argument("--upsert-workers", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--upsert-batch-size", type=int, default=64)
    p.add_argument("--agentic-queue-size", type=int, default=0)
    p.add_argument("--agentic-upsert-coalesce-target", type=int, default=0)
    p.add_argument("--framework-filter", type=str, default="")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--embed-overhead-ms", type=float, default=20.0)
    p.add_argument("--embed-per-item-ms", type=float, default=1.5)
    p.add_argument("--upsert-overhead-ms", type=float, default=8.0)
    p.add_argument("--upsert-per-item-ms", type=float, default=0.5)
    p.add_argument("--generate-overhead-ms", type=float, default=30.0)
    p.add_argument("--generate-ms-per-token", type=float, default=0.6)
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    physical_workers = args.physical_workers if args.physical_workers is not None else args.async_workers
    load_workers = min(max(1, args.load_workers), max(1, physical_workers))
    transform_workers = min(max(1, args.transform_workers), max(1, physical_workers))
    async_workers = min(max(1, args.async_workers), max(1, physical_workers))
    embed_workers = None if args.embed_workers is None else min(max(1, args.embed_workers), max(1, physical_workers))
    upsert_workers = None if args.upsert_workers is None else min(max(1, args.upsert_workers), max(1, physical_workers))
    config = BenchmarkConfig(
        benchmark_mode=args.benchmark_mode,
        embedding_backend=args.embedding_backend,
        generation_backend=args.generation_backend,
        vector_backend=args.vector_backend,
        generation_cost_mode=args.generation_cost_mode,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        chroma_path=args.chroma_path,
        faiss_path=args.faiss_path,
        data_dir=args.data_dir,
        file_glob=args.file_glob,
        nodes=args.nodes,
        files=args.files,
        node_chars=args.node_chars,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        generation_samples=args.generation_samples,
        generation_output_tokens=args.generation_output_tokens,
        load_workers=load_workers,
        transform_workers=transform_workers,
        async_workers=async_workers,
        physical_workers=physical_workers,
        embed_workers=embed_workers,
        upsert_workers=upsert_workers,
        embed_dim=args.embed_dim,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        agentic_queue_size=args.agentic_queue_size,
        agentic_upsert_coalesce_target=args.agentic_upsert_coalesce_target,
        seed=args.seed,
        embed_overhead_ms=args.embed_overhead_ms,
        embed_per_item_ms=args.embed_per_item_ms,
        upsert_overhead_ms=args.upsert_overhead_ms,
        upsert_per_item_ms=args.upsert_per_item_ms,
        generate_overhead_ms=args.generate_overhead_ms,
        generate_ms_per_token=args.generate_ms_per_token,
    )

    ensure_synthetic_corpus(config)

    wanted = {item.strip() for item in args.framework_filter.split(",") if item.strip()}
    selected_runners = [runner_cls for runner_cls in RUNNERS if not wanted or runner_cls.framework in wanted]

    all_rows = []
    grouped = {runner_cls.framework: [] for runner_cls in selected_runners}
    for repeat_index in range(max(1, args.repeat)):
        for runner_cls in selected_runners:
            runner = runner_cls(config)
            row = runner.run()
            grouped[row.framework].append(row)
            full_row = row.__dict__.copy()
            full_row["repeat_index"] = repeat_index
            all_rows.append(full_row)
            print(
                f"[repeat {repeat_index + 1}/{max(1, args.repeat)}] "
                f"{row.framework:10s} mode={row.runtime_mode:8s} docs={row.documents_loaded:4d} chunks={row.chunks:5d} "
                f"load={row.load_s:.3f}s transform={row.transform_s:.3f}s gen={row.generation_s:.3f}s "
                f"tok/s={row.tokens_per_second:.1f} embed={row.embed_s:.3f}s upsert={row.upsert_s:.3f}s total={row.total_s:.3f}s"
            )

    rows = [median_metrics(grouped[runner_cls.framework]) for runner_cls in selected_runners]
    write_metrics(Path(args.output_dir), rows, full_rows=all_rows)
    print(f"Wrote {Path(args.output_dir) / 'summary.csv'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
