#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from benchmark_arrow_configs import run_arrow_no_ray
from benchmark_configs_1_to_5 import FakeEmbedder, LocalHashEmbedder, ResultRow, print_table
from distributed_agentic_scaling import _aggregate_rows, _barrier, _ensure_cached_corpus, _make_rank_shard_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Arrow AAFLOW no-Ray runner across Slurm tasks.")
    p.add_argument("--profile", choices=["strong", "weak"], required=True)
    p.add_argument("--physical-workers", type=int, required=True)
    p.add_argument("--base-nodes", type=int, default=65536)
    p.add_argument("--base-files", type=int, default=512)
    p.add_argument("--base-nodes-per-worker", type=int, default=128)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunks-per-file", type=int, default=0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--upsert-batch-size", type=int, default=512)
    p.add_argument("--embedder-backend", choices=["local-hash", "fake"], default="local-hash")
    p.add_argument("--sink-backend", choices=["chroma", "thin-batched", "faiss"], default="faiss")
    p.add_argument("--request-overhead-ms", type=float, default=20.0)
    p.add_argument("--per-item-ms", type=float, default=0.5)
    p.add_argument("--data-dir", default="")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--shared-corpus-root", default="/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache")
    p.add_argument("--persist-dir", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-arrow-workers", type=int, default=1)
    return p.parse_args()


async def _run_empty() -> ResultRow:
    return ResultRow("AAFLOW+", 0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    total_nodes = int(args.base_nodes if args.profile == "strong" else args.physical_workers * args.base_nodes_per_worker)
    min_files_for_chunks = max(1, int(math.ceil(total_nodes / max(1, args.chunks_per_file)))) if args.chunks_per_file > 0 else 1
    total_files = max(int(args.base_files), int(args.physical_workers), min_files_for_chunks)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = _ensure_cached_corpus(
            cache_root=Path(args.shared_corpus_root),
            total_nodes=total_nodes,
            node_chars=args.node_chars,
            total_files=total_files,
            seed=args.seed,
            rank=rank,
            world=world,
            run_dir=run_dir,
        )

    if rank == 0:
        (run_dir / "allocation.txt").write_text(
            json.dumps(
                {
                    "profile": args.profile,
                    "physical_workers": args.physical_workers,
                    "world_size": world,
                    "total_nodes": total_nodes,
                    "total_files": total_files,
                    "node_chars": args.node_chars,
                    "chunks_per_file": args.chunks_per_file,
                    "embed_batch_size": args.embed_batch_size,
                    "upsert_batch_size": args.upsert_batch_size,
                    "embedder_backend": args.embedder_backend,
                    "sink_backend": args.sink_backend,
                    "shared_corpus_dir": str(data_dir),
                    "data_dir_override": bool(args.data_dir),
                    "local_arrow_workers": args.local_arrow_workers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _barrier(run_dir, "corpus_ready", rank, world)

    file_paths = sorted(str(p) for p in data_dir.glob("*.txt"))
    local_files = file_paths[rank::world]
    shard_dir = _make_rank_shard_dir(run_dir, rank, local_files)
    persist_dir = args.persist_dir.strip() or None

    if args.embedder_backend == "local-hash":
        _ = LocalHashEmbedder(args.dim)
    else:
        _ = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)

    if local_files:
        row = asyncio.run(
            run_arrow_no_ray(
                data_dir=str(shard_dir),
                persist_dir=persist_dir,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=max(1, args.embed_batch_size),
                upsert_batch_size=max(1, args.upsert_batch_size),
                workers=max(1, args.local_arrow_workers),
                embedder_backend=args.embedder_backend,
                request_overhead_ms=args.request_overhead_ms,
                per_item_ms=args.per_item_ms,
            )
        )
    else:
        row = asyncio.run(_run_empty())

    results_dir = run_dir / "rank_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"rank_{rank:06d}.json").write_text(json.dumps([row.__dict__]), encoding="utf-8")
    _barrier(run_dir, "results_ready", rank, world)

    if rank == 0:
        rows_by_config: Dict[str, List[Dict[str, Any]]] = {}
        for path in sorted(results_dir.glob("rank_*.json")):
            for row_data in json.loads(path.read_text(encoding="utf-8")):
                rows_by_config.setdefault(row_data["config"], []).append(row_data)
        agg_rows = [_aggregate_rows(rows_by_config.get("AAFLOW+", []), "AAFLOW+")]
        print_table(agg_rows, baseline_label="AAFLOW+")
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config",
                "chunks",
                "load_s",
                "transform_s",
                "embed_s",
                "upsert_s",
                "total_s",
                "physical_workers",
                "world_size",
            ])
            for agg in agg_rows:
                writer.writerow([
                    agg.config,
                    agg.nodes,
                    agg.load_s,
                    agg.transform_s,
                    agg.embed_s,
                    agg.upsert_s,
                    agg.total_s,
                    args.physical_workers,
                    world,
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
