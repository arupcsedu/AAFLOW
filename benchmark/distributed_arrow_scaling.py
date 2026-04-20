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
from benchmark_configs_1_to_5 import (
    FakeEmbedder,
    LocalHashEmbedder,
    ResultRow,
    print_table,
    run_set7_dask_data,
    run_set9_higress_data,
)
from distributed_agentic_scaling import (
    _aggregate_rows,
    _barrier,
    _ensure_cached_corpus,
    _make_sink,
    _make_rank_shard_dir,
    _resolve_tuned_config_rank_safe,
    _run_local_async_parallel_only,
    _run_local_ingest,
    _run_local_ingest_streaming,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Arrow scaling runner across Slurm tasks.")
    p.add_argument("--profile", choices=["strong", "weak"], required=True)
    p.add_argument("--physical-workers", type=int, required=True)
    p.add_argument("--stage-cpu-cap", type=int, default=1)
    p.add_argument("--base-nodes", type=int, default=65536)
    p.add_argument("--base-files", type=int, default=512)
    p.add_argument("--base-nodes-per-worker", type=int, default=128)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunks-per-file", type=int, default=0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--embedder-backend", choices=["local-hash", "fake"], default="local-hash")
    p.add_argument("--sink-backend", choices=["chroma", "thin-batched", "faiss"], default="faiss")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--upsert-batch-size", type=int, default=512)
    p.add_argument("--request-overhead-ms", type=float, default=20.0)
    p.add_argument("--per-item-ms", type=float, default=0.5)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--shared-corpus-root", default="/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache")
    p.add_argument("--persist-dir", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-arrow-workers", type=int, default=1)
    return p.parse_args()


async def _run_empty(config: str) -> ResultRow:
    return ResultRow(config, 0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tuning_args = argparse.Namespace(
        profile=args.profile,
        async_workers=args.physical_workers,
        physical_workers=args.physical_workers,
        stage_cpu_cap=max(1, args.stage_cpu_cap),
        base_nodes=args.base_nodes,
        base_files=args.base_files,
        base_nodes_per_worker=args.base_nodes_per_worker,
        node_chars=args.node_chars,
        chunks_per_file=max(1, args.chunks_per_file),
        dim=args.dim,
        embedder_backend=args.embedder_backend,
        set5_sink_backend=args.sink_backend,
        persist_dir=args.persist_dir,
        run_dir=str(run_dir),
        python_bin="",
        bench_script="",
        mode="agentic",
    )
    tuned, config_path = _resolve_tuned_config_rank_safe(tuning_args, run_dir, rank, world)

    total_nodes = int(tuned["nodes"])
    min_files_for_chunks = max(1, int(math.ceil(total_nodes / max(1, args.chunks_per_file)))) if args.chunks_per_file > 0 else 1
    total_files = max(int(tuned["files"]), int(args.physical_workers), min_files_for_chunks)
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
                    "worker_config_path": str(config_path),
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
        embedder = LocalHashEmbedder(args.dim)
    else:
        embedder = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)

    config_rows: List[ResultRow] = []
    if local_files:
        config_rows.append(
            asyncio.run(
                _run_local_async_parallel_only(
                    data_dir=str(shard_dir),
                    embedder=embedder,
                    sink_backend=args.sink_backend,
                    persist_dir=persist_dir,
                    dim=args.dim,
                    rank=rank,
                )
            )
        )
        config_rows.append(
            run_set7_dask_data(
                data_dir=str(shard_dir),
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=max(1, int(tuned["embed_batch"])),
                upsert_batch_size=max(1, int(tuned["upsert_batch"])),
                dask_workers=1,
                upsert_workers=1,
                set45_upsert_shards=1,
            )
        )
        config_rows.append(
            asyncio.run(
                run_set9_higress_data(
                    data_dir=str(shard_dir),
                    persist_dir=persist_dir,
                    embedder=embedder,
                    sink_backend=args.sink_backend,
                    dim=args.dim,
                    embed_batch_size=max(1, int(tuned["embed_batch"])),
                    upsert_batch_size=max(1, int(tuned["upsert_batch"])),
                    higress_workers=1,
                    upsert_workers=1,
                    set45_upsert_shards=1,
                )
            )
        )
        if args.profile == "weak" and args.sink_backend == "faiss" and args.physical_workers <= 256:
            config_rows.append(
                asyncio.run(
                    _run_local_ingest_streaming(
                        data_dir=str(shard_dir),
                        embedder=embedder,
                        sink=_make_sink(args.sink_backend, f"dist_arrow_agentic_rank_{rank}", args.dim, persist_dir),
                        embed_batch_size=max(1, args.embed_batch_size),
                        upsert_batch_size=max(1, args.upsert_batch_size),
                        load_workers=4,
                        transform_workers=4,
                        embed_num_workers=4,
                        upsert_num_workers=2,
                        upsert_coalesce_multiplier=2,
                    )
                )
            )
        else:
            config_rows.append(
                asyncio.run(
                    _run_local_ingest(
                        data_dir=str(shard_dir),
                        embedder=embedder,
                        sink=_make_sink(args.sink_backend, f"dist_arrow_agentic_rank_{rank}", args.dim, persist_dir),
                        embed_batch_size=max(1, args.embed_batch_size),
                        upsert_batch_size=max(1, args.upsert_batch_size),
                    )
                )
            )
        config_rows.append(
            asyncio.run(
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
        )
    else:
        for name in ("AsyncParallelOnly", "DaskScalableRAG", "HigressRAG", "AAFLOW", "AAFLOW+"):
            config_rows.append(asyncio.run(_run_empty(name)))

    results_dir = run_dir / "rank_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"rank_{rank:06d}.json").write_text(
        json.dumps([row.__dict__ for row in config_rows]),
        encoding="utf-8",
    )
    _barrier(run_dir, "results_ready", rank, world)

    if rank == 0:
        rows_by_config: Dict[str, List[Dict[str, Any]]] = {}
        for path in sorted(results_dir.glob("rank_*.json")):
            for row_data in json.loads(path.read_text(encoding="utf-8")):
                rows_by_config.setdefault(row_data["config"], []).append(row_data)
        order = ["AsyncParallelOnly", "DaskScalableRAG", "HigressRAG", "AAFLOW", "AAFLOW+"]
        agg_rows = [_aggregate_rows(rows_by_config.get(name, []), name) for name in order if name in rows_by_config]
        print_table(agg_rows, baseline_label="AsyncParallelOnly")
        with (run_dir / "summary_no_ray.csv").open("w", newline="", encoding="utf-8") as f:
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
            for row in agg_rows:
                writer.writerow([
                    row.config,
                    row.nodes,
                    row.load_s,
                    row.transform_s,
                    row.embed_s,
                    row.upsert_s,
                    row.total_s,
                    args.physical_workers,
                    world,
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
