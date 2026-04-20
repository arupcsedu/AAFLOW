#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import time
from pathlib import Path
import shutil
from typing import Any, Dict, List

from benchmark_configs_1_to_5 import (
    DELIM,
    LocalHashEmbedder,
    FakeEmbedder,
    FaissSink,
    ThinBatchedSink,
    ResultRow,
    _rand_text,
    _read_text_file,
    _chunk_text_record,
    chroma_upsert_batches,
    embed_and_upsert,
    embed_and_upsert_streaming,
    get_chroma_collection,
    init_sink_collections,
    load_docs_sync,
    print_table,
    run_direct_batched_ingest,
    run_set7_dask_data,
    run_set8_bsp_data,
    run_set9_higress_data,
    transform_async,
    write_synthetic_corpus,
)
from agentic_scaling_runner import resolve_tuned_config


def _barrier(run_dir: Path, name: str, rank: int, world: int, poll_s: float = 0.1) -> None:
    barrier_dir = run_dir / "barriers" / name
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"arrive_{rank:06d}").write_text("1", encoding="utf-8")
    release = barrier_dir / "release"
    if rank == 0:
        while len(list(barrier_dir.glob("arrive_*"))) < world:
            time.sleep(poll_s)
        release.write_text("1", encoding="utf-8")
    else:
        while not release.exists():
            time.sleep(poll_s)


def _resolve_tuned_config_rank_safe(
    tuning_args: argparse.Namespace,
    run_dir: Path,
    rank: int,
    world: int,
) -> tuple[Dict[str, Any], Path]:
    worker_config_path = run_dir / "worker_config.json"
    tuned_config_path = run_dir / "tuned_config.json"

    if rank == 0:
        tuned, config_path = resolve_tuned_config(tuning_args)
        payload = {
            "inputs": {
                "profile": tuning_args.profile,
                "mode": tuning_args.mode,
                "physical_workers": tuning_args.physical_workers,
                "async_workers": tuning_args.async_workers,
                "stage_cpu_cap": tuning_args.stage_cpu_cap,
                "base_nodes": tuning_args.base_nodes,
                "base_files": tuning_args.base_files,
                "base_nodes_per_worker": tuning_args.base_nodes_per_worker,
                "node_chars": tuning_args.node_chars,
                "chunks_per_file": tuning_args.chunks_per_file,
                "dim": tuning_args.dim,
                "embedder_backend": tuning_args.embedder_backend,
                "set5_sink_backend": tuning_args.set5_sink_backend,
            },
            "tuned": tuned,
            "source_config_path": str(config_path),
        }
        worker_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tuned_config_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")

    _barrier(run_dir, "worker_config_ready", rank, world)

    payload = json.loads(worker_config_path.read_text(encoding="utf-8"))
    tuned = payload["tuned"]
    return tuned, worker_config_path


def _make_sink(backend: str, name: str, dim: int, persist_dir: str | None) -> Any:
    if backend == "faiss":
        return FaissSink(name, dim)
    if backend == "thin-batched":
        return ThinBatchedSink(name)
    return get_chroma_collection(persist_dir, name, reset=True)


def _make_rank_shard_dir(run_dir: Path, rank: int, local_files: List[str]) -> Path:
    shard_dir = run_dir / "rank_shards" / f"rank_{rank:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for src in local_files:
        dst = shard_dir / Path(src).name
        if dst.exists():
            continue
        os.symlink(src, dst)
    return shard_dir


def _corpus_cache_dir(base_dir: Path, total_nodes: int, total_files: int, node_chars: int, seed: int) -> Path:
    return base_dir / f"nodes_{total_nodes}__files_{total_files}__chars_{node_chars}__seed_{seed}"


def _corpus_ready_manifest(corpus_dir: Path) -> Path:
    return corpus_dir / "_READY.json"


def _ensure_cached_corpus(
    cache_root: Path,
    total_nodes: int,
    node_chars: int,
    total_files: int,
    seed: int,
    rank: int,
    world: int,
    run_dir: Path,
) -> Path:
    corpus_dir = _corpus_cache_dir(cache_root, total_nodes, total_files, node_chars, seed)
    manifest = _corpus_ready_manifest(corpus_dir)

    if manifest.exists():
        return corpus_dir

    if rank == 0:
        if corpus_dir.exists() and not manifest.exists():
            shutil.rmtree(corpus_dir, ignore_errors=True)
        corpus_dir.mkdir(parents=True, exist_ok=True)

    _barrier(run_dir, "corpus_cache_prepare", rank, world)

    if not manifest.exists():
        _write_synthetic_corpus_shard(
            data_dir=corpus_dir,
            total_nodes=total_nodes,
            node_chars=node_chars,
            total_files=total_files,
            seed=seed,
            rank=rank,
            world=world,
        )

    _barrier(run_dir, "corpus_cache_written", rank, world)

    if rank == 0 and not manifest.exists():
        manifest.write_text(json.dumps({
            "total_nodes": total_nodes,
            "total_files": total_files,
            "node_chars": node_chars,
            "seed": seed,
        }, indent=2), encoding="utf-8")

    _barrier(run_dir, "corpus_cache_ready", rank, world)
    return corpus_dir


def _nodes_before_file(file_index: int, total_nodes: int, total_files: int) -> int:
    base = total_nodes // total_files
    rem = total_nodes % total_files
    return file_index * base + min(file_index, rem)


def _nodes_in_file(file_index: int, total_nodes: int, total_files: int) -> int:
    base = total_nodes // total_files
    rem = total_nodes % total_files
    return base + (1 if file_index < rem else 0)


def _write_synthetic_corpus_shard(
    data_dir: Path,
    total_nodes: int,
    node_chars: int,
    total_files: int,
    seed: int,
    rank: int,
    world: int,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(rank, total_files, world):
        chunk_count = _nodes_in_file(fi, total_nodes, total_files)
        start_idx = _nodes_before_file(fi, total_nodes, total_files)
        parts = [_rand_text(node_chars, seed + start_idx + offset) for offset in range(chunk_count)]
        content = DELIM.join(parts)
        (data_dir / f"doc_{fi:04d}.txt").write_text(content, encoding="utf-8")




async def _run_local_ingest(
    data_dir: str,
    embedder: Any,
    sink: Any,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> ResultRow:
    t0 = time.perf_counter()
    node_count, load_s, transform_s, embed_s, upsert_s = await run_direct_batched_ingest(
        data_dir=data_dir,
        embedder=embedder,
        collections=[sink],
        io_workers=1,
        embed_batch_size=max(1, embed_batch_size),
        upsert_batch_size=max(1, upsert_batch_size),
        embed_num_workers=1,
        upsert_num_workers=1,
    )
    total_s = time.perf_counter() - t0
    return ResultRow(
        "AAFLOW",
        node_count,
        load_s,
        transform_s,
        embed_s,
        upsert_s,
        total_s,
    )


async def _run_local_ingest_streaming(
    data_dir: str,
    embedder: Any,
    sink: Any,
    embed_batch_size: int,
    upsert_batch_size: int,
    load_workers: int,
    transform_workers: int,
    embed_num_workers: int,
    upsert_num_workers: int,
    upsert_coalesce_multiplier: int,
) -> ResultRow:
    t0 = time.perf_counter()
    node_count, load_s, transform_s, embed_s, upsert_s = await embed_and_upsert_streaming(
        data_dir=data_dir,
        embedder=embedder,
        collections=[sink],
        load_workers=max(1, load_workers),
        transform_workers=max(1, transform_workers),
        embed_batch_size=max(1, embed_batch_size),
        upsert_batch_size=max(1, upsert_batch_size),
        embed_num_workers=max(1, embed_num_workers),
        upsert_num_workers=max(1, upsert_num_workers),
        upsert_coalesce_timeout_s=0.002,
        upsert_coalesce_multiplier=max(1, upsert_coalesce_multiplier),
    )
    total_s = time.perf_counter() - t0
    return ResultRow(
        "AAFLOW",
        node_count,
        load_s,
        transform_s,
        embed_s,
        upsert_s,
        total_s,
    )



async def _run_local_async_parallel_only(
    data_dir: str,
    embedder: Any,
    sink_backend: str,
    persist_dir: str | None,
    dim: int,
    rank: int,
) -> ResultRow:
    t0 = time.perf_counter()
    docs, load_s = load_docs_sync(data_dir)
    nodes_async, transform_s = await transform_async(docs, num_workers=1)
    collections = init_sink_collections(sink_backend, persist_dir, f"dist_set4_rank_{rank}", 1, dim)
    embed_s, upsert_s = await embed_and_upsert(
        nodes=nodes_async,
        embedder=embedder,
        collections=collections,
        embed_batch_size=1,
        upsert_batch_size=1,
        embed_num_workers=1,
        upsert_num_workers=1,
    )
    total_s = time.perf_counter() - t0
    return ResultRow("AsyncParallelOnly", len(nodes_async), load_s, transform_s, embed_s, upsert_s, total_s)


def _aggregate_rows(rows: List[Dict[str, Any]], config: str) -> ResultRow:
    return ResultRow(
        config,
        sum(int(r["nodes"]) for r in rows),
        max(float(r["load_s"]) for r in rows) if rows else 0.0,
        max(float(r["transform_s"]) for r in rows) if rows else 0.0,
        max(float(r["embed_s"]) for r in rows) if rows else 0.0,
        max(float(r["upsert_s"]) for r in rows) if rows else 0.0,
        max(float(r["total_s"]) for r in rows) if rows else 0.0,
        max(float(r.get("embed_plus_drain_s") or 0.0) for r in rows) if rows and any(r.get("embed_plus_drain_s") is not None for r in rows) else None,
        max(float(r.get("consumer_drain_s") or 0.0) for r in rows) if rows and any(r.get("consumer_drain_s") is not None for r in rows) else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed no-Ray AAFLOW scaling runner across Slurm tasks.")
    p.add_argument("--profile", choices=["strong", "weak"], required=True)
    p.add_argument("--physical-workers", type=int, required=True)
    p.add_argument("--stage-cpu-cap", type=int, default=1)
    p.add_argument("--base-nodes", type=int, default=65536)
    p.add_argument("--base-files", type=int, default=512)
    p.add_argument("--base-nodes-per-worker", type=int, default=128)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunks-per-file", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--embedder-backend", default="local-hash")
    p.add_argument("--set5-sink-backend", default="faiss")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--shared-corpus-root", default="/scratch/djy8hg/aaflow_data/drc_rag_scaling_corpus_cache")
    p.add_argument("--persist-dir", default="")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


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
        chunks_per_file=args.chunks_per_file,
        dim=args.dim,
        embedder_backend=args.embedder_backend,
        set5_sink_backend=args.set5_sink_backend,
        persist_dir=args.persist_dir,
        run_dir=str(run_dir),
        python_bin="",
        bench_script="",
        mode="agentic",
    )
    tuned, config_path = _resolve_tuned_config_rank_safe(tuning_args, run_dir, rank, world)

    total_nodes = tuned["nodes"]
    min_files_for_chunks = max(1, int(math.ceil(int(total_nodes) / max(1, args.chunks_per_file))))
    total_files = max(int(tuned["files"]), args.physical_workers, min_files_for_chunks)
    cache_root = Path(args.shared_corpus_root)
    data_dir = _ensure_cached_corpus(
        cache_root=cache_root,
        total_nodes=int(total_nodes),
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
                    "physical_workers": args.physical_workers,
                    "world_size": world,
                    "profile": args.profile,
                    "total_nodes": total_nodes,
                    "total_files": total_files,
                    "sink_backend": args.set5_sink_backend,
                    "embedder_backend": args.embedder_backend,
                    "shared_corpus_dir": str(data_dir),
                    "worker_config_path": str(config_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _barrier(run_dir, "corpus_ready", rank, world)

    file_paths = sorted(str(p) for p in data_dir.glob("*.txt"))
    local_files = file_paths[rank::world]

    if args.embedder_backend == "local-hash":
        embedder = LocalHashEmbedder(args.dim)
    else:
        embedder = FakeEmbedder(args.dim, 60.0, 1.2)

    shard_dir = _make_rank_shard_dir(run_dir, rank, local_files)
    persist_dir = args.persist_dir.strip() or None

    config_rows: List[ResultRow] = []
    if local_files:
        config_rows.append(
            asyncio.run(
                _run_local_async_parallel_only(
                    data_dir=str(shard_dir),
                    embedder=embedder,
                    sink_backend=args.set5_sink_backend,
                    persist_dir=persist_dir,
                    dim=args.dim,
                    rank=rank,
                )
            )
        )
    else:
        config_rows.append(ResultRow("AsyncParallelOnly", 0, 0.0, 0.0, 0.0, 0.0, 0.0))

    if local_files:
        config_rows.append(
            run_set7_dask_data(
                data_dir=str(shard_dir),
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.set5_sink_backend,
                dim=args.dim,
                embed_batch_size=int(tuned["embed_batch"]),
                upsert_batch_size=int(tuned["upsert_batch"]),
                dask_workers=1,
                upsert_workers=1,
                set45_upsert_shards=1,
            )
        )
        config_rows.append(
            run_set8_bsp_data(
                data_dir=str(shard_dir),
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.set5_sink_backend,
                dim=args.dim,
                embed_batch_size=int(tuned["embed_batch"]),
                upsert_batch_size=int(tuned["upsert_batch"]),
                bsp_workers=1,
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
                    sink_backend=args.set5_sink_backend,
                    dim=args.dim,
                    embed_batch_size=int(tuned["embed_batch"]),
                    upsert_batch_size=int(tuned["upsert_batch"]),
                    higress_workers=1,
                    upsert_workers=1,
                    set45_upsert_shards=1,
                )
            )
        )
        sink_name = f"dist_agentic_rank_{rank}"
        sink = _make_sink(args.set5_sink_backend, sink_name, args.dim, persist_dir)
        if args.profile == "weak" and args.set5_sink_backend == "faiss" and args.physical_workers <= 256:
            config_rows.append(
                asyncio.run(
                    _run_local_ingest_streaming(
                        data_dir=str(shard_dir),
                        embedder=embedder,
                        sink=sink,
                        embed_batch_size=min(1024, max(1, int(tuned.get("agentic_embed_batch", tuned["embed_batch"])))),
                        upsert_batch_size=min(8192, max(1, int(tuned.get("agentic_upsert_batch", tuned["upsert_batch"])))),
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
                        sink=sink,
                        embed_batch_size=max(1, int(tuned.get("agentic_embed_batch", tuned["embed_batch"]))),
                        upsert_batch_size=max(1, int(tuned.get("agentic_upsert_batch", tuned["upsert_batch"]))),
                    )
                )
            )
    else:
        config_rows.extend([
            ResultRow("DaskScalableRAG", 0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ResultRow("BulkSynchronousParallelRAG", 0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ResultRow("HigressRAG", 0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ResultRow("AAFLOW", 0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ])

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
            for row in json.loads(path.read_text(encoding="utf-8")):
                rows_by_config.setdefault(row["config"], []).append(row)
        order = ["AsyncParallelOnly", "DaskScalableRAG", "BulkSynchronousParallelRAG", "HigressRAG", "AAFLOW"]
        agg_rows = [_aggregate_rows(rows_by_config.get(name, []), name) for name in order if name in rows_by_config]
        print_table(agg_rows, baseline_label="AsyncParallelOnly")
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
