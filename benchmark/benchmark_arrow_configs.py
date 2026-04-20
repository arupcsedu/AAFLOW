#!/usr/bin/env python3
"""
Arrow-based Agentic DRC benchmark variants.

This file intentionally does not modify benchmark_configs_1_to_5.py.
It adds Arrow experiments for the Agentic DRC path only:

  1. AAFLOW+
     - thread-parallel file read
     - Arrow in-memory tables between stages
     - batched embed
     - batched sink upsert

  2. ArrowRayDataScalableRAG
     - Ray Data with Arrow batches
     - explicit Arrow load stage (path -> text)
     - explicit Arrow transform stage (text -> chunks)
     - batched embed
     - batched sink upsert

The goal is to evaluate whether Arrow lowers Python-object overhead in the
front half of the Agentic DRC pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from benchmark_configs_1_to_5 import (
    DELIM,
    FakeEmbedder,
    LocalHashEmbedder,
    ResultRow,
    chroma_upsert_batches,
    init_sink_collections,
    pct_faster,
    print_table,
    run_set6_ray_data,
    write_synthetic_corpus,
)


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for benchmark_arrow_configs.py") from exc
    return pa


def _read_text_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


def _list_text_files(data_dir: str) -> List[str]:
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".txt")
    )


def _make_chunk_id(path: str, chunk_idx: int) -> str:
    return f"{os.path.basename(path)}-{chunk_idx}"


def _hash_embed_vector(text: str, dim: int) -> List[float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
    out: List[float] = []
    for i in range(dim):
        b = digest[i % len(digest)]
        out.append((b / 127.5) - 1.0)
    return out


def _embed_texts_sync(
    texts: Sequence[str],
    dim: int,
    backend: str,
    request_overhead_ms: float,
    per_item_ms: float,
) -> List[List[float]]:
    if not texts:
        return []
    if backend == "fake":
        sleep_s = (request_overhead_ms + per_item_ms * len(texts)) / 1000.0
        time.sleep(sleep_s)
    return [_hash_embed_vector(text, dim) for text in texts]


def _batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    step = max(1, batch_size)
    for start in range(0, len(items), step):
        yield items[start:start + step]


def _arrow_chunk_records(path: str, text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for chunk_idx, chunk in enumerate(text.split(DELIM)):
        if not chunk:
            continue
        chunk_id = _make_chunk_id(path, chunk_idx)
        metadata_json = json.dumps({"path": path, "chunk_index": chunk_idx}, separators=(",", ":"))
        out.append((chunk_id, chunk, metadata_json))
    return out


def _load_arrow_table(data_dir: str, workers: int):
    pa = _require_pyarrow()
    file_paths = _list_text_files(data_dir)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        docs = list(executor.map(_read_text_file, file_paths))
    table = pa.table(
        {
            "path": [path for path, _ in docs],
            "text": [text for _, text in docs],
        }
    )
    t1 = time.perf_counter()
    return table, (t1 - t0)


def _transform_arrow_table(load_table) -> Tuple[Any, float]:
    pa = _require_pyarrow()
    t0 = time.perf_counter()
    ids: List[str] = []
    texts: List[str] = []
    metadata_json: List[str] = []
    for path, text in zip(load_table["path"].to_pylist(), load_table["text"].to_pylist()):
        for chunk_id, chunk, meta_json in _arrow_chunk_records(path, text):
            ids.append(chunk_id)
            texts.append(chunk)
            metadata_json.append(meta_json)
    out = pa.table({"id": ids, "text": texts, "metadata_json": metadata_json})
    t1 = time.perf_counter()
    return out, (t1 - t0)


async def run_arrow_no_ray(
    data_dir: str,
    persist_dir: str | None,
    sink_backend: str,
    dim: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    workers: int,
    embedder_backend: str,
    request_overhead_ms: float,
    per_item_ms: float,
) -> ResultRow:
    load_table, load_s = _load_arrow_table(data_dir, workers)
    node_table, transform_s = _transform_arrow_table(load_table)

    texts = node_table["text"].to_pylist()
    ids = node_table["id"].to_pylist()
    metadata_json = node_table["metadata_json"].to_pylist()
    metas = [json.loads(item) for item in metadata_json]

    te0 = time.perf_counter()
    embeddings: List[List[float]] = []
    for batch in _batch_iter(texts, embed_batch_size):
        embeddings.extend(
            _embed_texts_sync(
                batch,
                dim=dim,
                backend=embedder_backend,
                request_overhead_ms=request_overhead_ms,
                per_item_ms=per_item_ms,
            )
        )
    te1 = time.perf_counter()

    collections = init_sink_collections(
        sink_backend=sink_backend,
        persist_dir=persist_dir,
        base_name="arrow_no_ray_bench",
        count=1,
        dim=dim,
    )

    tu0 = time.perf_counter()
    await chroma_upsert_batches(
        collections=collections,
        ids=ids,
        embeddings=embeddings,
        metadatas=metas,
        documents=texts,
        upsert_batch_size=upsert_batch_size,
        num_workers=max(1, workers),
    )
    tu1 = time.perf_counter()

    embed_s = te1 - te0
    upsert_s = tu1 - tu0
    total_s = load_s + transform_s + embed_s + upsert_s

    return ResultRow(
        config="AAFLOW+",
        nodes=len(ids),
        load_s=load_s,
        transform_s=transform_s,
        embed_s=embed_s,
        upsert_s=upsert_s,
        total_s=total_s,
    )


def _ray_arrow_read_batch(batch):
    pa = _require_pyarrow()
    paths = batch["path"].to_pylist()
    out_paths: List[str] = []
    texts: List[str] = []
    for path in paths:
        _, text = _read_text_file(path)
        out_paths.append(path)
        texts.append(text)
    return pa.table({"path": out_paths, "text": texts})


def _ray_arrow_chunk_batch(batch):
    pa = _require_pyarrow()
    ids: List[str] = []
    texts: List[str] = []
    metadata_json: List[str] = []
    for path, text in zip(batch["path"].to_pylist(), batch["text"].to_pylist()):
        for chunk_id, chunk, meta_json in _arrow_chunk_records(path, text):
            ids.append(chunk_id)
            texts.append(chunk)
            metadata_json.append(meta_json)
    return pa.table({"id": ids, "text": texts, "metadata_json": metadata_json})


def _ray_arrow_embed_batch(batch, *, dim: int, backend: str, request_overhead_ms: float, per_item_ms: float):
    pa = _require_pyarrow()
    texts = batch["text"].to_pylist()
    embeddings = _embed_texts_sync(
        texts,
        dim=dim,
        backend=backend,
        request_overhead_ms=request_overhead_ms,
        per_item_ms=per_item_ms,
    )
    return pa.table(
        {
            "id": batch["id"].to_pylist(),
            "text": texts,
            "metadata_json": batch["metadata_json"].to_pylist(),
            "embedding": embeddings,
        }
    )


async def run_arrow_ray(
    data_dir: str,
    persist_dir: str | None,
    sink_backend: str,
    dim: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    workers: int,
    ray_num_cpus: int | None,
    ray_object_store_memory_mb: int,
    embedder_backend: str,
    request_overhead_ms: float,
    per_item_ms: float,
) -> ResultRow:
    try:
        import ray  # type: ignore
    except Exception as exc:
        raise RuntimeError("ray[data] is required for ArrowRayScalableRAG") from exc

    pa = _require_pyarrow()
    file_paths = _list_text_files(data_dir)
    if not ray.is_initialized():
        init_kwargs: Dict[str, Any] = {"ignore_reinit_error": True}
        if ray_num_cpus:
            init_kwargs["num_cpus"] = ray_num_cpus
        if ray_object_store_memory_mb > 0:
            init_kwargs["object_store_memory"] = int(ray_object_store_memory_mb * 1024 * 1024)
        ray.init(**init_kwargs)

    source = pa.table({"path": file_paths})
    ds_paths = ray.data.from_arrow(source)

    tl0 = time.perf_counter()
    ds_text = ds_paths.map_batches(_ray_arrow_read_batch, batch_format="pyarrow").materialize()
    tl1 = time.perf_counter()

    tt0 = time.perf_counter()
    ds_nodes = ds_text.map_batches(_ray_arrow_chunk_batch, batch_format="pyarrow").materialize()
    node_count = ds_nodes.count()
    tt1 = time.perf_counter()

    te0 = time.perf_counter()
    ds_emb = ds_nodes.map_batches(
        _ray_arrow_embed_batch,
        batch_format="pyarrow",
        batch_size=max(1, embed_batch_size),
        fn_kwargs={
            "dim": dim,
            "backend": embedder_backend,
            "request_overhead_ms": request_overhead_ms,
            "per_item_ms": per_item_ms,
        },
    )
    te1 = None

    collections = init_sink_collections(
        sink_backend=sink_backend,
        persist_dir=persist_dir,
        base_name="arrow_ray_bench",
        count=1,
        dim=dim,
    )

    tu0 = time.perf_counter()
    for batch in ds_emb.iter_batches(batch_size=max(1, upsert_batch_size), batch_format="pyarrow"):
        ids = batch["id"].to_pylist()
        texts = batch["text"].to_pylist()
        metadata_json = batch["metadata_json"].to_pylist()
        metas = [json.loads(item) for item in metadata_json]
        embeddings = batch["embedding"].to_pylist()
        await chroma_upsert_batches(
            collections=collections,
            ids=ids,
            embeddings=embeddings,
            metadatas=metas,
            documents=texts,
            upsert_batch_size=upsert_batch_size,
            num_workers=max(1, workers),
        )
    tu1 = time.perf_counter()
    te1 = tu1

    return ResultRow(
        config="ArrowRayDataScalableRAG",
        nodes=node_count,
        load_s=tl1 - tl0,
        transform_s=tt1 - tt0,
        embed_s=te1 - te0,
        upsert_s=tu1 - tu0,
        total_s=(tl1 - tl0) + (tt1 - tt0) + (te1 - te0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arrow Agentic DRC benchmark variants.")
    p.add_argument("--data-dir", type=str, default="", help="Existing corpus directory. If omitted, synthetic data is created.")
    p.add_argument("--persist-dir", type=str, default="", help="Optional sink persistence directory.")
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--files", type=int, default=200)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunks-per-file", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--workers", type=int, default=16, help="No-Ray worker count and sink concurrency.")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--upsert-batch-size", type=int, default=512)
    p.add_argument("--sink-backend", choices=["chroma", "thin-batched", "faiss"], default="faiss")
    p.add_argument("--embedder-backend", choices=["local-hash", "fake"], default="local-hash")
    p.add_argument("--request-overhead-ms", type=float, default=20.0)
    p.add_argument("--per-item-ms", type=float, default=0.5)
    p.add_argument("--ray-num-cpus", type=int, default=0)
    p.add_argument("--ray-object-store-memory-mb", type=int, default=0)
    p.add_argument("--run-no-ray", action="store_true", help="Run Arrow Agentic DRC path.")
    p.add_argument("--run-ray", action="store_true", help="Run Arrow Agentic DRC Ray path.")
    p.add_argument("--run-ray-baseline", action="store_true", help="Also run the baseline RayDataScalableRAG path.")
    p.add_argument("--summary-csv", type=str, default="", help="Optional CSV path for writing summary rows.")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    if not args.run_no_ray and not args.run_ray:
        args.run_no_ray = True
        args.run_ray = True

    data_dir = args.data_dir
    if not data_dir:
        import tempfile
        data_dir = tempfile.mkdtemp(prefix="arrow_bench_data_")
        write_synthetic_corpus(
            data_dir=data_dir,
            nodes=args.nodes,
            node_chars=args.node_chars,
            num_files=args.files,
            seed=args.seed,
            chunks_per_file=args.chunks_per_file,
        )

    rows: List[ResultRow] = []
    persist_dir = args.persist_dir or None

    if args.run_no_ray:
        rows.append(
            await run_arrow_no_ray(
                data_dir=data_dir,
                persist_dir=persist_dir,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=args.embed_batch_size,
                upsert_batch_size=args.upsert_batch_size,
                workers=args.workers,
                embedder_backend=args.embedder_backend,
                request_overhead_ms=args.request_overhead_ms,
                per_item_ms=args.per_item_ms,
            )
        )

    if args.run_ray:
        if args.run_ray_baseline:
            if args.embedder_backend == "local-hash":
                embedder = LocalHashEmbedder(args.dim)
            else:
                embedder = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)
            rows.append(
                run_set6_ray_data(
                    data_dir=data_dir,
                    persist_dir=persist_dir,
                    embedder=embedder,
                    sink_backend=args.sink_backend,
                    dim=args.dim,
                    embed_batch_size=args.embed_batch_size,
                    upsert_batch_size=args.upsert_batch_size,
                    ray_parallelism=max(1, args.workers),
                    upsert_workers=max(1, args.workers),
                    # This path connects to an already-running Ray cluster in the
                    # Slurm launcher, so connect-time resource sizing must not be
                    # forwarded here.
                    ray_num_cpus=None,
                    ray_object_store_memory_mb=0,
                    ray_input_format="raw",
                )
            )
        rows.append(
            await run_arrow_ray(
                data_dir=data_dir,
                persist_dir=persist_dir,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=args.embed_batch_size,
                upsert_batch_size=args.upsert_batch_size,
                workers=args.workers,
                ray_num_cpus=args.ray_num_cpus or None,
                ray_object_store_memory_mb=args.ray_object_store_memory_mb,
                embedder_backend=args.embedder_backend,
                request_overhead_ms=args.request_overhead_ms,
                per_item_ms=args.per_item_ms,
            )
        )

    baseline = "AsyncParallelOnly" if any(r.config == "AsyncParallelOnly" for r in rows) else (
        "RayDataScalableRAG" if any(r.config == "RayDataScalableRAG" for r in rows) else (
            "AAFLOW+" if any(r.config == "AAFLOW+" for r in rows) else None
        )
    )
    print_table(rows, baseline_label=baseline)
    if args.summary_csv:
        with open(args.summary_csv, "w", encoding="utf-8") as f:
            f.write("config,chunks,load_s,transform_s,embed_s,upsert_s,total_s\n")
            for row in rows:
                f.write(
                    f"{row.config},{row.nodes},{row.load_s},{row.transform_s},{row.embed_s},{row.upsert_s},{row.total_s}\n"
                )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
