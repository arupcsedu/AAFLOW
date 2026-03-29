#!/usr/bin/env python3
"""
Benchmark ingestion configs (Set 1..8) with configurable node count.
Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 06/01/2026

Implements:
  Set 1: Default
    - Sync load (SimpleDirectoryReader.load_data())
    - Sync pipeline (IngestionPipeline.run()) sequential transforms
    - Embed + upsert sequential, no batching

  Set 2: Reader Parallel
    - Parallel reader load (SimpleDirectoryReader.load_data(num_workers=8)) :contentReference[oaicite:1]{index=1}
    - Sync pipeline sequential transforms
    - Embed + upsert sequential, no batching

  Set 3: Pipeline Parallel (Sync)
    - Sync reader load
    - Sync pipeline with multiprocessing (IngestionPipeline.run(num_workers=K)) :contentReference[oaicite:2]{index=2}
    - Embed + upsert sequential, no batching

  Set 4: Async Pipeline (parallel concurrency only)
    - Sync reader load
    - Async pipeline transforms (IngestionPipeline.arun)
    - Embed: async with concurrency (num_workers), BUT embed_batch_size=1
    - Upsert: upsert_batch_size=1 (parallel upserts)

  Set 5: Async + Batching
    - Sync reader load
    - Async pipeline transforms
    - Embed: async + batching (embed_batch_size scales with workers)
    - Upsert: batched inserts (upsert_batch_size scales with workers, parallelized)

  Set 6: Ray Data Scalable RAG ingestion
    - Parallel file load with ray.data.read_text
    - Parallel chunking with map_batches
    - Parallel embedding with map_batches (batch_size=--set5-embed-batch)
    - Parallel upsert with Ray actor sink (batch_size=--set5-upsert-batch)

  Set 7: Dask Scalable RAG ingestion
    - Parallel file load with dask.bag
    - Parallel chunking with bag.map(...).flatten()
    - Parallel embedding with delayed batch tasks
    - Parallel upsert with sharded Chroma collections

  Set 8: Bulk Synchronous Parallel (BSP) ingestion
    - Stage-by-stage parallel load, chunk, embed, and upsert
    - Barrier between each wave of work
    - Uses the same embed/upsert batch knobs as Set5


    source /tmp/ragbench_env/bin/activate
    ray stop --force || true
    RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 RAY_raylet_start_wait_time_s=60 \
    python /project/bi_dsc_community/drc_rag/benchmark/benchmark_configs_1_to_5.py \
    --only-ray \
    --nodes 64 --files 8 --node-chars 900 \
    --async-workers 1 --upsert-workers-cap 1 \
    --set5-embed-batch 16 --set5-upsert-batch 32 \
    --ray-num-cpus 1 --ray-object-store-memory-mb 80

Notes:
- We use a delimiter-based splitter transform so node count is exactly controllable.
- We simulate embedding latency to show batching benefits reliably.

Dependencies:
  pip install llama-index chromadb ray[data] "dask[distributed]"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import string
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, TransformComponent


# -------------------------
# Synthetic data generation
# -------------------------

DELIM = "\n\n<<<NODE_SPLIT>>>\n\n"

def _rand_text(chars: int, seed: int) -> str:
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "     "
    return "".join(rnd.choice(alphabet) for _ in range(chars)).strip()

def write_synthetic_corpus(
    data_dir: str,
    nodes: int,
    node_chars: int,
    num_files: int,
    seed: int,
) -> None:
    """
    Writes num_files text files. The total number of node-chunks across all files is exactly `nodes`.
    Each node-chunk is separated by DELIM. Our splitter transform will create exactly one TextNode per chunk.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Distribute nodes across files as evenly as possible
    base = nodes // num_files
    rem = nodes % num_files
    per_file = [base + (1 if i < rem else 0) for i in range(num_files)]

    idx = 0
    for fi, k in enumerate(per_file):
        parts = []
        for _ in range(k):
            parts.append(_rand_text(node_chars, seed + idx))
            idx += 1
        content = DELIM.join(parts)
        with open(os.path.join(data_dir, f"doc_{fi:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(content)


# -------------------------
# LlamaIndex Transform: Delimiter splitter
# -------------------------

class DelimiterNodeSplitter(TransformComponent):
    """
    Custom splitter compatible with IngestionPipeline.
    Must inherit TransformComponent (pydantic model) per LlamaIndex docs. :contentReference[oaicite:1]{index=1}
    Splits each incoming item (Document or Node-like) by a delimiter and returns TextNodes.
    """
    delimiter: str = DELIM  # pydantic field

    def __call__(self, nodes, **kwargs):
        out: List[TextNode] = []

        for doc_i, obj in enumerate(nodes):
            # LlamaIndex may pass Documents first, then nodes through subsequent transforms.
            text = getattr(obj, "text", None)
            if text is None and hasattr(obj, "get_text"):
                text = obj.get_text()
            if text is None:
                text = str(obj)

            chunks = [c for c in text.split(self.delimiter) if c.strip()]

            base_id = getattr(obj, "doc_id", None) or getattr(obj, "id_", None) or f"item{doc_i}"
            base_meta = {}
            if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
                base_meta = dict(obj.metadata)

            for chunk_i, chunk in enumerate(chunks):
                node_id = f"{base_id}-chunk{chunk_i}"
                meta = dict(base_meta)
                meta.update({"doc_index": doc_i, "chunk_index": chunk_i})
                out.append(TextNode(id_=node_id, text=chunk, metadata=meta))

        return out


# -------------------------
# Fake embedding (API-like)
# -------------------------

class FakeEmbedder:
    """
    Simulates remote embedding API:
      latency(batch) = request_overhead_ms + per_item_ms * len(batch)
    Returns deterministic vectors of length `dim` derived from SHA256(text).
    """
    def __init__(self, dim: int, request_overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.request_overhead_ms = request_overhead_ms
        self.per_item_ms = per_item_ms

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = []
            for j in range(self.dim):
                b = h[j % len(h)]
                vec.append((b / 127.5) - 1.0)  # [-1, 1]
            out.append(vec)
        return out


# -------------------------
# Vector DB: Chroma helpers
# -------------------------

def get_chroma_collection(persist_dir: str | None, collection_name: str, reset: bool = False) -> Any:
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.EphemeralClient()

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def init_chroma(persist_dir: str | None, collection_name: str) -> Any:
    return get_chroma_collection(persist_dir, collection_name, reset=True)


def init_chroma_collections(persist_dir: str | None, base_name: str, count: int) -> List[Any]:
    out: List[Any] = []
    n = max(1, count)
    for i in range(n):
        suffix = f"_s{i}" if n > 1 else ""
        out.append(init_chroma(persist_dir, f"{base_name}{suffix}"))
    return out


async def chroma_upsert_batches(
    collections: Sequence[Any],
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
    documents: Sequence[str],
    upsert_batch_size: int,
    num_workers: int,
) -> None:
    n = len(ids)
    if n == 0:
        return

    upsert_batch_size = max(1, upsert_batch_size)
    num_workers = max(1, num_workers)

    collections = list(collections) if collections else []
    if not collections:
        return

    if num_workers == 1:
        for start in range(0, n, upsert_batch_size):
            end = min(n, start + upsert_batch_size)
            shard_idx = (start // upsert_batch_size) % len(collections)
            collections[shard_idx].add(
                ids=list(ids[start:end]),
                embeddings=[list(x) for x in embeddings[start:end]],
                metadatas=list(metadatas[start:end]),
                documents=list(documents[start:end]),
            )
            await asyncio.sleep(0)
        return

    sem = asyncio.Semaphore(num_workers)

    async def _upsert_batch(start: int, end: int, shard_idx: int) -> None:
        async with sem:
            await asyncio.to_thread(
                collections[shard_idx].add,
                ids=list(ids[start:end]),
                embeddings=[list(x) for x in embeddings[start:end]],
                metadatas=list(metadatas[start:end]),
                documents=list(documents[start:end]),
            )

    tasks = []
    for start in range(0, n, upsert_batch_size):
        end = min(n, start + upsert_batch_size)
        shard_idx = (start // upsert_batch_size) % len(collections)
        tasks.append(asyncio.create_task(_upsert_batch(start, end, shard_idx)))
    await asyncio.gather(*tasks)


# -------------------------
# Async embedding runner
# -------------------------

async def embed_all_async(
    embedder: FakeEmbedder,
    texts: Sequence[str],
    embed_batch_size: int,
    num_workers: int,
) -> List[List[float]]:
    if not texts:
        return []
    num_workers = max(1, num_workers)
    embed_batch_size = max(1, embed_batch_size)

    sem = asyncio.Semaphore(num_workers)

    async def _run_batch(batch: Sequence[str]) -> List[List[float]]:
        async with sem:
            return await embedder.embed_batch(batch)

    tasks = []
    for start in range(0, len(texts), embed_batch_size):
        batch = texts[start:start + embed_batch_size]
        tasks.append(asyncio.create_task(_run_batch(batch)))

    # preserve order
    batches = await asyncio.gather(*tasks)
    out: List[List[float]] = []
    for b in batches:
        out.extend(b)
    return out


# -------------------------
# Benchmark data structures
# -------------------------

@dataclass
class ResultRow:
    config: str
    nodes: int
    load_s: float
    transform_s: float
    embed_s: float
    upsert_s: float
    total_s: float


def pct_faster(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - new) / baseline * 100.0


# -------------------------
# Config implementations
# -------------------------

def load_docs_sync(data_dir: str) -> Tuple[List[Any], float]:
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(data_dir).load_data()
    t1 = time.perf_counter()
    return docs, (t1 - t0)

def load_docs_parallel(data_dir: str, num_workers: int) -> Tuple[List[Any], float]:
    # SimpleDirectoryReader parallel load example uses load_data(num_workers=...) :contentReference[oaicite:3]{index=3}
    t0 = time.perf_counter()
    reader = SimpleDirectoryReader(data_dir)
    docs = reader.load_data(num_workers=num_workers)
    t1 = time.perf_counter()
    return docs, (t1 - t0)

def transform_sync(docs: List[Any], num_workers: int | None) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # IngestionPipeline.run supports parallel processing with num_workers (multiprocessing.Pool) :contentReference[oaicite:4]{index=4}
    nodes = pipeline.run(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)

async def transform_async(docs: List[Any], num_workers: int) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # Async pipeline exists; num_workers controls outgoing concurrency as semaphore in async examples :contentReference[oaicite:5]{index=5}
    nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)

async def embed_and_upsert(
    nodes: List[TextNode],
    embedder: FakeEmbedder,
    collections: Sequence[Any],
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
    upsert_num_workers: int,
) -> Tuple[float, float]:
    texts = [n.text for n in nodes]
    ids = [n.id_ for n in nodes]
    metas = [dict(n.metadata or {}) for n in nodes]

    te0 = time.perf_counter()
    embs = await embed_all_async(embedder, texts, embed_batch_size=embed_batch_size, num_workers=embed_num_workers)
    te1 = time.perf_counter()

    tu0 = time.perf_counter()
    await chroma_upsert_batches(
        collections=collections,
        ids=ids,
        embeddings=embs,
        metadatas=metas,
        documents=texts,
        upsert_batch_size=upsert_batch_size,
        num_workers=upsert_num_workers,
    )
    tu1 = time.perf_counter()

    return (te1 - te0), (tu1 - tu0)


# -------------------------
# Set 6: Ray Data scalable ingestion
# -------------------------

def _hash_embed_vector(text: str, dim: int) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec: List[float] = []
    for j in range(dim):
        b = h[j % len(h)]
        vec.append((b / 127.5) - 1.0)
    return vec


def _ray_chunk_batch(batch: Dict[str, Any], delimiter: str) -> Dict[str, List[Any]]:
    texts = batch["text"]
    paths = batch.get("path")

    out_ids: List[str] = []
    out_texts: List[str] = []
    out_metas: List[str] = []

    for row_i, text in enumerate(texts):
        if text is None:
            continue
        path = str(paths[row_i]) if paths is not None else f"row_{row_i}"
        chunks = [c for c in str(text).split(delimiter) if c.strip()]
        for chunk_i, chunk in enumerate(chunks):
            node_id = f"{os.path.basename(path)}-{chunk_i}-{uuid.uuid4().hex}"
            meta = {"path": path, "row_index": row_i, "chunk_index": chunk_i}
            out_ids.append(node_id)
            out_texts.append(chunk)
            out_metas.append(json.dumps(meta, separators=(",", ":")))

    return {"id": out_ids, "text": out_texts, "metadata_json": out_metas}


def _ray_decode_binary_batch(batch: Dict[str, Any]) -> Dict[str, List[str]]:
    paths = batch.get("path")
    blobs = batch.get("bytes")
    out_paths: List[str] = []
    out_texts: List[str] = []

    for row_i, blob in enumerate(blobs):
        if blob is None:
            continue
        path = str(paths[row_i]) if paths is not None else f"row_{row_i}"
        out_paths.append(path)
        out_texts.append(bytes(blob).decode("utf-8", errors="replace"))
    return {"path": out_paths, "text": out_texts}


def _ray_embed_batch(
    batch: Dict[str, Any],
    dim: int,
    request_overhead_ms: float,
    per_item_ms: float,
) -> Dict[str, List[Any]]:
    texts = [str(t) for t in batch["text"]]
    sleep_s = (request_overhead_ms + per_item_ms * len(texts)) / 1000.0
    time.sleep(sleep_s)

    embeddings = [_hash_embed_vector(t, dim) for t in texts]
    return {
        "id": [str(x) for x in batch["id"]],
        "text": texts,
        "metadata_json": [str(x) for x in batch["metadata_json"]],
        "embedding": embeddings,
    }


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    step = max(1, batch_size)
    for start in range(0, len(items), step):
        yield items[start:start + step]


def _read_text_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


def _chunk_text_record(path: str, text: str, delimiter: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    chunks = [c for c in text.split(delimiter) if c.strip()]
    for chunk_i, chunk in enumerate(chunks):
        node_id = f"{os.path.basename(path)}-{chunk_i}-{uuid.uuid4().hex}"
        meta = {"path": path, "chunk_index": chunk_i}
        out.append((node_id, chunk, meta))
    return out


def _embed_batch_sync(
    batch: Sequence[Tuple[str, str, Dict[str, Any]]],
    dim: int,
    request_overhead_ms: float,
    per_item_ms: float,
) -> List[Tuple[str, str, Dict[str, Any], List[float]]]:
    time.sleep((request_overhead_ms + per_item_ms * len(batch)) / 1000.0)
    out: List[Tuple[str, str, Dict[str, Any], List[float]]] = []
    for node_id, text, meta in batch:
        out.append((node_id, text, meta, _hash_embed_vector(text, dim)))
    return out


def _upsert_batch_sync(
    persist_dir: str | None,
    collection_name: str,
    batch: Sequence[Tuple[str, str, Dict[str, Any], List[float]]],
) -> int:
    if not batch:
        return 0
    collection = get_chroma_collection(persist_dir, collection_name, reset=False)
    collection.add(
        ids=[x[0] for x in batch],
        documents=[x[1] for x in batch],
        metadatas=[x[2] for x in batch],
        embeddings=[x[3] for x in batch],
    )
    return len(batch)


def _ray_upsert_sink_batch(batch: Dict[str, Any], actors: Sequence[Any]) -> Dict[str, List[int]]:
    ids = [str(x) for x in batch["id"]]
    texts = [str(x) for x in batch["text"]]
    metas = [json.loads(str(x)) for x in batch["metadata_json"]]
    embs = [list(v) for v in batch["embedding"]]

    if not ids:
        return {"upserted": []}

    actor_idx = int(hashlib.md5(ids[0].encode("utf-8")).hexdigest(), 16) % len(actors)
    ray = __import__("ray")
    ray.get(actors[actor_idx].upsert.remote(ids, embs, metas, texts))
    return {"upserted": [len(ids)]}


def run_set6_ray_data(
    data_dir: str,
    persist_dir: str | None,
    embedder: FakeEmbedder,
    embed_batch_size: int,
    upsert_batch_size: int,
    ray_parallelism: int,
    upsert_workers: int,
    ray_num_cpus: int | None,
    ray_object_store_memory_mb: int,
) -> ResultRow:
    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Ray is required for Set6. Install with `pip install ray[data]`.") from exc

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            object_store_memory=max(80, ray_object_store_memory_mb) * 1024 * 1024,
        )

    @ray.remote(num_cpus=0)
    class ChromaSinkActor:
        def __init__(self, persist: str | None, collection_name: str):
            self.collection = init_chroma(persist, collection_name)
            self.total = 0

        def upsert(
            self,
            ids: Sequence[str],
            embeddings: Sequence[Sequence[float]],
            metadatas: Sequence[Dict[str, Any]],
            documents: Sequence[str],
        ) -> int:
            if not ids:
                return 0
            self.collection.add(
                ids=list(ids),
                embeddings=[list(x) for x in embeddings],
                metadatas=list(metadatas),
                documents=list(documents),
            )
            self.total += len(ids)
            return len(ids)

        def count(self) -> int:
            return self.total

    t0 = time.perf_counter()

    # Step 1: load files in parallel (preserve full-file content for delimiter-based chunking)
    tl0 = time.perf_counter()
    ds = ray.data.read_binary_files(data_dir, parallelism=max(1, ray_parallelism))
    ds = ds.map_batches(
        _ray_decode_binary_batch,
        batch_format="numpy",
        batch_size=max(1, upsert_batch_size),
    ).materialize()
    tl1 = time.perf_counter()

    # Step 2: chunk into nodes in parallel
    tt0 = time.perf_counter()
    ds_nodes = ds.map_batches(
        _ray_chunk_batch,
        fn_kwargs={"delimiter": DELIM},
        batch_format="numpy",
        batch_size=max(1, embed_batch_size),
    ).materialize()
    node_count = ds_nodes.count()
    tt1 = time.perf_counter()

    # Step 3: embed in parallel
    te0 = time.perf_counter()
    ds_emb = ds_nodes.map_batches(
        _ray_embed_batch,
        fn_kwargs={
            "dim": embedder.dim,
            "request_overhead_ms": embedder.request_overhead_ms,
            "per_item_ms": embedder.per_item_ms,
        },
        batch_format="numpy",
        batch_size=max(1, embed_batch_size),
    ).materialize()
    te1 = time.perf_counter()

    # Step 4: actor sink upsert in parallel
    tu0 = time.perf_counter()
    sink_workers = max(1, upsert_workers)
    actors = [
        ChromaSinkActor.remote(persist_dir, f"bench_set6_actor_{i}")
        for i in range(sink_workers)
    ]
    pending = []
    batch_size = max(1, upsert_batch_size)
    for batch in ds_emb.iter_batches(batch_size=batch_size, batch_format="numpy"):
        ids = [str(x) for x in batch["id"]]
        texts = [str(x) for x in batch["text"]]
        metas = [json.loads(str(x)) for x in batch["metadata_json"]]
        embs = [list(v) for v in batch["embedding"]]
        if not ids:
            continue
        actor_idx = int(hashlib.md5(ids[0].encode("utf-8")).hexdigest(), 16) % sink_workers
        pending.append(actors[actor_idx].upsert.remote(ids, embs, metas, texts))
        if len(pending) >= sink_workers * 4:
            ray.get(pending[:sink_workers])
            pending = pending[sink_workers:]
    if pending:
        ray.get(pending)
    _ = ray.get([a.count.remote() for a in actors])
    tu1 = time.perf_counter()

    total_s = time.perf_counter() - t0
    return ResultRow(
        "RayDataScalableRAG",
        node_count,
        tl1 - tl0,
        tt1 - tt0,
        te1 - te0,
        tu1 - tu0,
        total_s,
    )


def run_set7_dask_data(
    data_dir: str,
    persist_dir: str | None,
    embedder: FakeEmbedder,
    embed_batch_size: int,
    upsert_batch_size: int,
    dask_workers: int,
    upsert_workers: int,
    set45_upsert_shards: int,
) -> ResultRow:
    try:
        import dask
        import dask.bag as db
    except Exception as exc:
        raise RuntimeError("Dask is required for Set7. Install with `pip install dask distributed`.") from exc

    file_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".txt")
    )
    workers = max(1, dask_workers)
    shards = max(1, set45_upsert_shards)
    shard_names = [f"bench_set7_dask_s{i}" for i in range(shards)]
    for name in shard_names:
        init_chroma(persist_dir, name)

    t0 = time.perf_counter()

    tl0 = time.perf_counter()
    docs_bag = db.from_sequence(file_paths, npartitions=min(workers, max(1, len(file_paths))))
    docs = docs_bag.map(_read_text_file).compute(scheduler="threads", num_workers=workers)
    tl1 = time.perf_counter()

    tt0 = time.perf_counter()
    chunk_bag = db.from_sequence(docs, npartitions=min(workers, max(1, len(docs))))
    chunked = chunk_bag.map(lambda item: _chunk_text_record(item[0], item[1], DELIM)).flatten()
    node_records = list(chunked.compute(scheduler="threads", num_workers=workers))
    tt1 = time.perf_counter()

    te0 = time.perf_counter()
    embed_tasks = [
        dask.delayed(_embed_batch_sync)(
            batch,
            embedder.dim,
            embedder.request_overhead_ms,
            embedder.per_item_ms,
        )
        for batch in _iter_batches(node_records, embed_batch_size)
    ]
    embedded_batches = dask.compute(*embed_tasks, scheduler="threads", num_workers=workers)
    embedded_records = [item for batch in embedded_batches for item in batch]
    te1 = time.perf_counter()

    tu0 = time.perf_counter()
    upsert_concurrency = max(1, min(upsert_workers, workers))
    upsert_tasks = []
    for batch_idx, batch in enumerate(_iter_batches(embedded_records, upsert_batch_size)):
        shard_name = shard_names[batch_idx % len(shard_names)]
        upsert_tasks.append(dask.delayed(_upsert_batch_sync)(persist_dir, shard_name, batch))
    if upsert_tasks:
        dask.compute(*upsert_tasks, scheduler="threads", num_workers=upsert_concurrency)
    tu1 = time.perf_counter()

    total_s = time.perf_counter() - t0
    return ResultRow(
        "DaskScalableRAG",
        len(node_records),
        tl1 - tl0,
        tt1 - tt0,
        te1 - te0,
        tu1 - tu0,
        total_s,
    )


def _run_bsp_stage(
    executor: ThreadPoolExecutor,
    items: Sequence[Any],
    fn,
    stage_width: int,
) -> List[Any]:
    if not items:
        return []
    out: List[Any] = []
    width = max(1, stage_width)
    it = iter(items)
    while True:
        chunk = list(islice(it, width))
        if not chunk:
            break
        futures = [executor.submit(fn, item) for item in chunk]
        for future in futures:
            out.append(future.result())
    return out


def _run_bsp_batch_stage(
    executor: ThreadPoolExecutor,
    batches: Sequence[Sequence[Any]],
    fn,
    stage_width: int,
) -> List[Any]:
    if not batches:
        return []
    out: List[Any] = []
    width = max(1, stage_width)
    it = iter(batches)
    while True:
        wave = list(islice(it, width))
        if not wave:
            break
        futures = [executor.submit(fn, batch) for batch in wave]
        for future in futures:
            result = future.result()
            if isinstance(result, list):
                out.extend(result)
            else:
                out.append(result)
    return out


def run_set8_bsp_data(
    data_dir: str,
    persist_dir: str | None,
    embedder: FakeEmbedder,
    embed_batch_size: int,
    upsert_batch_size: int,
    bsp_workers: int,
    upsert_workers: int,
    set45_upsert_shards: int,
) -> ResultRow:
    file_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".txt")
    )
    workers = max(1, bsp_workers)
    shards = max(1, set45_upsert_shards)
    shard_names = [f"bench_set8_bsp_s{i}" for i in range(shards)]
    for name in shard_names:
        init_chroma(persist_dir, name)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tl0 = time.perf_counter()
        docs = _run_bsp_stage(executor, file_paths, _read_text_file, workers)
        tl1 = time.perf_counter()

        tt0 = time.perf_counter()
        chunked_lists = _run_bsp_stage(
            executor,
            docs,
            lambda item: _chunk_text_record(item[0], item[1], DELIM),
            workers,
        )
        node_records = [node for nodes in chunked_lists for node in nodes]
        tt1 = time.perf_counter()

        te0 = time.perf_counter()
        embed_batches = list(_iter_batches(node_records, embed_batch_size))
        embedded_records = _run_bsp_batch_stage(
            executor,
            embed_batches,
            lambda batch: _embed_batch_sync(
                batch,
                embedder.dim,
                embedder.request_overhead_ms,
                embedder.per_item_ms,
            ),
            workers,
        )
        te1 = time.perf_counter()

        tu0 = time.perf_counter()
        upsert_batches = list(_iter_batches(embedded_records, upsert_batch_size))
        upsert_width = max(1, min(upsert_workers, workers))
        _run_bsp_batch_stage(
            executor,
            [
                (shard_names[batch_idx % len(shard_names)], batch)
                for batch_idx, batch in enumerate(upsert_batches)
            ],
            lambda item: _upsert_batch_sync(persist_dir, item[0], item[1]),
            upsert_width,
        )
        tu1 = time.perf_counter()

    total_s = time.perf_counter() - t0
    return ResultRow(
        "BulkSynchronousParallelRAG",
        len(node_records),
        tl1 - tl0,
        tt1 - tt0,
        te1 - te0,
        tu1 - tu0,
        total_s,
    )


# -------------------------
# Pretty table
# -------------------------

def print_table(rows: List[ResultRow], baseline_label: str | None = None) -> None:
    if not rows:
        return

    if baseline_label is None:
        baseline_label = rows[0].config
    baseline_total = next((r.total_s for r in rows if r.config == baseline_label), rows[0].total_s)

    headers = [
        "Config",
        "Chunks",
        "Load(s)",
        "Transform(s)",
        "Embed(s)",
        "Upsert(s)",
        "Total(s)",
        f"Δ vs {baseline_label}",
    ]

    # Build formatted strings
    data = []
    for r in rows:
        delta = f"{pct_faster(baseline_total, r.total_s):.1f}% faster" if r.config != baseline_label else "baseline"
        data.append([
            r.config,
            str(r.nodes),
            f"{r.load_s:.3f}",
            f"{r.transform_s:.3f}",
            f"{r.embed_s:.3f}",
            f"{r.upsert_s:.3f}",
            f"{r.total_s:.3f}",
            delta,
        ])

    # Column widths
    cols = list(zip(headers, *data))
    widths = [max(len(x) for x in col) for col in cols]

    def fmt_row(items: Sequence[str]) -> str:
        return " | ".join(s.ljust(w) for s, w in zip(items, widths))

    sep = "-+-".join("-" * w for w in widths)

    print("\n" + fmt_row(headers))
    print(sep)
    for row in data:
        print(fmt_row(row))
    print()


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ingestion configs Set1..Set8.")
    p.add_argument("--nodes", type=int, default=1024, help="Exact number of nodes to create.")
    p.add_argument("--node-chars", type=int, default=900, help="Characters per node (controls text size).")
    p.add_argument("--files", type=int, default=200, help="Number of files for SimpleDirectoryReader to load.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Reader/pipeline parallelism
    p.add_argument("--reader-workers", type=int, default=8, help="Set2 reader parallel workers.")
    p.add_argument("--pipeline-workers", type=int, default=8, help="Set3 pipeline multiprocessing workers.")

    # Async concurrency
    p.add_argument("--async-workers", type=int, default=64, help="Set4/5 async concurrency for transforms and embeddings.")
    p.add_argument(
        "--no-cap-async-workers",
        action="store_true",
        help="Do not cap async workers to CPU count (use requested value).",
    )
    p.add_argument("--only-async", action="store_true", help="Run only Set4/Set5 (skip Set1-Set3).")

    # Set5 batching
    p.add_argument(
        "--set5-embed-batch",
        type=int,
        default=64,
        help="Set5 embedding base batch size (scaled by async workers).",
    )
    p.add_argument(
        "--set5-upsert-batch",
        type=int,
        default=256,
        help="Set5 upsert base batch size (scaled by async workers).",
    )
    p.add_argument(
        "--batch-scale-baseline",
        type=int,
        default=32,
        help="Worker count used as baseline for scaling Set5 batch sizes.",
    )
    p.add_argument(
        "--no-scale-set5-batches",
        action="store_true",
        help="Use fixed Set5 embed/upsert batch sizes instead of scaling them with workers.",
    )
    p.add_argument(
        "--set45-upsert-shards",
        type=int,
        default=1,
        help="Number of Chroma collections to shard Set4/Set5 upserts across (1 = single collection).",
    )
    p.add_argument(
        "--strict-stage-scaling",
        action="store_true",
        help=(
            "Favor worker scaling: fix Set5 batch sizes, set upsert cap to async workers, "
            "and shard Set4/Set5 upserts by worker count."
        ),
    )
    p.add_argument(
        "--upsert-workers-cap",
        type=int,
        default=0,
        help="Cap upsert concurrency to this value (0 = auto-tune from CPU count).",
    )
    p.add_argument(
        "--no-cap-upsert-workers",
        action="store_true",
        help="Do not cap upsert concurrency (use async workers).",
    )

    # Sweeps/graphs
    p.add_argument(
        "--graph-async-workers",
        type=str,
        default="",
        help="Comma list (e.g., 8,16,32,64) or range start:end:step to sweep async workers.",
    )
    p.add_argument("--graph-out", type=str, default="async_workers.png", help="Output path for async worker plot.")
    p.add_argument("--graph-csv", type=str, default="async_workers.csv", help="Output path for async worker CSV.")
    p.add_argument("--run-ray-set6", action="store_true", help="Run Set6 Ray Data scalable ingestion experiment.")
    p.add_argument("--run-dask-set7", action="store_true", help="Run Set7 Dask scalable ingestion experiment.")
    p.add_argument("--run-bsp-set8", action="store_true", help="Run Set8 BSP scalable ingestion experiment.")
    p.add_argument("--only-ray", action="store_true", help="Run only Set6 Ray Data experiment.")
    p.add_argument("--only-dask", action="store_true", help="Run only Set7 Dask experiment.")
    p.add_argument("--only-bsp", action="store_true", help="Run only Set8 BSP experiment.")
    p.add_argument("--ray-num-cpus", type=int, default=0, help="Optional CPUs for ray.init (0 = Ray default).")
    p.add_argument(
        "--ray-object-store-memory-mb",
        type=int,
        default=128,
        help="Ray plasma object store size in MB for Set6.",
    )
    p.add_argument(
        "--dask-workers",
        type=int,
        default=0,
        help="Optional worker count for Set7 Dask experiment (0 = use effective async workers).",
    )
    p.add_argument(
        "--bsp-workers",
        type=int,
        default=0,
        help="Optional worker count for Set8 BSP experiment (0 = use effective async workers).",
    )

    # Fake embedder latency model
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    p.add_argument("--request-overhead-ms", type=float, default=60.0, help="Fixed overhead per embedding request (ms).")
    p.add_argument("--per-item-ms", type=float, default=1.2, help="Per-item cost inside embedding request (ms).")

    # Chroma
    p.add_argument("--persist-dir", type=str, default="", help="Optional Chroma persistence dir (empty = in-memory).")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.only_ray:
        args.run_ray_set6 = True
    if args.only_dask:
        args.run_dask_set7 = True
    if args.only_bsp:
        args.run_bsp_set8 = True
    persist_dir = args.persist_dir.strip() or None
    cpu_count = os.cpu_count() or 1
    if args.no_cap_async_workers:
        effective_async_workers = max(1, args.async_workers)
    else:
        effective_async_workers = min(max(1, args.async_workers), cpu_count)
        if effective_async_workers != args.async_workers:
            print(
                f"Requested async-workers={args.async_workers} exceeds CPU count ({cpu_count}); "
                f"capping to {effective_async_workers}."
            )

    upsert_workers_cap = max(0, args.upsert_workers_cap)
    set45_upsert_shards = max(1, args.set45_upsert_shards)
    if args.strict_stage_scaling:
        args.no_scale_set5_batches = True
        set45_upsert_shards = effective_async_workers
        upsert_workers_cap = effective_async_workers
    if args.no_cap_upsert_workers:
        effective_upsert_workers = effective_async_workers
    else:
        if upsert_workers_cap == 0:
            upsert_workers_cap = max(1, cpu_count // 2)
        effective_upsert_workers = min(effective_async_workers, upsert_workers_cap)
    effective_dask_workers = max(1, args.dask_workers) if args.dask_workers > 0 else effective_async_workers
    effective_bsp_workers = max(1, args.bsp_workers) if args.bsp_workers > 0 else effective_async_workers

    def scale_set5_batches(total_items: int, workers: int) -> Tuple[int, int]:
        if args.no_scale_set5_batches:
            embed_batch = max(1, args.set5_embed_batch)
            upsert_batch = max(1, args.set5_upsert_batch)
            if total_items > 0:
                embed_batch = min(embed_batch, total_items)
                upsert_batch = min(upsert_batch, total_items)
            return embed_batch, upsert_batch
        baseline = max(1, args.batch_scale_baseline)
        scale = max(1.0, workers / baseline)
        embed_batch = max(1, int(math.ceil(args.set5_embed_batch * scale)))
        upsert_batch = max(1, int(math.ceil(args.set5_upsert_batch * scale)))
        if total_items > 0:
            embed_batch = min(embed_batch, total_items)
            upsert_batch = min(upsert_batch, total_items)
        return embed_batch, upsert_batch

    def parse_worker_list(spec: str) -> List[int]:
        if not spec:
            return []
        if ":" in spec:
            parts = [p.strip() for p in spec.split(":")]
            if len(parts) != 3:
                raise ValueError("Range format must be start:end:step")
            start, end, step = (int(p) for p in parts)
            if step <= 0:
                raise ValueError("Range step must be > 0")
            return list(range(start, end + 1, step))
        return [int(p.strip()) for p in spec.split(",") if p.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        write_synthetic_corpus(
            data_dir=data_dir,
            nodes=args.nodes,
            node_chars=args.node_chars,
            num_files=args.files,
            seed=args.seed,
        )

        embedder = FakeEmbedder(
            dim=args.dim,
            request_overhead_ms=args.request_overhead_ms,
            per_item_ms=args.per_item_ms,
        )

        rows: List[ResultRow] = []

        async_workers_sweep = parse_worker_list(args.graph_async_workers)
        if async_workers_sweep:
            sweep_rows: List[Dict[str, Any]] = []
            for requested_workers in async_workers_sweep:
                workers = min(max(1, requested_workers), cpu_count)

                t0 = time.perf_counter()
                docs, load_s = load_docs_sync(data_dir)
                nodes_async, transform_s = await transform_async(docs, num_workers=workers)
                collections = init_chroma_collections(
                    persist_dir, f"bench_set4_w{workers}", min(set45_upsert_shards, workers)
                )
                embed_s, upsert_s = await embed_and_upsert(
                    nodes=nodes_async,
                    embedder=embedder,
                    collections=collections,
                    embed_batch_size=1,
                    upsert_batch_size=1,
                    embed_num_workers=workers,
                    upsert_num_workers=min(workers, upsert_workers_cap) if upsert_workers_cap else workers,
                )
                total_s = time.perf_counter() - t0
                set4_total = total_s

                t0 = time.perf_counter()
                docs, load_s = load_docs_sync(data_dir)
                nodes_async, transform_s = await transform_async(docs, num_workers=workers)
                collections = init_chroma_collections(
                    persist_dir, f"bench_set5_w{workers}", min(set45_upsert_shards, workers)
                )
                embed_batch, upsert_batch = scale_set5_batches(len(nodes_async), workers)
                embed_s, upsert_s = await embed_and_upsert(
                    nodes=nodes_async,
                    embedder=embedder,
                    collections=collections,
                    embed_batch_size=embed_batch,
                    upsert_batch_size=upsert_batch,
                    embed_num_workers=workers,
                    upsert_num_workers=min(workers, upsert_workers_cap) if upsert_workers_cap else workers,
                )
                total_s = time.perf_counter() - t0
                set5_total = total_s

                sweep_rows.append(
                    {
                        "requested_workers": requested_workers,
                        "effective_workers": workers,
                        "set4_total_s": set4_total,
                        "set5_total_s": set5_total,
                    }
                )

            with open(args.graph_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["requested_workers", "effective_workers", "set4_total_s", "set5_total_s"],
                )
                writer.writeheader()
                writer.writerows(sweep_rows)

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                xs = [row["effective_workers"] for row in sweep_rows]
                set4 = [row["set4_total_s"] for row in sweep_rows]
                set5 = [row["set5_total_s"] for row in sweep_rows]

                plt.figure(figsize=(8, 4.5))
                plt.plot(xs, set4, marker="o", label="AsyncParallelOnly (Set4)")
                plt.plot(xs, set5, marker="o", label="AgenticDRC (Set5)")
                plt.title("Async Workers vs Total Time")
                plt.xlabel("Async Workers (effective)")
                plt.ylabel("Total Time (s)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(args.graph_out, dpi=150)
                print(f"\nWrote sweep CSV to {args.graph_csv} and plot to {args.graph_out}\n")
            except Exception as exc:
                print(f"\nWrote sweep CSV to {args.graph_csv}. Plot skipped: {exc}\n")
            return

        only_scalable = args.only_ray or args.only_dask or args.only_bsp

        if not args.only_async and not only_scalable:
            # ---------------- Set 1 ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes, transform_s = transform_sync(docs, num_workers=None)
            collections = init_chroma_collections(persist_dir, "bench_set1", 1)
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collections=collections,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,  # no concurrency to match "no workers anywhere"
                upsert_num_workers=1,
            )
            total_s = time.perf_counter() - t0
            rows.append(ResultRow("LoaderParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

            # ---------------- Set 2 ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_parallel(data_dir, num_workers=args.reader_workers)
            nodes, transform_s = transform_sync(docs, num_workers=None)
            collections = init_chroma_collections(persist_dir, "bench_set2", 1)
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collections=collections,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,  # keep "pipeline still sync" and no extra embed concurrency
                upsert_num_workers=1,
            )
            total_s = time.perf_counter() - t0
            rows.append(ResultRow("ReaderParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

            # ---------------- Set 3 ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes, transform_s = transform_sync(docs, num_workers=args.pipeline_workers)
            collections = init_chroma_collections(persist_dir, "bench_set3", 1)
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collections=collections,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,  # keep embedding/upsert sequential to isolate pipeline parallelism
                upsert_num_workers=1,
            )
            total_s = time.perf_counter() - t0
            rows.append(ResultRow("PipelineParallelSync", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))
 
        if not only_scalable:
            # ---------------- Set 4 ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes_async, transform_s = await transform_async(docs, num_workers=effective_async_workers)
            collections = init_chroma_collections(persist_dir, "bench_set4", set45_upsert_shards)
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes_async,
                embedder=embedder,
                collections=collections,
                embed_batch_size=1,   # key: no batching
                upsert_batch_size=1,  # key: no batching
                embed_num_workers=effective_async_workers,  # concurrency only
                upsert_num_workers=effective_upsert_workers,
            )
            total_s = time.perf_counter() - t0
            rows.append(ResultRow("AsyncParallelOnly", len(nodes_async), load_s, transform_s, embed_s, upsert_s, total_s))

            # ---------------- Set 5 ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes_async, transform_s = await transform_async(docs, num_workers=effective_async_workers)
            collections = init_chroma_collections(persist_dir, "bench_set5", set45_upsert_shards)
            embed_batch, upsert_batch = scale_set5_batches(len(nodes_async), effective_async_workers)
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes_async,
                embedder=embedder,
                collections=collections,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                embed_num_workers=effective_async_workers,
                upsert_num_workers=effective_upsert_workers,
            )
            total_s = time.perf_counter() - t0
            rows.append(ResultRow("AgenticDRC", len(nodes_async), load_s, transform_s, embed_s, upsert_s, total_s))

        if args.run_ray_set6:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_async_workers)
            ray_row = run_set6_ray_data(
                data_dir=data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                ray_parallelism=effective_async_workers,
                upsert_workers=effective_upsert_workers,
                ray_num_cpus=args.ray_num_cpus if args.ray_num_cpus > 0 else None,
                ray_object_store_memory_mb=args.ray_object_store_memory_mb,
            )
            rows.append(ray_row)

        if args.run_dask_set7:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_dask_workers)
            dask_row = run_set7_dask_data(
                data_dir=data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                dask_workers=effective_dask_workers,
                upsert_workers=effective_upsert_workers,
                set45_upsert_shards=set45_upsert_shards,
            )
            rows.append(dask_row)

        if args.run_bsp_set8:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_bsp_workers)
            bsp_row = run_set8_bsp_data(
                data_dir=data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                bsp_workers=effective_bsp_workers,
                upsert_workers=effective_upsert_workers,
                set45_upsert_shards=set45_upsert_shards,
            )
            rows.append(bsp_row)

        baseline_label = "AsyncParallelOnly" if args.only_async and not only_scalable else None
        print_table(rows, baseline_label=baseline_label)

        # Small highlight: Set5 vs Set4
        r4 = next((r for r in rows if r.config == "AsyncParallelOnly"), None)
        r5 = next((r for r in rows if r.config == "AgenticDRC"), None)
        if r4 and r5:
            print(
                "AgenticDRC vs AsyncParallelOnly total improvement: "
                f"{pct_faster(r4.total_s, r5.total_s):.1f}% faster with {effective_async_workers} Workers\n"
            )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
