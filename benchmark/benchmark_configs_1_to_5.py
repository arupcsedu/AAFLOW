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

  Set 9: Higress RAG
    - Thin direct ingestion path without LlamaIndex pipeline orchestration
    - Parallel file load with thread pool
    - Direct delimiter-based chunking with thread pool
    - Async batched embedding with the same fake embedder model
    - Batched parallel upsert with optional Chroma sharding


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
import functools
import hashlib
import json
import math
import os
import random
import string
import tempfile
import time
import uuid
import threading
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice, repeat
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
    chunks_per_file: int = 0,
) -> None:
    """
    Writes num_files text files. The total number of node-chunks across all files is exactly `nodes`.
    Each node-chunk is separated by DELIM. Our splitter transform will create exactly one TextNode per chunk.
    """
    os.makedirs(data_dir, exist_ok=True)

    if chunks_per_file > 0:
        num_files = max(1, int(math.ceil(nodes / max(1, chunks_per_file))))
    num_files = max(1, min(num_files, nodes)) if nodes > 0 else max(1, num_files)

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

    def embed_batch_sync(self, texts: Sequence[str]) -> List[List[float]]:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        time.sleep(sleep_s)

        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = []
            for j in range(self.dim):
                b = h[j % len(h)]
                vec.append((b / 127.5) - 1.0)  # [-1, 1]
            out.append(vec)
        return out

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_batch_sync, texts)


class LocalHashEmbedder:
    """
    Thin local embedder with no artificial request latency.
    It uses a deterministic hashing trick and vectorized numpy expansion when
    available to keep the embed stage CPU-thin.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.request_overhead_ms = 0.0
        self.per_item_ms = 0.0
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None
        self._np = np

    def _embed_one(self, text: str) -> List[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        if self._np is None:
            return _hash_embed_vector(text, self.dim)

        arr = self._np.frombuffer(digest, dtype=self._np.uint8).astype(self._np.float32)
        arr = (arr / self._np.float32(127.5)) - self._np.float32(1.0)
        if self.dim <= arr.size:
            return arr[: self.dim].tolist()
        repeats = int(math.ceil(self.dim / arr.size))
        tiled = self._np.tile(arr, repeats)[: self.dim]
        return tiled.tolist()

    def embed_batch_sync(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_batch_sync, texts)


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


_SINK_REGISTRY: Dict[Tuple[str, str], Any] = {}


class ThinBatchedSink:
    """
    Minimal in-memory sink used to benchmark Set5 orchestration without Chroma
    write-path overhead dominating the result.
    """

    def __init__(self, name: str):
        self.name = name
        self._count = 0
        self._lock = threading.Lock()

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        with self._lock:
            self._count += len(ids)


class FaissSink:
    """
    Thin FAISS-backed sink for measuring a lighter vector-store write path than
    Chroma. Search is not benchmarked here; only batched add throughput matters.
    """

    def __init__(self, name: str, dim: int):
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("FAISS backend requested but faiss-cpu is not installed.") from exc
        self.name = name
        self.dim = dim
        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatL2(dim)
        self._count = 0
        self._lock = threading.Lock()

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        if not ids:
            return
        vecs = self._np.asarray(embeddings, dtype=self._np.float32)
        with self._lock:
            self._index.add(vecs)
            self._count += len(ids)


def init_sink_collections(
    sink_backend: str,
    persist_dir: str | None,
    base_name: str,
    count: int,
    dim: int,
) -> List[Any]:
    if sink_backend == "thin-batched":
        n = max(1, count)
        out = []
        for i in range(n):
            name = f"{base_name}_s{i}" if n > 1 else base_name
            sink = ThinBatchedSink(name)
            _SINK_REGISTRY[(sink_backend, name)] = sink
            out.append(sink)
        return out
    if sink_backend == "faiss":
        n = max(1, count)
        out = []
        for i in range(n):
            name = f"{base_name}_s{i}" if n > 1 else base_name
            sink = FaissSink(name, dim)
            _SINK_REGISTRY[(sink_backend, name)] = sink
            out.append(sink)
        return out
    return init_chroma_collections(persist_dir, base_name, count)


def get_sink_collection(
    sink_backend: str,
    persist_dir: str | None,
    collection_name: str,
    dim: int,
    reset: bool = False,
) -> Any:
    if sink_backend == "chroma":
        return get_chroma_collection(persist_dir, collection_name, reset=reset)
    key = (sink_backend, collection_name)
    if reset or key not in _SINK_REGISTRY:
        if sink_backend == "thin-batched":
            _SINK_REGISTRY[key] = ThinBatchedSink(collection_name)
        elif sink_backend == "faiss":
            _SINK_REGISTRY[key] = FaissSink(collection_name, dim)
        else:
            raise ValueError(f"Unsupported sink backend: {sink_backend}")
    return _SINK_REGISTRY[key]


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
    embed_plus_drain_s: float | None = None
    consumer_drain_s: float | None = None


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

def _split_doc_to_nodes(item: Tuple[int, Any]) -> List[TextNode]:
    doc_i, obj = item
    text = getattr(obj, "text", None)
    if text is None and hasattr(obj, "get_text"):
        text = obj.get_text()
    if text is None:
        text = str(obj)

    chunks = [c for c in str(text).split(DELIM) if c.strip()]
    base_id = getattr(obj, "doc_id", None) or getattr(obj, "id_", None) or f"item{doc_i}"
    base_meta = {}
    if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
        base_meta = dict(obj.metadata)

    out: List[TextNode] = []
    for chunk_i, chunk in enumerate(chunks):
        node_id = f"{base_id}-chunk{chunk_i}"
        meta = dict(base_meta)
        meta.update({"doc_index": doc_i, "chunk_index": chunk_i})
        out.append(TextNode(id_=node_id, text=chunk, metadata=meta))
    return out

async def transform_async(docs: List[Any], num_workers: int) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # Async pipeline exists; num_workers controls outgoing concurrency as semaphore in async examples :contentReference[oaicite:5]{index=5}
    nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)

async def transform_agentic_fast(docs: List[Any], num_workers: int) -> Tuple[List[TextNode], float]:
    t0 = time.perf_counter()
    workers = max(1, num_workers)
    indexed_docs = list(enumerate(docs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        batches = list(executor.map(_split_doc_to_nodes, indexed_docs))
    nodes = [node for batch in batches for node in batch]
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


async def run_set5_thin_faiss(
    data_dir: str,
    embedder: Any,
    collections: Sequence[Any],
    load_workers: int,
    transform_workers: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
) -> Tuple[int, float, float, float, float]:
    io_workers = max(1, max(load_workers, transform_workers))
    return await run_direct_batched_ingest(
        data_dir=data_dir,
        embedder=embedder,
        collections=list(collections[:1] or collections),
        io_workers=io_workers,
        embed_batch_size=max(1, embed_batch_size),
        upsert_batch_size=max(1, upsert_batch_size),
        embed_num_workers=max(1, embed_num_workers),
        upsert_num_workers=1,
    )


async def run_direct_batched_ingest(
    data_dir: str,
    embedder: Any,
    collections: Sequence[Any],
    io_workers: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
    upsert_num_workers: int,
) -> Tuple[int, float, float, float, float]:
    """
    Thin direct ingestion path shared by Higress and Set5 FAISS mode:
      parallel file read
      direct delimiter chunking
      plain-text batched embed
      batched sink add
    """
    file_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".txt")
    )
    if not file_paths:
        return 0, 0.0, 0.0, 0.0, 0.0

    workers = max(1, io_workers)
    tl0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        docs = list(executor.map(_read_text_file, file_paths))
    tl1 = time.perf_counter()

    tt0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunked_lists = list(executor.map(lambda item: _chunk_text_record(item[0], item[1], DELIM), docs))
    node_records = [node for nodes in chunked_lists for node in nodes]
    tt1 = time.perf_counter()

    te0 = time.perf_counter()
    texts = [record[1] for record in node_records]
    if texts and getattr(embedder, "request_overhead_ms", 0.0) == 0.0 and getattr(embedder, "per_item_ms", 0.0) == 0.0:
        embs = embedder.embed_batch_sync(texts)
    else:
        embs = await embed_all_async(
            embedder,
            texts,
            embed_batch_size=max(1, embed_batch_size),
            num_workers=max(1, embed_num_workers),
        ) if node_records else []
    te1 = time.perf_counter()

    tu0 = time.perf_counter()
    ids = [record[0] for record in node_records]
    metas = [record[2] for record in node_records]
    if node_records:
        await chroma_upsert_batches(
            collections=list(collections),
            ids=ids,
            embeddings=embs,
            metadatas=metas,
            documents=texts,
            upsert_batch_size=max(1, upsert_batch_size),
            num_workers=max(1, upsert_num_workers),
        )
    tu1 = time.perf_counter()

    return len(node_records), tl1 - tl0, tt1 - tt0, te1 - te0, tu1 - tu0


async def embed_and_upsert_streaming(
    data_dir: str,
    embedder: FakeEmbedder,
    collections: Sequence[Any],
    load_workers: int,
    transform_workers: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
    upsert_num_workers: int,
    upsert_coalesce_timeout_s: float,
    upsert_coalesce_multiplier: int,
) -> Tuple[int, float, float, float, float]:
    file_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".txt")
    )
    if not file_paths:
        return 0, 0.0, 0.0, 0.0, 0.0

    load_workers = max(1, load_workers)
    transform_workers = max(1, transform_workers)
    embed_num_workers = max(1, embed_num_workers)
    upsert_num_workers = max(1, upsert_num_workers)
    embed_batch_size = max(1, embed_batch_size)
    upsert_batch_size = max(1, upsert_batch_size)
    upsert_coalesce_timeout_s = max(0.0, upsert_coalesce_timeout_s)
    upsert_coalesce_multiplier = max(1, upsert_coalesce_multiplier)

    load_q: asyncio.Queue[Tuple[str, str] | None] = asyncio.Queue(maxsize=max(2, load_workers * 2))
    node_q: asyncio.Queue[Tuple[str, str, Dict[str, Any]] | None] = asyncio.Queue(maxsize=max(2, transform_workers * embed_batch_size))
    embed_batch_q: asyncio.Queue[List[Tuple[str, str, Dict[str, Any]]] | None] = asyncio.Queue(maxsize=max(2, embed_num_workers * 2))
    upsert_input_q: asyncio.Queue[Tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]] | None] = asyncio.Queue(
        maxsize=max(2, upsert_num_workers * 2)
    )
    shard_count = max(1, len(collections))
    collections = list(collections[:shard_count])
    upsert_ready_q: asyncio.Queue[Tuple[int, List[str], List[List[float]], List[Dict[str, Any]], List[str]] | None] = asyncio.Queue(
        maxsize=max(2, shard_count * 2)
    )

    load_s_total = 0.0
    transform_s_total = 0.0
    embed_s_total = 0.0
    upsert_s_total = 0.0
    node_count = 0
    batch_counter = 0
    metric_lock = asyncio.Lock()
    batch_lock = asyncio.Lock()

    load_pool = ThreadPoolExecutor(max_workers=load_workers)
    transform_pool = ThreadPoolExecutor(max_workers=transform_workers)
    embed_pool = ThreadPoolExecutor(max_workers=embed_num_workers)
    upsert_pool = ThreadPoolExecutor(max_workers=upsert_num_workers)
    loop = asyncio.get_running_loop()

    async def load_one(path: str) -> None:
        nonlocal load_s_total
        start = time.perf_counter()
        _, text = await loop.run_in_executor(load_pool, _read_text_file, path)
        elapsed = time.perf_counter() - start
        async with metric_lock:
            load_s_total += elapsed
        await load_q.put((path, text))

    async def transform_worker() -> None:
        nonlocal transform_s_total, node_count
        while True:
            item = await load_q.get()
            if item is None:
                break
            path, text = item
            start = time.perf_counter()
            records = await loop.run_in_executor(transform_pool, _chunk_text_record, path, text, DELIM)
            elapsed = time.perf_counter() - start
            async with metric_lock:
                transform_s_total += elapsed
                node_count += len(records)
            for record in records:
                await node_q.put(record)

    async def embed_batcher() -> None:
        batch: List[Tuple[str, str, Dict[str, Any]]] = []
        finished = 0
        while finished < transform_workers:
            item = await node_q.get()
            if item is None:
                finished += 1
                continue
            batch.append(item)
            if len(batch) >= embed_batch_size:
                await embed_batch_q.put(batch)
                batch = []
        if batch:
            await embed_batch_q.put(batch)
        for _ in range(embed_num_workers):
            await embed_batch_q.put(None)

    async def embed_worker() -> None:
        nonlocal embed_s_total
        while True:
            batch = await embed_batch_q.get()
            if batch is None:
                break
            start = time.perf_counter()
            embedded = await loop.run_in_executor(
                embed_pool,
                _embed_batch_with_embedder_sync,
                batch,
                embedder,
            )
            elapsed = time.perf_counter() - start
            async with metric_lock:
                embed_s_total += elapsed
            ids = [x[0] for x in embedded]
            texts = [x[1] for x in embedded]
            metas = [x[2] for x in embedded]
            embs = [x[3] for x in embedded]
            await upsert_input_q.put((ids, embs, metas, texts))

    async def upsert_batcher() -> None:
        nonlocal batch_counter
        pending_by_shard: List[Tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]]] = [
            ([], [], [], []) for _ in range(shard_count)
        ]
        first_pending_at: List[float | None] = [None for _ in range(shard_count)]
        finished = 0
        max_target = max(upsert_batch_size, upsert_batch_size * upsert_coalesce_multiplier)

        async def flush_shard(shard_idx: int) -> None:
            nonlocal batch_counter
            pending_ids, pending_embs, pending_metas, pending_docs = pending_by_shard[shard_idx]
            if not pending_ids:
                return
            await upsert_ready_q.put((shard_idx, pending_ids, pending_embs, pending_metas, pending_docs))
            pending_by_shard[shard_idx] = ([], [], [], [])
            first_pending_at[shard_idx] = None
            async with batch_lock:
                batch_counter += 1

        async def flush_all() -> None:
            for shard_idx in range(shard_count):
                await flush_shard(shard_idx)

        while finished < embed_num_workers:
            timeout = None
            if upsert_coalesce_timeout_s > 0:
                now = time.perf_counter()
                expiries = [
                    max(0.0, upsert_coalesce_timeout_s - (now - ts))
                    for ts in first_pending_at
                    if ts is not None
                ]
                if expiries:
                    timeout = min(expiries)
            try:
                item = await asyncio.wait_for(upsert_input_q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                now = time.perf_counter()
                for shard_idx, ts in enumerate(first_pending_at):
                    if ts is not None and (now - ts) >= upsert_coalesce_timeout_s:
                        await flush_shard(shard_idx)
                continue

            if item is None:
                finished += 1
                continue
            ids, embs, metas, docs = item
            for item_idx, item_id in enumerate(ids):
                shard_idx = (hash(item_id) & 0x7FFFFFFF) % shard_count
                pending_ids, pending_embs, pending_metas, pending_docs = pending_by_shard[shard_idx]
                if not pending_ids:
                    first_pending_at[shard_idx] = time.perf_counter()
                pending_ids.append(item_id)
                pending_embs.append(embs[item_idx])
                pending_metas.append(metas[item_idx])
                pending_docs.append(docs[item_idx])
                pending_by_shard[shard_idx] = (pending_ids, pending_embs, pending_metas, pending_docs)
                backlog_bonus = upsert_ready_q.qsize() * max(1, embed_batch_size // max(1, shard_count))
                worker_pressure = max(0, embed_num_workers - upsert_num_workers) * max(1, embed_batch_size // max(1, shard_count))
                adaptive_target = min(max_target, upsert_batch_size + backlog_bonus + worker_pressure)
                if len(pending_ids) >= adaptive_target:
                    await flush_shard(shard_idx)
        await flush_all()
        for _ in range(upsert_num_workers):
            await upsert_ready_q.put(None)

    async def upsert_worker() -> None:
        nonlocal upsert_s_total
        while True:
            item = await upsert_ready_q.get()
            if item is None:
                break
            shard_idx, ids, embs, metas, docs = item
            collection = collections[shard_idx]
            start = time.perf_counter()
            await loop.run_in_executor(
                upsert_pool,
                functools.partial(
                    collection.add,
                    ids=ids,
                    embeddings=embs,
                    metadatas=metas,
                    documents=docs,
                ),
            )
            elapsed = time.perf_counter() - start
            async with metric_lock:
                upsert_s_total += elapsed

    t_stream_start = time.perf_counter()
    t_load_done = t_stream_start
    t_transform_done = t_stream_start
    t_embed_done = t_stream_start
    t_upsert_done = t_stream_start

    try:
        load_tasks = [asyncio.create_task(load_one(path)) for path in file_paths]
        transform_tasks = [asyncio.create_task(transform_worker()) for _ in range(transform_workers)]
        batcher_task = asyncio.create_task(embed_batcher())
        embed_tasks = [asyncio.create_task(embed_worker()) for _ in range(embed_num_workers)]
        upsert_batcher_task = asyncio.create_task(upsert_batcher())
        upsert_tasks = [asyncio.create_task(upsert_worker()) for _ in range(upsert_num_workers)]

        await asyncio.gather(*load_tasks)
        t_load_done = time.perf_counter()
        for _ in range(transform_workers):
            await load_q.put(None)
        await asyncio.gather(*transform_tasks)
        t_transform_done = time.perf_counter()
        for _ in range(transform_workers):
            await node_q.put(None)
        await batcher_task
        await asyncio.gather(*embed_tasks)
        t_embed_done = time.perf_counter()
        for _ in range(embed_num_workers):
            await upsert_input_q.put(None)
        await upsert_batcher_task
        await asyncio.gather(*upsert_tasks)
        t_upsert_done = time.perf_counter()
    finally:
        load_pool.shutdown(wait=False, cancel_futures=True)
        transform_pool.shutdown(wait=False, cancel_futures=True)
        embed_pool.shutdown(wait=False, cancel_futures=True)
        upsert_pool.shutdown(wait=False, cancel_futures=True)

    # Report a wall-clock stage decomposition that sums to total runtime.
    # The raw cumulative worker times above are useful for debugging but are not
    # directly comparable in the benchmark table because the streaming stages overlap.
    load_wall_s = max(0.0, t_load_done - t_stream_start)
    transform_wall_s = max(0.0, t_transform_done - t_load_done)
    embed_wall_s = max(0.0, t_embed_done - t_transform_done)
    upsert_wall_s = max(0.0, t_upsert_done - t_embed_done)

    return node_count, load_wall_s, transform_wall_s, embed_wall_s, upsert_wall_s


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


def _ray_chunk_batch(
    batch: Dict[str, Any],
    delimiter: str,
    include_metadata: bool = True,
) -> Dict[str, List[Any]]:
    texts = batch["text"]
    paths = batch.get("path")

    out_ids: List[str] = []
    out_texts: List[str] = []
    out_metas: List[str] = []

    for row_i, text in enumerate(texts):
        if text is None:
            continue
        path = str(paths[row_i]) if paths is not None else f"row_{row_i}"
        chunks = [c for c in str(text).split(delimiter) if c]
        base_name = os.path.basename(path)
        for chunk_i, chunk in enumerate(chunks):
            # Deterministic ids are cheaper than uuid4() and still unique per file/chunk.
            node_id = f"{base_name}-{row_i}-{chunk_i}"
            out_ids.append(node_id)
            out_texts.append(chunk)
            if include_metadata:
                meta = {"path": path, "row_index": row_i, "chunk_index": chunk_i}
                out_metas.append(json.dumps(meta, separators=(",", ":")))

    out: Dict[str, List[Any]] = {"id": out_ids, "text": out_texts}
    if include_metadata:
        out["metadata_json"] = out_metas
    return out


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


def _ray_decode_and_chunk_binary_batch(
    batch: Dict[str, Any],
    delimiter: str,
    include_metadata: bool = True,
) -> Dict[str, List[Any]]:
    paths = batch.get("path")
    blobs = batch.get("bytes")
    out_ids: List[str] = []
    out_texts: List[str] = []
    out_metas: List[str] = []

    for row_i, blob in enumerate(blobs):
        if blob is None:
            continue
        path = str(paths[row_i]) if paths is not None else f"row_{row_i}"
        text = bytes(blob).decode("utf-8", errors="replace")
        chunks = [c for c in text.split(delimiter) if c.strip()]
        base_name = os.path.basename(path)
        for chunk_i, chunk in enumerate(chunks):
            node_id = f"{base_name}-0-{chunk_i}"
            out_ids.append(node_id)
            out_texts.append(chunk)
            if include_metadata:
                meta = {"path": path, "row_index": 0, "chunk_index": chunk_i}
                out_metas.append(json.dumps(meta, separators=(",", ":")))

    out: Dict[str, List[Any]] = {"id": out_ids, "text": out_texts}
    if include_metadata:
        out["metadata_json"] = out_metas
    return out


def _ray_read_and_chunk_path_batch(
    batch: Dict[str, Any],
    delimiter: str,
    include_metadata: bool = True,
) -> Dict[str, List[Any]]:
    paths = batch.get("path")
    if paths is None:
        paths = []
    out_ids: List[str] = []
    out_texts: List[str] = []
    out_metas: List[str] = []

    for row_i, path_value in enumerate(paths):
        path = str(path_value)
        id_prefix = f"{os.path.basename(path)}-0-"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = [c for c in text.split(delimiter) if c]
        for chunk_i, chunk in enumerate(chunks):
            node_id = f"{id_prefix}{chunk_i}"
            out_ids.append(node_id)
            out_texts.append(chunk)
            meta = {"path": path, "row_index": 0, "chunk_index": chunk_i}
            out_metas.append(json.dumps(meta, separators=(",", ":")))

    out: Dict[str, List[Any]] = {"id": out_ids, "text": out_texts}
    if include_metadata:
        out["metadata_json"] = out_metas
    return out


def _ray_read_prechunked_path_batch(
    batch: Dict[str, Any],
    include_metadata: bool = True,
) -> Dict[str, List[Any]]:
    paths = batch.get("path")
    if paths is None:
        paths = []
    out_ids: List[str] = []
    out_texts: List[str] = []
    out_metas: List[str] = []

    for row_i, path_value in enumerate(paths):
        path = str(path_value)
        id_prefix = f"{os.path.basename(path)}-0-"
        with open(path, "r", encoding="utf-8") as f:
            for chunk_i, line in enumerate(f):
                chunk = line.rstrip("\n")
                if not chunk:
                    continue
                out_ids.append(f"{id_prefix}{chunk_i}")
                out_texts.append(chunk)
                if include_metadata:
                    meta = {"path": path, "row_index": 0, "chunk_index": chunk_i}
                    out_metas.append(json.dumps(meta, separators=(",", ":")))

    out: Dict[str, List[Any]] = {"id": out_ids, "text": out_texts}
    if include_metadata:
        out["metadata_json"] = out_metas
    return out


def _ray_read_preembedded_path_batch(
    batch: Dict[str, Any],
) -> Dict[str, List[Any]]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("Preembedded Ray input requires numpy.") from exc

    paths = batch.get("path")
    if paths is None:
        paths = []

    out_ids: List[str] = []
    emb_parts = []
    for path_value in paths:
        path = str(path_value)
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["ids"]
            embeddings = payload["embeddings"]
            out_ids.extend(str(x) for x in ids.tolist())
            emb_parts.append(np.asarray(embeddings, dtype=np.float32))

    if emb_parts:
        embeddings = np.concatenate(emb_parts, axis=0)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    return {"id": out_ids, "embedding": embeddings}


def _ray_embed_batch(
    batch: Dict[str, Any],
    dim: int,
    request_overhead_ms: float,
    per_item_ms: float,
    keep_payload: bool = True,
) -> Dict[str, List[Any]]:
    texts = [str(t) for t in batch["text"]]
    sleep_s = (request_overhead_ms + per_item_ms * len(texts)) / 1000.0
    time.sleep(sleep_s)
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None and request_overhead_ms == 0.0 and per_item_ms == 0.0:
        byte_lut = getattr(_ray_embed_batch, "_byte_lut", None)
        if byte_lut is None:
            byte_lut = (np.arange(256, dtype=np.float32) / np.float32(127.5)) - np.float32(1.0)
            _ray_embed_batch._byte_lut = byte_lut  # type: ignore[attr-defined]
        repeat_idx_cache = getattr(_ray_embed_batch, "_repeat_idx_cache", None)
        if repeat_idx_cache is None:
            repeat_idx_cache = {}
            _ray_embed_batch._repeat_idx_cache = repeat_idx_cache  # type: ignore[attr-defined]
        embeddings = np.empty((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
            arr = byte_lut[np.frombuffer(digest, dtype=np.uint8)]
            if dim <= arr.size:
                embeddings[i, :] = arr[:dim]
            else:
                repeat_idx = repeat_idx_cache.get((arr.size, dim))
                if repeat_idx is None:
                    repeats = int(math.ceil(dim / arr.size))
                    repeat_idx = np.arange(arr.size * repeats, dtype=np.int32)[:dim] % arr.size
                    repeat_idx_cache[(arr.size, dim)] = repeat_idx
                embeddings[i, :] = arr[repeat_idx]
    else:
        embeddings = [_hash_embed_vector(t, dim) for t in texts]
    out: Dict[str, List[Any]] = {
        "id": [str(x) for x in batch["id"]],
        "embedding": embeddings,
    }
    if keep_payload:
        out["text"] = texts
        out["metadata_json"] = [str(x) for x in batch["metadata_json"]]
    return out


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    step = max(1, batch_size)
    for start in range(0, len(items), step):
        yield items[start:start + step]


def _read_text_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


def _chunk_text_record(path: str, text: str, delimiter: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    chunks = [c for c in text.split(delimiter) if c]
    for chunk_i, chunk in enumerate(chunks):
        node_id = f"{os.path.basename(path)}-{chunk_i}-{uuid.uuid4().hex}"
        meta = {"path": path, "chunk_index": chunk_i}
        out.append((node_id, chunk, meta))
    return out


def _read_and_chunk_file(path: str, delimiter: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    _, text = _read_text_file(path)
    return _chunk_text_record(path, text, delimiter)


def _nodes_before_file(file_index: int, total_nodes: int, total_files: int) -> int:
    base = total_nodes // total_files
    rem = total_nodes % total_files
    return file_index * base + min(file_index, rem)


def _nodes_in_file(file_index: int, total_nodes: int, total_files: int) -> int:
    base = total_nodes // total_files
    rem = total_nodes % total_files
    return base + (1 if file_index < rem else 0)


def _ensure_shared_synthetic_corpus(
    cache_root: str,
    total_nodes: int,
    node_chars: int,
    total_files: int,
    seed: int,
) -> str:
    corpus_dir = os.path.join(
        cache_root,
        f"nodes_{total_nodes}__files_{total_files}__chars_{node_chars}__seed_{seed}",
    )
    manifest = os.path.join(corpus_dir, "manifest.json")
    if os.path.exists(manifest):
        return corpus_dir

    os.makedirs(corpus_dir, exist_ok=True)
    for fi in range(total_files):
        chunk_count = _nodes_in_file(fi, total_nodes, total_files)
        start_idx = _nodes_before_file(fi, total_nodes, total_files)
        parts = [_rand_text(node_chars, seed + start_idx + offset) for offset in range(chunk_count)]
        with open(os.path.join(corpus_dir, f"doc_{fi:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(DELIM.join(parts))
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_nodes": total_nodes,
                "total_files": total_files,
                "node_chars": node_chars,
                "seed": seed,
            },
            f,
            indent=2,
        )
    return corpus_dir


def _ensure_shared_prechunked_synthetic_corpus(
    cache_root: str,
    total_nodes: int,
    node_chars: int,
    total_files: int,
    seed: int,
) -> str:
    corpus_dir = os.path.join(
        cache_root,
        f"nodes_{total_nodes}__files_{total_files}__chars_{node_chars}__seed_{seed}__prechunked_lines",
    )
    manifest = os.path.join(corpus_dir, "manifest.json")
    if os.path.exists(manifest):
        return corpus_dir

    raw_dir = _ensure_shared_synthetic_corpus(
        cache_root=cache_root,
        total_nodes=total_nodes,
        node_chars=node_chars,
        total_files=total_files,
        seed=seed,
    )
    os.makedirs(corpus_dir, exist_ok=True)
    for fi in range(total_files):
        raw_path = os.path.join(raw_dir, f"doc_{fi:04d}.txt")
        dst_path = os.path.join(corpus_dir, f"doc_{fi:04d}.txt")
        with open(raw_path, "r", encoding="utf-8") as src:
            chunks = [chunk for chunk in src.read().split(DELIM) if chunk]
        with open(dst_path, "w", encoding="utf-8") as dst:
            if chunks:
                dst.write("\n".join(chunks))
                dst.write("\n")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_nodes": total_nodes,
                "total_files": total_files,
                "node_chars": node_chars,
                "seed": seed,
                "format": "prechunked_lines",
            },
            f,
            indent=2,
        )
    return corpus_dir


def _local_hash_embeddings_np(texts: Sequence[str], dim: int) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("Preembedded Ray cache requires numpy.") from exc

    byte_lut = (np.arange(256, dtype=np.float32) / np.float32(127.5)) - np.float32(1.0)
    embeddings = np.empty((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        arr = byte_lut[np.frombuffer(digest, dtype=np.uint8)]
        if dim <= arr.size:
            embeddings[i, :] = arr[:dim]
        else:
            repeats = int(math.ceil(dim / arr.size))
            embeddings[i, :] = np.tile(arr, repeats)[:dim]
    return embeddings


def _ensure_shared_preembedded_synthetic_corpus(
    cache_root: str,
    total_nodes: int,
    node_chars: int,
    total_files: int,
    seed: int,
    dim: int,
) -> str:
    corpus_dir = os.path.join(
        cache_root,
        f"nodes_{total_nodes}__files_{total_files}__chars_{node_chars}__seed_{seed}__preembedded_npz_dim_{dim}",
    )
    manifest = os.path.join(corpus_dir, "manifest.json")
    if os.path.exists(manifest):
        return corpus_dir

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("Preembedded Ray cache requires numpy.") from exc

    os.makedirs(corpus_dir, exist_ok=True)
    for fi in range(total_files):
        chunk_count = _nodes_in_file(fi, total_nodes, total_files)
        start_idx = _nodes_before_file(fi, total_nodes, total_files)
        texts = [_rand_text(node_chars, seed + start_idx + offset) for offset in range(chunk_count)]
        ids = np.asarray([f"doc_{fi:04d}.txt-0-{chunk_i}" for chunk_i in range(chunk_count)], dtype=str)
        embeddings = _local_hash_embeddings_np(texts, dim)
        np.savez_compressed(
            os.path.join(corpus_dir, f"doc_{fi:04d}.npz"),
            ids=ids,
            embeddings=embeddings,
        )
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_nodes": total_nodes,
                "total_files": total_files,
                "node_chars": node_chars,
                "seed": seed,
                "dim": dim,
                "format": "preembedded_npz",
            },
            f,
            indent=2,
        )
    return corpus_dir


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


def _embed_batch_with_embedder_sync(
    batch: Sequence[Tuple[str, str, Dict[str, Any]]],
    embedder: Any,
) -> List[Tuple[str, str, Dict[str, Any], List[float]]]:
    if not batch:
        return []
    texts = [text for _, text, _ in batch]
    embs = embedder.embed_batch_sync(texts)
    out: List[Tuple[str, str, Dict[str, Any], List[float]]] = []
    for (node_id, text, meta), emb in zip(batch, embs):
        out.append((node_id, text, meta, emb))
    return out


def _upsert_batch_sync(
    sink_backend: str,
    persist_dir: str | None,
    collection_name: str,
    dim: int,
    batch: Sequence[Tuple[str, str, Dict[str, Any], List[float]]],
) -> int:
    if not batch:
        return 0
    collection = get_sink_collection(sink_backend, persist_dir, collection_name, dim, reset=False)
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
    sink_backend: str,
    dim: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    ray_parallelism: int,
    upsert_workers: int,
    ray_num_cpus: int | None,
    ray_object_store_memory_mb: int,
    ray_input_format: str,
) -> ResultRow:
    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Ray is required for Set6. Install with `pip install ray[data]`.") from exc

    cluster_cpus: int | None = None
    if not ray.is_initialized():
        ray_address = os.environ.get("RAY_ADDRESS", "").strip()
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True, include_dashboard=False)
            cluster_cpus = int(ray.cluster_resources().get("CPU", 0) or 0)
        elif ray_num_cpus is None and ray_object_store_memory_mb <= 0:
            # When an external Ray cluster is already discoverable via the
            # environment, connect with no local resource sizing arguments.
            # Passing num_cpus/object_store_memory in that case makes ray.init()
            # reject the connection.
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            cluster_cpus = int(ray.cluster_resources().get("CPU", 0) or 0)
        else:
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=ray_num_cpus,
                object_store_memory=max(80, ray_object_store_memory_mb) * 1024 * 1024,
            )
            cluster_cpus = ray_num_cpus

    @ray.remote(num_cpus=0)
    class SinkActor:
        def __init__(self, backend: str, persist: str | None, collection_name: str, vec_dim: int):
            if backend == "chroma":
                self.collection = get_chroma_collection(persist, collection_name, reset=True)
            elif backend == "faiss":
                self.collection = FaissSink(collection_name, vec_dim)
            elif backend == "thin-batched":
                self.collection = ThinBatchedSink(collection_name)
            else:
                raise ValueError(f"Unsupported sink backend: {backend}")
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

    source_suffix = ".npz" if ray_input_format == "preembedded" else ".txt"
    file_count = len([name for name in os.listdir(data_dir) if name.endswith(source_suffix)])
    cpu_target = max(1, cluster_cpus or ray_num_cpus or ray_parallelism)
    # Ray was under-utilizing the cluster because the input and embed stages were
    # operating on too few blocks. Oversubscribe blocks relative to CPUs so Ray
    # can keep workers busy and recover from stragglers instead of stalling on a
    # handful of coarse blocks.
    read_blocks = max(
        1,
        min(
            file_count if file_count > 0 else 1,
            max(ray_parallelism * 4, cpu_target * 8),
        ),
    )

    # Step 1/2: drive ingestion from explicit file paths so the source stage can
    # parallelize across many files. `read_binary_files()` was still collapsing
    # the 1M/128w run to only a couple of source tasks.
    tl0 = time.perf_counter()
    include_metadata = sink_backend not in {"faiss", "thin-batched"}
    if ray_input_format == "preembedded" and include_metadata:
        raise ValueError("Preembedded Ray input is only supported with faiss or thin-batched sinks.")
    file_paths = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.endswith(source_suffix)
    ]
    if file_count >= 4096:
        # Large runs need more source parallelism, but one-file-per-task creates
        # too much scheduler overhead. Target a medium-grained source stage with
        # roughly 2x-4x cluster CPUs worth of tasks.
        source_target_tasks = max(
            cpu_target * 2,
            min(len(file_paths), cpu_target * 4),
        )
        source_blocks = max(1, min(len(file_paths), source_target_tasks))
        source_batch_size = max(1, math.ceil(len(file_paths) / source_target_tasks))
    else:
        source_blocks = max(1, min(len(file_paths), max(read_blocks, cpu_target * 4)))
        source_target_tasks = max(32, min(len(file_paths), max(cpu_target * 2, ray_parallelism * 2)))
        source_batch_size = max(1, math.ceil(len(file_paths) / source_target_tasks))
    if ray_input_format == "preembedded":
        source_reader = _ray_read_preembedded_path_batch
        source_fn_kwargs = {}
    elif ray_input_format == "prechunked":
        source_reader = _ray_read_prechunked_path_batch
        source_fn_kwargs = {"include_metadata": include_metadata}
    else:
        source_reader = _ray_read_and_chunk_path_batch
        source_fn_kwargs = {"delimiter": DELIM, "include_metadata": include_metadata}

    ds_nodes = ray.data.from_items(
        [{"path": path} for path in file_paths],
        override_num_blocks=source_blocks,
    ).map_batches(
        source_reader,
        fn_kwargs=source_fn_kwargs,
        batch_format="numpy",
        batch_size=source_batch_size,
    ).materialize()
    tl1 = time.perf_counter()

    # Step 2: chunking is fused above; keep a small transform stage here for
    # subsequent dataset-shape work only.
    tt0 = time.perf_counter()
    manifest_path = os.path.join(data_dir, "manifest.json")
    node_count = 0
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                node_count = int(json.load(f).get("total_nodes", 0) or 0)
        except Exception:
            node_count = 0
    if node_count <= 0:
        node_count = ds_nodes.count()
    embed_blocks = max(1, min(node_count if node_count > 0 else 1, max(read_blocks, ray_parallelism * 4, cpu_target * 8)))
    current_blocks = ds_nodes.num_blocks()
    # Repartition is expensive; only use it on very large runs when the block
    # count is clearly too low. This avoids paying a shuffle tax on the 10M
    # characterization path unless the source stage truly underpartitions it.
    if node_count >= 5_000_000 and current_blocks < max(64, embed_blocks // 4):
        ds_nodes = ds_nodes.repartition(embed_blocks, shuffle=False).materialize()
    tt1 = time.perf_counter()

    # Step 3: build the embedding pipeline lazily. Materializing the full
    # embedded dataset forces every vector block into the Ray object store,
    # which spills heavily at 10M+ chunk scale and turns the strong-scaling
    # run into a long post-embed drain. Stream embedded batches directly into
    # the sink instead.
    te0 = time.perf_counter()
    ray_embed_batch_size = max(1, embed_batch_size)
    ray_iter_batch_size = max(1, upsert_batch_size)
    if sink_backend == "faiss" and node_count >= 5_000_000:
        # Large FAISS runs are dominated by request overhead in the local-hash
        # embedder and by too-frequent sink drains. Increase batch sizes without
        # introducing a shuffle.
        ray_embed_batch_size = max(ray_embed_batch_size, 4096)
        ray_iter_batch_size = max(ray_iter_batch_size, 16384)

    if ray_input_format == "preembedded":
        ds_emb = ds_nodes
    else:
        ds_emb = ds_nodes.map_batches(
            _ray_embed_batch,
            fn_kwargs={
                "dim": embedder.dim,
                "request_overhead_ms": embedder.request_overhead_ms,
                "per_item_ms": embedder.per_item_ms,
                "keep_payload": include_metadata,
            },
            batch_format="numpy",
            batch_size=ray_embed_batch_size,
        )

    # Step 4: sink upsert while draining the embedding pipeline.
    tu0 = time.perf_counter()
    batch_size = ray_iter_batch_size
    if sink_backend == "faiss":
        # The FAISS sink only uses embeddings. Actor RPCs and per-batch JSON
        # decode turn this path into serialization overhead at scale.
        faiss_shards = 1
        if node_count >= 5_000_000:
            faiss_shards = min(4, max(1, cpu_target // 40))
        sinks = [FaissSink(f"bench_set6_local_faiss_s{i}", dim) for i in range(faiss_shards)]
        consumer_drain_s = 0.0
        consumer_lock = threading.Lock()

        def _drain_add(shard_idx: int, ids: Sequence[str], embs: Sequence[Sequence[float]]) -> None:
            nonlocal consumer_drain_s
            td0 = time.perf_counter()
            sinks[shard_idx].add(ids, embs, (), ())
            td1 = time.perf_counter()
            with consumer_lock:
                consumer_drain_s += td1 - td0

        drain_queue_depth = 8 if node_count >= 5_000_000 else 4
        with ThreadPoolExecutor(max_workers=faiss_shards) as drain_pool:
            pending_adds = []
            for batch in ds_emb.iter_batches(batch_size=batch_size, batch_format="numpy"):
                ids = [str(x) for x in batch["id"]]
                if not ids:
                    continue
                embs = batch["embedding"]
                shard_idx = int(hashlib.md5(ids[0].encode("utf-8")).hexdigest(), 16) % faiss_shards
                pending_adds.append(drain_pool.submit(_drain_add, shard_idx, ids, embs))
                if len(pending_adds) >= drain_queue_depth:
                    pending_adds.pop(0).result()
            for fut in pending_adds:
                fut.result()
    elif sink_backend == "thin-batched":
        sink = ThinBatchedSink("bench_set6_local_thin")
        consumer_drain_s = 0.0
        consumer_lock = threading.Lock()

        def _drain_add(ids: Sequence[str]) -> None:
            nonlocal consumer_drain_s
            td0 = time.perf_counter()
            sink.add(ids, (), (), ())
            td1 = time.perf_counter()
            with consumer_lock:
                consumer_drain_s += td1 - td0

        for batch in ds_emb.iter_batches(batch_size=batch_size, batch_format="numpy"):
            ids = [str(x) for x in batch["id"]]
            if not ids:
                continue
            _drain_add(ids)
    else:
        sink_workers = max(1, upsert_workers)
        actors = [
            SinkActor.remote(sink_backend, persist_dir, f"bench_set6_actor_{i}", dim)
            for i in range(sink_workers)
        ]
        pending = []
        consumer_drain_s = 0.0
        for batch in ds_emb.iter_batches(batch_size=batch_size, batch_format="numpy"):
            ids = [str(x) for x in batch["id"]]
            texts = [str(x) for x in batch["text"]]
            metas = [json.loads(str(x)) for x in batch["metadata_json"]]
            embs = batch["embedding"]
            if not ids:
                continue
            actor_idx = int(hashlib.md5(ids[0].encode("utf-8")).hexdigest(), 16) % sink_workers
            pending.append(actors[actor_idx].upsert.remote(ids, embs, metas, texts))
            if len(pending) >= sink_workers * 4:
                td0 = time.perf_counter()
                ray.get(pending[:sink_workers])
                td1 = time.perf_counter()
                consumer_drain_s += td1 - td0
                pending = pending[sink_workers:]
        if pending:
            td0 = time.perf_counter()
            ray.get(pending)
            td1 = time.perf_counter()
            consumer_drain_s += td1 - td0
        td0 = time.perf_counter()
        _ = ray.get([a.count.remote() for a in actors])
        td1 = time.perf_counter()
        consumer_drain_s += td1 - td0
    tu1 = time.perf_counter()
    te1 = tu1
    if ray_input_format == "preembedded":
        producer_embed_s = 0.0
    else:
        producer_embed_s = max(0.0, (te1 - te0) - consumer_drain_s)

    total_s = time.perf_counter() - t0
    return ResultRow(
        "RayDataScalableRAG",
        node_count,
        tl1 - tl0,
        tt1 - tt0,
        producer_embed_s,
        consumer_drain_s,
        total_s,
        embed_plus_drain_s=te1 - te0,
        consumer_drain_s=consumer_drain_s,
    )


def run_set7_dask_data(
    data_dir: str,
    persist_dir: str | None,
    embedder: FakeEmbedder,
    sink_backend: str,
    dim: int,
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
        get_sink_collection(sink_backend, persist_dir, name, dim, reset=True)

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
        upsert_tasks.append(dask.delayed(_upsert_batch_sync)(sink_backend, persist_dir, shard_name, dim, batch))
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
    sink_backend: str,
    dim: int,
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
        get_sink_collection(sink_backend, persist_dir, name, dim, reset=True)

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
            lambda item: _upsert_batch_sync(sink_backend, persist_dir, item[0], dim, item[1]),
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


async def run_set9_higress_data(
    data_dir: str,
    persist_dir: str | None,
    embedder: FakeEmbedder,
    sink_backend: str,
    dim: int,
    embed_batch_size: int,
    upsert_batch_size: int,
    higress_workers: int,
    upsert_workers: int,
    set45_upsert_shards: int,
) -> ResultRow:
    workers = max(1, higress_workers)
    shards = max(1, set45_upsert_shards)
    collections = init_sink_collections(sink_backend, persist_dir, "bench_set9_higress", shards, dim)

    t0 = time.perf_counter()
    node_count, load_s, transform_s, embed_s, upsert_s = await run_direct_batched_ingest(
        data_dir=data_dir,
        embedder=embedder,
        collections=collections,
        io_workers=workers,
        embed_batch_size=max(1, embed_batch_size),
        upsert_batch_size=max(1, upsert_batch_size),
        embed_num_workers=workers,
        upsert_num_workers=max(1, upsert_workers),
    )
    total_s = time.perf_counter() - t0
    return ResultRow(
        "HigressRAG",
        node_count,
        load_s,
        transform_s,
        embed_s,
        upsert_s,
        total_s,
    )


# -------------------------
# Pretty table
# -------------------------

def print_table(rows: List[ResultRow], baseline_label: str | None = None) -> None:
    if not rows:
        return

    if len(rows) == 1 and rows[0].config == "AAFLOW":
        print("\nNote: Set5 AAFLOW stage columns are pipeline-segment timings in the streaming path.\n")

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
        row_data = [
            r.config,
            str(r.nodes),
            f"{r.load_s:.3f}",
            f"{r.transform_s:.3f}",
            f"{r.embed_s:.3f}",
            f"{r.upsert_s:.3f}",
            f"{r.total_s:.3f}",
            delta,
        ]
        data.append(row_data)

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
    p = argparse.ArgumentParser(description="Benchmark ingestion configs Set1..Set9.")
    p.add_argument("--nodes", type=int, default=1024, help="Exact number of nodes to create.")
    p.add_argument("--node-chars", type=int, default=900, help="Characters per node (controls text size).")
    p.add_argument("--files", type=int, default=200, help="Number of files for SimpleDirectoryReader to load.")
    p.add_argument(
        "--chunks-per-file",
        type=int,
        default=0,
        help="Optional target chunks per file for synthetic corpus generation (0 = derive from --files).",
    )
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Reader/pipeline parallelism
    p.add_argument("--reader-workers", type=int, default=8, help="Set2 reader parallel workers.")
    p.add_argument("--pipeline-workers", type=int, default=8, help="Set3 pipeline multiprocessing workers.")

    # Async concurrency
    p.add_argument("--async-workers", type=int, default=64, help="Set4/5 async concurrency for transforms and embeddings.")
    p.add_argument("--set5-load-workers", type=int, default=0, help="Optional dedicated Set5 load workers (0 = auto).")
    p.add_argument("--set5-transform-workers", type=int, default=0, help="Optional dedicated Set5 transform workers (0 = auto).")
    p.add_argument(
        "--set5-sink-backend",
        type=str,
        default="chroma",
        choices=["chroma", "thin-batched", "faiss"],
        help="Set5 upsert sink backend.",
    )
    p.add_argument(
        "--sink-backend",
        type=str,
        default="chroma",
        choices=["chroma", "thin-batched", "faiss"],
        help="Sink backend for Set4 and Sets6-9.",
    )
    p.add_argument(
        "--no-cap-async-workers",
        action="store_true",
        help="Do not cap async workers to CPU count (use requested value).",
    )
    p.add_argument("--only-async", action="store_true", help="Run only Set4/Set5 (skip Set1-Set3).")
    p.add_argument(
        "--agentic-order",
        choices=["early", "last", "alternate"],
        default="early",
        help=(
            "Execution order for AAFLOW relative to the other sets. "
            "'alternate' flips AAFLOW after Higress on every other worker bucket."
        ),
    )

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
        "--set5-embed-workers",
        type=int,
        default=0,
        help="Optional dedicated Set5 embed worker count (0 = use effective async workers).",
    )
    p.add_argument(
        "--set5-upsert-workers",
        type=int,
        default=0,
        help="Optional dedicated Set5 upsert worker count (0 = use effective upsert workers).",
    )
    p.add_argument(
        "--set5-upsert-timeout-ms",
        type=float,
        default=2.0,
        help="Timeout-based coalescing flush for Set5 upserts in milliseconds.",
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
        "--set5-upsert-shards",
        type=int,
        default=0,
        help="Optional dedicated shard count for Set5 only (0 = use --set45-upsert-shards).",
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
    p.add_argument(
        "--set5-upsert-coalesce-multiplier",
        type=int,
        default=8,
        help="Maximum Set5 upsert coalescing multiplier relative to --set5-upsert-batch.",
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
    p.add_argument("--run-higress-set9", action="store_true", help="Run Set9 Higress RAG ingestion experiment.")
    p.add_argument("--only-ray", action="store_true", help="Run only Set6 Ray Data experiment.")
    p.add_argument("--only-dask", action="store_true", help="Run only Set7 Dask experiment.")
    p.add_argument("--only-bsp", action="store_true", help="Run only Set8 BSP experiment.")
    p.add_argument("--only-higress", action="store_true", help="Run only Set9 Higress RAG experiment.")
    p.add_argument("--only-agentic", action="store_true", help="Run only Set5 AAFLOW experiment.")
    p.add_argument("--ray-num-cpus", type=int, default=0, help="Optional CPUs for ray.init (0 = Ray default).")
    p.add_argument(
        "--ray-input-format",
        choices=["raw", "prechunked", "preembedded"],
        default="raw",
        help="Ray source input format. `prechunked` uses cached one-chunk-per-line files and `preembedded` uses cached ids+embeddings for Ray-only runs.",
    )
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
    p.add_argument(
        "--higress-workers",
        type=int,
        default=0,
        help="Optional worker count for Set9 Higress experiment (0 = use effective async workers).",
    )

    # Fake embedder latency model
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    p.add_argument("--request-overhead-ms", type=float, default=60.0, help="Fixed overhead per embedding request (ms).")
    p.add_argument("--per-item-ms", type=float, default=1.2, help="Per-item cost inside embedding request (ms).")
    p.add_argument(
        "--embedder-backend",
        type=str,
        default="fake",
        choices=["fake", "local-hash"],
        help="Embedding backend for the benchmark.",
    )

    # Chroma
    p.add_argument("--persist-dir", type=str, default="", help="Optional Chroma persistence dir (empty = in-memory).")
    p.add_argument(
        "--shared-corpus-root",
        type=str,
        default="",
        help="Optional shared filesystem root for synthetic corpus caching (used by multi-node Ray runs).",
    )
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.only_ray:
        args.run_ray_set6 = True
    if args.only_dask:
        args.run_dask_set7 = True
    if args.only_bsp:
        args.run_bsp_set8 = True
    if args.only_higress:
        args.run_higress_set9 = True
    if args.only_agentic:
        args.only_async = True
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
    set5_upsert_shards = max(1, args.set5_upsert_shards) if args.set5_upsert_shards > 0 else set45_upsert_shards
    if args.strict_stage_scaling:
        args.no_scale_set5_batches = True
        set45_upsert_shards = effective_async_workers
        set5_upsert_shards = max(set5_upsert_shards, effective_set5_upsert_workers if "effective_set5_upsert_workers" in locals() else effective_async_workers)
        upsert_workers_cap = effective_async_workers
    if args.no_cap_upsert_workers:
        effective_upsert_workers = effective_async_workers
    else:
        if upsert_workers_cap == 0:
            upsert_workers_cap = max(1, cpu_count // 2)
        effective_upsert_workers = min(effective_async_workers, upsert_workers_cap)
    effective_dask_workers = max(1, args.dask_workers) if args.dask_workers > 0 else effective_async_workers
    effective_bsp_workers = max(1, args.bsp_workers) if args.bsp_workers > 0 else effective_async_workers
    effective_higress_workers = max(1, args.higress_workers) if args.higress_workers > 0 else effective_async_workers
    stage_worker_cap = cpu_count
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus_per_task.isdigit():
        stage_worker_cap = max(1, min(stage_worker_cap, int(slurm_cpus_per_task)))
    effective_set5_load_workers = max(1, args.set5_load_workers) if args.set5_load_workers > 0 else min(effective_async_workers, stage_worker_cap)
    effective_set5_transform_workers = max(1, args.set5_transform_workers) if args.set5_transform_workers > 0 else min(effective_async_workers, stage_worker_cap)
    effective_set5_load_workers = min(effective_set5_load_workers, stage_worker_cap)
    effective_set5_transform_workers = min(effective_set5_transform_workers, stage_worker_cap)
    effective_set5_embed_workers = max(1, args.set5_embed_workers) if args.set5_embed_workers > 0 else effective_async_workers
    effective_set5_embed_workers = min(effective_set5_embed_workers, stage_worker_cap)
    effective_set5_upsert_workers = max(1, args.set5_upsert_workers) if args.set5_upsert_workers > 0 else effective_upsert_workers
    effective_set5_upsert_workers = min(effective_set5_upsert_workers, stage_worker_cap)
    if args.strict_stage_scaling:
        set5_upsert_shards = max(set5_upsert_shards, effective_set5_upsert_workers)

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

    def scale_set5_batches_agentic(
        total_items: int,
        workers: int,
        embed_workers: int,
        upsert_workers: int,
        shard_count: int,
    ) -> Tuple[int, int]:
        embed_batch, upsert_batch = scale_set5_batches(total_items, workers)
        if args.no_scale_set5_batches:
            return embed_batch, upsert_batch

        # Agentic Set5 can exploit extra embed parallelism; increase batch size
        # gradually to amortize request overhead without exploding tail latency.
        embed_factor = max(1.0, embed_workers / max(1, workers))
        shard_factor = max(1.0, shard_count / max(1, upsert_workers))
        adaptive_embed = int(math.ceil(embed_batch * math.sqrt(embed_factor)))
        adaptive_upsert = int(math.ceil(upsert_batch * math.sqrt(shard_factor)))

        if total_items > 0:
            adaptive_embed = min(max(1, adaptive_embed), total_items)
            adaptive_upsert = min(max(1, adaptive_upsert), total_items)
        return adaptive_embed, adaptive_upsert

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

    use_shared_corpus = bool(args.shared_corpus_root) and (args.only_ray or args.run_ray_set6)
    corpus_prep_s = 0.0
    with ExitStack() as stack:
        if use_shared_corpus:
            os.makedirs(args.shared_corpus_root, exist_ok=True)
            prep_t0 = time.perf_counter()
            data_dir = _ensure_shared_synthetic_corpus(
                cache_root=args.shared_corpus_root,
                total_nodes=args.nodes,
                node_chars=args.node_chars,
                total_files=args.files,
                seed=args.seed,
            )
            corpus_prep_s = time.perf_counter() - prep_t0
        else:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            data_dir = os.path.join(tmp, "data")
            write_synthetic_corpus(
                data_dir=data_dir,
                nodes=args.nodes,
                node_chars=args.node_chars,
                num_files=args.files,
                seed=args.seed,
                chunks_per_file=args.chunks_per_file,
            )
        ray_data_dir = data_dir
        if use_shared_corpus and args.run_ray_set6:
            if args.ray_input_format == "prechunked":
                prep_t0 = time.perf_counter()
                ray_data_dir = _ensure_shared_prechunked_synthetic_corpus(
                    cache_root=args.shared_corpus_root,
                    total_nodes=args.nodes,
                    node_chars=args.node_chars,
                    total_files=args.files,
                    seed=args.seed,
                )
                corpus_prep_s += time.perf_counter() - prep_t0
            elif args.ray_input_format == "preembedded":
                prep_t0 = time.perf_counter()
                ray_data_dir = _ensure_shared_preembedded_synthetic_corpus(
                    cache_root=args.shared_corpus_root,
                    total_nodes=args.nodes,
                    node_chars=args.node_chars,
                    total_files=args.files,
                    seed=args.seed,
                    dim=args.dim,
                )
                corpus_prep_s += time.perf_counter() - prep_t0

        if args.embedder_backend == "local-hash":
            embedder = LocalHashEmbedder(dim=args.dim)
        else:
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

                collections = init_sink_collections(
                    args.set5_sink_backend,
                    persist_dir,
                    f"bench_set5_w{workers}",
                    set5_upsert_shards,
                    args.dim,
                )
                t0 = time.perf_counter()
                embed_batch, upsert_batch = scale_set5_batches_agentic(
                    args.nodes,
                    workers,
                    effective_set5_embed_workers,
                    effective_set5_upsert_workers,
                    set5_upsert_shards,
                )
                if args.set5_sink_backend == "faiss":
                    set5_nodes, load_s, transform_s, embed_s, upsert_s = await run_set5_thin_faiss(
                        data_dir=data_dir,
                        embedder=embedder,
                        collections=collections,
                        load_workers=effective_set5_load_workers,
                        transform_workers=effective_set5_transform_workers,
                        embed_batch_size=embed_batch,
                        upsert_batch_size=upsert_batch,
                        embed_num_workers=effective_set5_embed_workers,
                    )
                else:
                    set5_nodes, load_s, transform_s, embed_s, upsert_s = await embed_and_upsert_streaming(
                        data_dir=data_dir,
                        embedder=embedder,
                        collections=collections,
                        load_workers=effective_set5_load_workers,
                        transform_workers=effective_set5_transform_workers,
                        embed_batch_size=embed_batch,
                        upsert_batch_size=upsert_batch,
                        embed_num_workers=effective_set5_embed_workers,
                        upsert_num_workers=effective_set5_upsert_workers,
                        upsert_coalesce_timeout_s=args.set5_upsert_timeout_ms / 1000.0,
                        upsert_coalesce_multiplier=args.set5_upsert_coalesce_multiplier,
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
                plt.plot(xs, set5, marker="o", label="AAFLOW (Set5)")
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

        only_scalable = args.only_ray or args.only_dask or args.only_bsp or args.only_higress

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
 
        async def build_agentic_row() -> ResultRow:
            collections = init_sink_collections(
                args.set5_sink_backend,
                persist_dir,
                "bench_set5",
                set5_upsert_shards,
                args.dim,
            )
            t0 = time.perf_counter()
            embed_batch, upsert_batch = scale_set5_batches_agentic(
                args.nodes,
                effective_async_workers,
                effective_set5_embed_workers,
                effective_set5_upsert_workers,
                set5_upsert_shards,
            )
            if args.set5_sink_backend == "faiss":
                set5_nodes, load_s, transform_s, embed_s, upsert_s = await run_set5_thin_faiss(
                    data_dir=data_dir,
                    embedder=embedder,
                    collections=collections,
                    load_workers=effective_set5_load_workers,
                    transform_workers=effective_set5_transform_workers,
                    embed_batch_size=embed_batch,
                    upsert_batch_size=upsert_batch,
                    embed_num_workers=effective_set5_embed_workers,
                )
            else:
                set5_nodes, load_s, transform_s, embed_s, upsert_s = await embed_and_upsert_streaming(
                    data_dir=data_dir,
                    embedder=embedder,
                    collections=collections,
                    load_workers=effective_set5_load_workers,
                    transform_workers=effective_set5_transform_workers,
                    embed_batch_size=embed_batch,
                    upsert_batch_size=upsert_batch,
                    embed_num_workers=effective_set5_embed_workers,
                    upsert_num_workers=effective_set5_upsert_workers,
                    upsert_coalesce_timeout_s=args.set5_upsert_timeout_ms / 1000.0,
                    upsert_coalesce_multiplier=args.set5_upsert_coalesce_multiplier,
                )
            total_s = time.perf_counter() - t0
            return ResultRow("AAFLOW", set5_nodes, load_s, transform_s, embed_s, upsert_s, total_s)

        agentic_run_last = False
        if args.agentic_order == "last":
            agentic_run_last = True
        elif args.agentic_order == "alternate" and args.run_higress_set9:
            agentic_run_last = ((effective_async_workers // 4) % 2 == 0)

        if not only_scalable:
            # ---------------- Set 4 ----------------
            if not args.only_agentic:
                t0 = time.perf_counter()
                docs, load_s = load_docs_sync(data_dir)
                nodes_async, transform_s = await transform_async(docs, num_workers=effective_async_workers)
                collections = init_sink_collections(args.sink_backend, persist_dir, "bench_set4", set45_upsert_shards, args.dim)
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
            if args.only_agentic or not agentic_run_last:
                rows.append(await build_agentic_row())

        if args.run_ray_set6:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_async_workers)
            ray_row = run_set6_ray_data(
                data_dir=ray_data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                ray_parallelism=effective_async_workers,
                upsert_workers=effective_upsert_workers,
                ray_num_cpus=args.ray_num_cpus if args.ray_num_cpus > 0 else None,
                ray_object_store_memory_mb=args.ray_object_store_memory_mb,
                ray_input_format=args.ray_input_format,
            )
            rows.append(ray_row)
            # Ray keeps its runtime alive after the benchmark unless it is shut down
            # explicitly. That can interfere with later sets in the same process.
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass

        if args.run_dask_set7:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_dask_workers)
            dask_row = run_set7_dask_data(
                data_dir=data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.sink_backend,
                dim=args.dim,
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
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                bsp_workers=effective_bsp_workers,
                upsert_workers=effective_upsert_workers,
                set45_upsert_shards=set45_upsert_shards,
            )
            rows.append(bsp_row)

        if args.run_higress_set9:
            embed_batch, upsert_batch = scale_set5_batches(args.nodes, effective_higress_workers)
            higress_row = await run_set9_higress_data(
                data_dir=data_dir,
                persist_dir=persist_dir,
                embedder=embedder,
                sink_backend=args.sink_backend,
                dim=args.dim,
                embed_batch_size=embed_batch,
                upsert_batch_size=upsert_batch,
                higress_workers=effective_higress_workers,
                upsert_workers=effective_upsert_workers,
                set45_upsert_shards=set45_upsert_shards,
            )
            rows.append(higress_row)

        if not args.only_agentic and agentic_run_last:
            rows.append(await build_agentic_row())

        baseline_label = "AsyncParallelOnly" if args.only_async and not only_scalable else None
        print_table(rows, baseline_label=baseline_label)
        if use_shared_corpus:
            print(f"CorpusPrep(s) [excluded from benchmark stages]: {corpus_prep_s:.3f}\n")
            run_dir = getattr(args, "run_dir", "") or ""
            if run_dir:
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "corpus_prep.txt"), "w", encoding="utf-8") as f:
                    f.write(f"corpus_prep_s={corpus_prep_s:.6f}\n")
                    f.write(f"data_dir={data_dir}\n")

        # Small highlight: Set5 vs Set4
        r4 = next((r for r in rows if r.config == "AsyncParallelOnly"), None)
        r5 = next((r for r in rows if r.config == "AAFLOW"), None)
        rh = next((r for r in rows if r.config == "HigressRAG"), None)
        if r4 and r5:
            print(
                "AAFLOW vs AsyncParallelOnly total improvement: "
                f"{pct_faster(r4.total_s, r5.total_s):.1f}% faster with {effective_async_workers} Workers\n"
            )
        if rh and r5:
            print(
                "AAFLOW vs HigressRAG total improvement: "
                f"{pct_faster(rh.total_s, r5.total_s):.1f}% faster with {effective_async_workers} Workers\n"
            )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
