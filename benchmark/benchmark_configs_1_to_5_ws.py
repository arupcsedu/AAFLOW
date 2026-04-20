#!/usr/bin/env python3
"""
Benchmark LlamaIndex ingestion configs (Set 1..5) with weak and strong scaling.

Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 06/01/2026 (weak/strong scaling + clearer CLI names + CSV/LaTeX emitters)

Implements:

  Set 1: Default
    - Sync load (SimpleDirectoryReader.load_data())
    - Sync pipeline (IngestionPipeline.run()) sequential transforms
    - Embed + upsert sequential, no batching

  Set 2: Reader Parallel
    - Parallel reader load (SimpleDirectoryReader.load_data(num_workers=...))
    - Sync pipeline sequential transforms
    - Embed + upsert sequential, no batching

  Set 3: Pipeline Parallel Sync
    - Sync reader
    - Pipeline parallel transforms (multiprocessing, num_workers > 1)
    - Embed + upsert sequential, no batching

  Set 4: Async Only
    - Sync reader
    - Async transforms (IngestionPipeline.arun(..., num_workers=...))
    - Async embedding with concurrency (no batching, batch size = 1)

  Set 5: Async + Batching (Agentic DRC-like)
    - Sync reader
    - Async transforms
    - Async embedding with concurrency + micro-batching + batched upserts

Scaling modes
-------------

We support two modes:

  --mode weak   (default)
    Weak scaling: per-node workload fixed, global workload grows with nnodes.

    Interpret flags as per-node shape:
      --chunks-per-node = number of text chunks per logical cluster node
      --files-per-node  = number of files per logical cluster node
      --nnodes-list     = comma-separated logical node counts (e.g., 1,2,4,8)

    For each nnodes in --nnodes-list, we generate:
      total_chunks = chunks-per-node * nnodes
      total_files  = files-per-node  * nnodes

  --mode strong
    Strong scaling: global workload fixed, nnodes changes but the dataset stays constant.

    Interpret flags as *global* totals:
      --chunks-per-node = TOTAL number of text chunks across the dataset
      --files-per-node  = TOTAL number of files across the dataset

    For each nnodes in --nnodes-list, we reuse:
      total_chunks = chunks-per-node
      total_files  = files-per-node

    nnodes then just labels the logical cluster size; you can change concurrency
    parameters manually (reader-workers, async-workers, etc.) if you want them
    to scale with nnodes.

CSV and LaTeX emitters
----------------------

After all runs, we emit:

  --csv-out   (default: benchmark_results.csv)
  --latex-out (default: benchmark_results.tex)

Both include columns indexed by `mode` and `nnodes`.

python benchmark_configs_1_to_5_ws.py \
  --mode weak \
  --chunks-per-node 4096 \
  --files-per-node 100 \
  --nnodes-list 32,64

python benchmark_configs_1_to_5_ws.py \
  --mode strong \
  --chunks-per-node 32768 \
  --files-per-node 800 \
  --nnodes-list 1,2,4,8

"""

import argparse
import asyncio
import hashlib
import os
import random
import string
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

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
        fname = os.path.join(data_dir, f"file_{fi:05d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)


# ------------------------------
# Simple delimiter-based splitter
# ------------------------------


class DelimiterNodeSplitter(TransformComponent):
    """
    Custom splitter compatible with IngestionPipeline.

    Important: IDs must be globally unique across the whole corpus.
    We derive node IDs from the parent document's id_ (if present),
    or from a hash of the text as a fallback.
    """

    delimiter: str = DELIM  # pydantic field

    def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> List[TextNode]:
        out: List[TextNode] = []

        for obj in nodes:
            # 1. Extract text
            text = getattr(obj, "text", None)
            if text is None and hasattr(obj, "get_text"):
                text = obj.get_text()
            if not text:
                continue

            # 2. Base ID = original document id if available, otherwise a hash
            base_id = getattr(obj, "id_", None)
            if not base_id:
                # deterministic fallback based on content
                base_id = "doc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

            # 3. Split into chunks and give each chunk a unique suffix
            parts = text.split(self.delimiter)
            for j, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                node_id = f"{base_id}_chunk{j}"
                out.append(TextNode(text=part, id_=node_id))

        return out


# -------------------------
# Fake embedding "service"
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
        # Sleep to simulate network+GPU latency
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec: List[float] = []
            # Simple, deterministic mapping from bytes → floats
            for j in range(self.dim):
                b = h[j % len(h)]
                vec.append((b - 128) / 128.0)
            out.append(vec)
        return out


# -------------
# Chroma helper
# -------------


def init_chroma(persist_dir: str | None, collection_name: str) -> Any:
    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.Client()
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    return client.create_collection(collection_name)


# -------------------------
# Result table & formatting
# -------------------------


@dataclass
class ResultRow:
    mode: str
    nnodes: int
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


def print_table(rows: List[ResultRow]) -> None:
    # Simple text table, already scoped to a single (mode, nnodes) in our loop.
    headers = [
        "Config",
        "Mode",
        "nnodes",
        "Nodes",
        "Load (s)",
        "Transform (s)",
        "Embed (s)",
        "Upsert (s)",
        "Total (s)",
    ]
    print("\n" + "-" * 100)
    print(
        "{:<26} {:<6} {:>6} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
            *headers
        )
    )
    print("-" * 100)
    for r in rows:
        print(
            "{:<26} {:<6} {:>6d} {:>8d} {:>10.3f} {:>12.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                r.config,
                r.mode,
                r.nnodes,
                r.nodes,
                r.load_s,
                r.transform_s,
                r.embed_s,
                r.upsert_s,
                r.total_s,
            )
        )
    print("-" * 100 + "\n")


# -------------------------
# CSV & LaTeX emitters
# -------------------------


def write_csv(results: List[ResultRow], mode: str, path: str) -> None:
    if not path:
        return
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "nnodes",
                "config",
                "nodes",
                "load_s",
                "transform_s",
                "embed_s",
                "upsert_s",
                "total_s",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.mode,
                    r.nnodes,
                    r.config,
                    r.nodes,
                    f"{r.load_s:.6f}",
                    f"{r.transform_s:.6f}",
                    f"{r.embed_s:.6f}",
                    f"{r.upsert_s:.6f}",
                    f"{r.total_s:.6f}",
                ]
            )


def write_latex(results: List[ResultRow], mode: str, path: str) -> None:
    if not path:
        return

    # Sort by nnodes then config for nicer LaTeX output
    results_sorted = sorted(results, key=lambda r: (r.nnodes, r.config))

    lines: List[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lllrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Mode & nnodes & Config & Nodes & Load (s) & Transform (s) & Embed (s) & Upsert (s) & Total (s) \\"
    )
    lines.append(r"\hline")

    for r in results_sorted:
        cfg = r.config.replace("_", r"\_")
        lines.append(
            rf"{r.mode} & {r.nnodes} & {cfg} & {r.nodes} & "
            rf"{r.load_s:.3f} & {r.transform_s:.3f} & {r.embed_s:.3f} & {r.upsert_s:.3f} & {r.total_s:.3f} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Benchmark results ({mode} scaling) by configuration and nnodes.}}"
    )
    lines.append(r"\label{tab:benchmark_results}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# -------------------------
# Config implementations
# -------------------------


def load_docs_sync(data_dir: str) -> Tuple[List[Any], float]:
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(data_dir).load_data()
    t1 = time.perf_counter()
    return docs, (t1 - t0)


def load_docs_parallel(data_dir: str, num_workers: int) -> Tuple[List[Any], float]:
    # SimpleDirectoryReader parallel load example uses load_data(num_workers=...)
    t0 = time.perf_counter()
    reader = SimpleDirectoryReader(data_dir)
    docs = reader.load_data(num_workers=num_workers)
    t1 = time.perf_counter()
    return docs, (t1 - t0)


def transform_sync(
    docs: List[Any],
    num_workers: int | None,
) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    if num_workers is None or num_workers <= 1:
        nodes = pipeline.run(documents=docs)
    else:
        # Synchronous multi-process pipeline
        nodes = pipeline.run(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)


async def transform_async(docs: List[Any], num_workers: int) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # Async pipeline; num_workers controls concurrency
    nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)


async def embed_and_upsert(
    nodes: List[TextNode],
    embedder: FakeEmbedder,
    collection: Any,
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
) -> Tuple[float, float]:
    """
    Generic embedding + upsert routine.

    - embed_batch_size   = logical micro-batch size to FakeEmbedder
    - upsert_batch_size  = how many embeddings per Chroma upsert call
    - embed_num_workers  = concurrency level for embedding
    """
    texts = [n.get_text() for n in nodes]

    # ---------- Embedding ----------
    async def worker_embed(
        batch_indices: List[int],
        result_vectors: List[List[float]],
    ) -> None:
        batch_texts = [texts[i] for i in batch_indices]
        vecs = await embedder.embed_batch(batch_texts)
        for idx, v in zip(batch_indices, vecs):
            result_vectors[idx] = v

    n = len(texts)
    embed_vectors: List[List[float]] = [[0.0] * embedder.dim for _ in range(n)]
    batches: List[List[int]] = [
        list(range(i, min(i + embed_batch_size, n)))
        for i in range(0, n, embed_batch_size)
    ]

    t0_embed = time.perf_counter()
    if embed_num_workers <= 1:
        # Purely sequential
        for b in batches:
            await worker_embed(b, embed_vectors)
    else:
        sem = asyncio.Semaphore(embed_num_workers)

        async def gated_worker(b: List[int]) -> None:
            async with sem:
                await worker_embed(b, embed_vectors)

        await asyncio.gather(*(gated_worker(b) for b in batches))
    t1_embed = time.perf_counter()
    embed_s = t1_embed - t0_embed

    # ---------- Upsert ----------
    t0_upsert = time.perf_counter()
    ids: List[str] = [n.id_ for n in nodes]

    for i in range(0, n, upsert_batch_size):
        j = min(i + upsert_batch_size, n)
        collection.upsert(
            ids=ids[i:j],
            embeddings=embed_vectors[i:j],
            metadatas=[{"idx": k} for k in range(i, j)],
            documents=[texts[k] for k in range(i, j)],
        )
    t1_upsert = time.perf_counter()
    upsert_s = t1_upsert - t0_upsert

    return embed_s, upsert_s


# -------------------------
# Argument parsing (renamed)
# -------------------------


def parse_nnodes_list(s: str) -> List[int]:
    """
    Parse a comma-separated list of logical cluster node counts.

      "1"        -> [1]
      "1,2,4,8"  -> [1, 2, 4, 8]
    """
    if not s:
        return [1]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise ValueError(f"Invalid --nnodes-list value {part!r}; expected integers.")
        if v <= 0:
            raise ValueError("--nnodes-list values must be positive.")
        out.append(v)
    if not out:
        out.append(1)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark LlamaIndex ingestion configs Set1..Set5 with strong and weak scaling.\n\n"
            "Weak scaling (mode=weak):\n"
            "  --chunks-per-node = number of text chunks per logical cluster node\n"
            "  --files-per-node  = number of files per logical cluster node\n"
            "  --nnodes-list     = comma-separated logical node counts (e.g., 1,2,4,8)\n"
            "  total_chunks = chunks-per-node * nnodes, total_files = files-per-node * nnodes\n\n"
            "Strong scaling (mode=strong):\n"
            "  --chunks-per-node = TOTAL number of text chunks (global dataset)\n"
            "  --files-per-node  = TOTAL number of files (global dataset)\n"
            "  total_chunks and total_files are independent of nnodes.\n"
        )
    )

    p.add_argument(
        "--mode",
        choices=["weak", "strong"],
        default="weak",
        help="Scaling mode: weak (default) or strong.",
    )

    # Shape parameters
    p.add_argument(
        "--chunks-per-node",
        type=int,
        default=4096,
        help=(
            "Weak mode: chunks per logical node. "
            "Strong mode: TOTAL chunks for the dataset."
        ),
    )
    p.add_argument(
        "--node-chars",
        type=int,
        default=900,
        help="Characters per chunk (controls text size).",
    )
    p.add_argument(
        "--files-per-node",
        type=int,
        default=100,
        help=(
            "Weak mode: files per logical node. "
            "Strong mode: TOTAL files for the dataset."
        ),
    )
    p.add_argument(
        "--nnodes-list",
        type=str,
        default="1",
        help=(
            "Comma-separated list of logical cluster node counts "
            "for (weak or strong) scaling, e.g., 1,2,4,8."
        ),
    )
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Reader/pipeline parallelism
    p.add_argument("--reader-workers", type=int, default=8, help="Set2 reader parallel workers.")
    p.add_argument("--pipeline-workers", type=int, default=8, help="Set3 pipeline multiprocessing workers.")

    # Async concurrency
    p.add_argument(
        "--async-workers",
        type=int,
        default=32,
        help="Set4/5 async concurrency for transforms and embeddings.",
    )

    # Set5 batching
    p.add_argument("--set5-embed-batch", type=int, default=64, help="Set5 embedding batch size.")
    p.add_argument("--set5-upsert-batch", type=int, default=256, help="Set5 upsert batch size.")

    # Fake embedder latency model
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    p.add_argument(
        "--request-overhead-ms",
        type=float,
        default=60.0,
        help="Fixed overhead per embedding request (ms).",
    )
    p.add_argument(
        "--per-item-ms",
        type=float,
        default=1.2,
        help="Per-item cost inside embedding request (ms).",
    )

    # Chroma
    p.add_argument(
        "--persist-dir",
        type=str,
        default="",
        help="Optional Chroma persistence dir (empty = in-memory).",
    )

    # Emitters
    p.add_argument(
        "--csv-out",
        type=str,
        default="benchmark_results.csv",
        help="Path to write CSV summary (empty to disable).",
    )
    p.add_argument(
        "--latex-out",
        type=str,
        default="benchmark_results.tex",
        help="Path to write LaTeX tabular summary (empty to disable).",
    )

    return p.parse_args()


# -------------------------
# Main scaling loop
# -------------------------


async def main_async() -> None:
    args = parse_args()
    persist_dir = args.persist_dir.strip() or None
    mode = args.mode

    nnodes_list = parse_nnodes_list(args.nnodes_list)
    chunks_param = args.chunks_per_node
    files_param = args.files_per_node

    if mode == "weak":
        print(
            f"[mode=weak] Per-node workload: chunks_per_node={chunks_param}, "
            f"files_per_node={files_param}, node_chars={args.node_chars}"
        )
    else:
        print(
            f"[mode=strong] Global workload: total_chunks={chunks_param}, "
            f"total_files={files_param}, node_chars={args.node_chars}"
        )
    print(f"nnodes_list = {nnodes_list}")

    all_results: List[ResultRow] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        for nnodes in nnodes_list:
            if mode == "weak":
                total_chunks = chunks_param * nnodes
                total_files = files_param * nnodes
            else:  # strong
                total_chunks = chunks_param
                total_files = files_param

            data_dir = os.path.join(tmp_root, f"data_mode_{mode}_nnodes_{nnodes}")
            os.makedirs(data_dir, exist_ok=True)

            print(
                f"\n==== {mode.capitalize()}-scaling run: nnodes={nnodes}, "
                f"total_chunks={total_chunks}, total_files={total_files} ===="
            )

            # Generate synthetic corpus for this scale
            write_synthetic_corpus(
                data_dir=data_dir,
                nodes=total_chunks,
                node_chars=args.node_chars,
                num_files=total_files,
                seed=args.seed,
            )

            # Embedding latency model
            embedder = FakeEmbedder(
                dim=args.dim,
                request_overhead_ms=args.request_overhead_ms,
                per_item_ms=args.per_item_ms,
            )

            rows: List[ResultRow] = []

            # ---------------- Set 1: Default ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes, transform_s = transform_sync(docs, num_workers=None)
            collection = init_chroma(persist_dir, f"bench_set1_{mode}_n{nnodes}")
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collection=collection,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,  # sequential embed
            )
            total_s = time.perf_counter() - t0
            rows.append(
                ResultRow(mode, nnodes, "NoParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s)
            )

            # ---------------- Set 2: Reader Parallel ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_parallel(data_dir, num_workers=args.reader_workers)
            nodes, transform_s = transform_sync(docs, num_workers=None)
            collection = init_chroma(persist_dir, f"bench_set2_{mode}_n{nnodes}")
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collection=collection,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,
            )
            total_s = time.perf_counter() - t0
            rows.append(
                ResultRow(
                    mode,
                    nnodes,
                    "ReaderParallel",
                    len(nodes),
                    load_s,
                    transform_s,
                    embed_s,
                    upsert_s,
                    total_s,
                )
            )

            # ---------------- Set 3: Pipeline Parallel Sync ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes, transform_s = transform_sync(docs, num_workers=args.pipeline_workers)
            collection = init_chroma(persist_dir, f"bench_set3_{mode}_n{nnodes}")
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes,
                embedder=embedder,
                collection=collection,
                embed_batch_size=1,
                upsert_batch_size=1,
                embed_num_workers=1,
            )
            total_s = time.perf_counter() - t0
            rows.append(
                ResultRow(
                    mode,
                    nnodes,
                    "PipelineParallelSync",
                    len(nodes),
                    load_s,
                    transform_s,
                    embed_s,
                    upsert_s,
                    total_s,
                )
            )

            # ---------------- Set 4: Async Only ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes_async, transform_s = await transform_async(docs, num_workers=args.async_workers)
            collection = init_chroma(persist_dir, f"bench_set4_{mode}_n{nnodes}")
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes_async,
                embedder=embedder,
                collection=collection,
                embed_batch_size=1,  # no batching
                upsert_batch_size=1,
                embed_num_workers=args.async_workers,
            )
            total_s = time.perf_counter() - t0
            rows.append(
                ResultRow(
                    mode,
                    nnodes,
                    "AsyncParallelOnly",
                    len(nodes_async),
                    load_s,
                    transform_s,
                    embed_s,
                    upsert_s,
                    total_s,
                )
            )

            # ---------------- Set 5: Async + Batching ----------------
            t0 = time.perf_counter()
            docs, load_s = load_docs_sync(data_dir)
            nodes_async, transform_s = await transform_async(docs, num_workers=args.async_workers)
            collection = init_chroma(persist_dir, f"bench_set5_{mode}_n{nnodes}")
            embed_s, upsert_s = await embed_and_upsert(
                nodes=nodes_async,
                embedder=embedder,
                collection=collection,
                embed_batch_size=max(1, args.set5_embed_batch),
                upsert_batch_size=max(1, args.set5_upsert_batch),
                embed_num_workers=args.async_workers,
            )
            total_s = time.perf_counter() - t0
            rows.append(
                ResultRow(
                    mode,
                    nnodes,
                    "AAFLOW",
                    len(nodes_async),
                    load_s,
                    transform_s,
                    embed_s,
                    upsert_s,
                    total_s,
                )
            )

            # Report for this nnodes
            print_table(rows)

            # Highlight Set5 vs Set4 at this scale
            r4 = next(r for r in rows if r.config == "AsyncParallelOnly")
            r5 = next(r for r in rows if r.config == "AAFLOW")
            print(
                f"AAFLOW vs AsyncParallelOnly total improvement at nnodes={nnodes}: "
                f"{pct_faster(r4.total_s, r5.total_s):.1f}% faster\n"
            )

            all_results.extend(rows)

    # Emit CSV and LaTeX summaries
    if all_results:
        if args.csv_out:
            write_csv(all_results, mode, args.csv_out)
            print(f"Wrote CSV results to {args.csv_out}")
        if args.latex_out:
            write_latex(all_results, mode, args.latex_out)
            print(f"Wrote LaTeX table to {args.latex_out}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
