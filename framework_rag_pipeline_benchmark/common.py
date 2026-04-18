from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

DELIM = "\n\n<<<NODE_SPLIT>>>\n\n"


@dataclass
class BenchmarkConfig:
    benchmark_mode: str
    embedding_backend: str
    generation_backend: str
    vector_backend: str
    generation_cost_mode: str
    embedding_model: str
    generation_model: str
    chroma_path: str | None
    faiss_path: str | None
    data_dir: str
    file_glob: str
    nodes: int
    files: int
    node_chars: int
    chunk_tokens: int
    chunk_overlap: int
    generation_samples: int
    generation_output_tokens: int
    load_workers: int
    transform_workers: int
    async_workers: int
    physical_workers: int | None
    embed_workers: int | None
    upsert_workers: int | None
    embed_dim: int
    embed_batch_size: int
    upsert_batch_size: int
    agentic_queue_size: int
    agentic_upsert_coalesce_target: int
    seed: int
    embed_overhead_ms: float
    embed_per_item_ms: float
    upsert_overhead_ms: float
    upsert_per_item_ms: float
    generate_overhead_ms: float
    generate_ms_per_token: float


@dataclass
class PipelineMetrics:
    framework: str
    runtime_mode: str
    documents_loaded: int
    chunks: int
    generated_prompts: int
    generated_tokens: int
    load_s: float
    transform_s: float
    generation_s: float
    tokens_per_second: float
    embed_s: float
    upsert_s: float
    total_s: float


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return end - self.start


def _rand_text(chars: int, seed: int) -> str:
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "     "
    return "".join(rnd.choice(alphabet) for _ in range(chars)).strip()


def ensure_synthetic_corpus(config: BenchmarkConfig) -> None:
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in data_dir.glob(config.file_glob) if p.is_file())
    if len(existing) >= config.files:
        return

    base = config.nodes // config.files
    rem = config.nodes % config.files
    per_file = [base + (1 if i < rem else 0) for i in range(config.files)]
    idx = 0
    for fi, k in enumerate(per_file):
        parts = []
        for _ in range(k):
            parts.append(_rand_text(config.node_chars, config.seed + idx))
            idx += 1
        (data_dir / f"doc_{fi:04d}.txt").write_text(DELIM.join(parts), encoding="utf-8")


def list_input_files(config: BenchmarkConfig) -> List[Path]:
    paths = sorted(Path(config.data_dir).glob(config.file_glob))
    return [p for p in paths if p.is_file()][: config.files]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str, chunk_tokens: int, chunk_overlap: int) -> List[str]:
    if DELIM in text:
        return [part.strip() for part in text.split(DELIM) if part.strip()]

    words = text.split()
    if not words:
        return []
    chunk_tokens = max(1, chunk_tokens)
    step = max(1, chunk_tokens - max(0, chunk_overlap))
    out = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_tokens]
        if not piece:
            continue
        out.append(" ".join(piece))
        if start + chunk_tokens >= len(words):
            break
    return out


class FakeEmbedder:
    def __init__(self, dim: int, overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.overhead_ms = overhead_ms
        self.per_item_ms = per_item_ms

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        time.sleep((self.overhead_ms + self.per_item_ms * len(texts)) / 1000.0)
        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([((h[i % len(h)] / 127.5) - 1.0) for i in range(self.dim)])
        return out


class TransformersEmbedder:
    def __init__(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.torch = torch

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            output = self.model(**encoded)
        last_hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return pooled.cpu().tolist()


class FakeGenerator:
    def __init__(self, base_latency_ms: float, ms_per_token: float, cost_mode: str = "linear"):
        self.base_latency_ms = base_latency_ms
        self.ms_per_token = ms_per_token
        self.cost_mode = cost_mode

    def generate_batch(self, prompts: Sequence[str], output_tokens: int) -> List[str]:
        if self.cost_mode == "fixed":
            sleep_ms = self.base_latency_ms
        else:
            sleep_ms = self.base_latency_ms + self.ms_per_token * output_tokens * len(prompts)
        time.sleep(sleep_ms / 1000.0)
        out = []
        for prompt in prompts:
            seed = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]
            out.append((seed + " ") * max(1, output_tokens // 2))
        return out


class TransformersGenerator:
    def __init__(self, model_name: str):
        from transformers import pipeline

        self.pipeline = pipeline("text-generation", model=model_name, framework="pt")
        if getattr(self.pipeline.tokenizer, "pad_token_id", None) is None:
            self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
            self.pipeline.model.config.pad_token_id = self.pipeline.tokenizer.eos_token_id

    def generate_batch(self, prompts: Sequence[str], output_tokens: int) -> List[str]:
        if not prompts:
            return []
        outputs = self.pipeline(
            list(prompts),
            max_new_tokens=output_tokens,
            do_sample=False,
            return_full_text=False,
            batch_size=max(1, len(prompts)),
        )
        out: List[str] = []
        for item in outputs:
            if isinstance(item, list):
                out.append(item[0].get("generated_text", ""))
            else:
                out.append(item.get("generated_text", ""))
        return out


class SimpleVectorStore:
    def __init__(self, overhead_ms: float, per_item_ms: float):
        self.overhead_ms = overhead_ms
        self.per_item_ms = per_item_ms
        self.rows: List[Dict[str, Any]] = []

    def upsert_batch(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], documents: Sequence[str]) -> None:
        time.sleep((self.overhead_ms + self.per_item_ms * len(ids)) / 1000.0)
        for row_id, vec, doc in zip(ids, vectors, documents):
            self.rows.append({"id": row_id, "vector": list(vec), "document": doc})


class ChromaVectorStore:
    def __init__(self, path: str | None):
        import chromadb

        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=path)
        else:
            client = chromadb.EphemeralClient()
        name = f"fw_rag_bench_{int(time.time() * 1e6)}"
        self.collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def upsert_batch(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], documents: Sequence[str]) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=list(ids),
            embeddings=[list(v) for v in vectors],
            documents=list(documents),
        )


class FaissVectorStore:
    def __init__(self, dim: int, path: str | None):
        import faiss
        import numpy as np
        import threading

        self.faiss = faiss
        self.np = np
        self.lock = threading.Lock()
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(dim)
        self.rows: List[Dict[str, Any]] = []

    def upsert_batch(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], documents: Sequence[str]) -> None:
        if not ids:
            return
        arr = self.np.asarray(vectors, dtype="float32")
        with self.lock:
            self.index.add(arr)
            for row_id, vec, doc in zip(ids, vectors, documents):
                self.rows.append({"id": row_id, "vector": list(vec), "document": doc})


def batched(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, batch_size)
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _requested_row(row: PipelineMetrics) -> Dict[str, Any]:
    return {
        "framework": row.framework,
        "documents_loaded": row.documents_loaded,
        "generated_tokens": row.generated_tokens,
        "load_s": row.load_s,
        "transform_s": row.transform_s,
        "tokens_per_second": row.tokens_per_second,
        "embed_s": row.embed_s,
        "upsert_s": row.upsert_s,
        "total_s": row.total_s,
    }


def median_metrics(rows: Sequence[PipelineMetrics]) -> PipelineMetrics:
    if not rows:
        raise ValueError("median_metrics requires at least one row")
    first = rows[0]

    def med(attr: str) -> float:
        return float(statistics.median(getattr(row, attr) for row in rows))

    def med_int(attr: str) -> int:
        return int(round(statistics.median(getattr(row, attr) for row in rows)))

    return PipelineMetrics(
        framework=first.framework,
        runtime_mode=first.runtime_mode,
        documents_loaded=med_int("documents_loaded"),
        chunks=med_int("chunks"),
        generated_prompts=med_int("generated_prompts"),
        generated_tokens=med_int("generated_tokens"),
        load_s=med("load_s"),
        transform_s=med("transform_s"),
        generation_s=med("generation_s"),
        tokens_per_second=med("tokens_per_second"),
        embed_s=med("embed_s"),
        upsert_s=med("upsert_s"),
        total_s=med("total_s"),
    )


def write_metrics(output_dir: Path, rows: Sequence[PipelineMetrics], full_rows: Sequence[Dict[str, Any]] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    full_path = output_dir / "full_summary.csv"
    json_path = output_dir / "summary.json"

    summary_fields = [
        "framework",
        "documents_loaded",
        "generated_tokens",
        "load_s",
        "transform_s",
        "tokens_per_second",
        "embed_s",
        "upsert_s",
        "total_s",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_requested_row(row))

    if full_rows is None:
        full_rows = [asdict(row) for row in rows]

    full_fields = list(full_rows[0].keys()) if full_rows else []
    with full_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_fields)
        writer.writeheader()
        for row in full_rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_rows, f, indent=2)


def build_embedder(config: BenchmarkConfig):
    if config.embedding_backend == "transformers":
        return TransformersEmbedder(config.embedding_model)
    return FakeEmbedder(config.embed_dim, config.embed_overhead_ms, config.embed_per_item_ms)


def build_generator(config: BenchmarkConfig):
    if config.generation_backend == "transformers":
        return TransformersGenerator(config.generation_model)
    return FakeGenerator(config.generate_overhead_ms, config.generate_ms_per_token, config.generation_cost_mode)


def build_vector_store(config: BenchmarkConfig):
    if config.vector_backend == "chroma":
        return ChromaVectorStore(config.chroma_path)
    if config.vector_backend == "faiss":
        return FaissVectorStore(config.embed_dim, config.faiss_path)
    return SimpleVectorStore(config.upsert_overhead_ms, config.upsert_per_item_ms)
