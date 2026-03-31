import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import clean_text, chunk_text, load_raw_documents  # type: ignore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


@dataclass
class QueryCase:
    query_id: str
    query: str
    complex_query: bool = False
    allow_cache: bool = True
    expected_cache_hit: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class RetrievalHit:
    text: str
    metadata: Dict[str, str]
    dense_score: float
    lexical_score: float
    hybrid_score: float


@dataclass
class QueryMetrics:
    engine: str
    scenario: str
    query_id: str
    cache_hit: bool
    semantic_cache_lookup_ms: float
    retrieval_ms: float
    memory_load_ms: float
    memory_store_ms: float
    llm_generation_ms: float
    total_ms: float
    tokens_generated: int
    answer_preview: str


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                slot = hash(token) % self.dim
                sign = 1.0 if (hash(token + "#") & 1) == 0 else -1.0
                out[row, slot] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class BM25Retriever:
    def __init__(self, chunks: Sequence[CorpusChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        for tokens in self.doc_tokens:
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))
            for token in freqs:
                self.doc_freqs[token] += 1
        self.num_docs = max(len(self.chunks), 1)
        self.avg_doc_len = mean(self.doc_lengths) if self.doc_lengths else 1.0

    def score_query(self, query: str) -> Dict[int, float]:
        q_tokens = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        for token in q_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            for idx, freqs in enumerate(self.term_freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lengths[idx] / self.avg_doc_len)
                scores[idx] += idf * (tf * (self.k1 + 1.0) / denom)
        return scores


class DenseRetriever:
    def __init__(self, chunks: Sequence[CorpusChunk], embeddings: np.ndarray):
        self.chunks = list(chunks)
        self.embeddings = embeddings.astype(np.float32)

    def score_query_embedding(self, query_embedding: np.ndarray) -> Dict[int, float]:
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        sims = self.embeddings @ query_embedding.astype(np.float32)
        return {idx: float(score) for idx, score in enumerate(sims)}


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[CorpusChunk],
        embedder: HashingEmbedder,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [chunk.text for chunk in self.chunks]
        self.chunk_embeddings = embedder.embed_texts(texts)
        self.dense = DenseRetriever(self.chunks, self.chunk_embeddings)
        self.lexical = BM25Retriever(self.chunks)
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[RetrievalHit]:
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        dense_scores = self.dense.score_query_embedding(query_embedding)
        lexical_scores = self.lexical.score_query(query)
        dense_max = max((abs(v) for v in dense_scores.values()), default=1.0)
        lexical_max = max((abs(v) for v in lexical_scores.values()), default=1.0)
        combined: List[Tuple[int, float, float, float]] = []
        candidate_ids = set(dense_scores) | set(lexical_scores)
        for idx in candidate_ids:
            dense_norm = dense_scores.get(idx, 0.0) / dense_max if dense_max else 0.0
            lexical_norm = lexical_scores.get(idx, 0.0) / lexical_max if lexical_max else 0.0
            hybrid = self.dense_weight * dense_norm + self.lexical_weight * lexical_norm
            combined.append((idx, dense_scores.get(idx, 0.0), lexical_scores.get(idx, 0.0), hybrid))
        combined.sort(key=lambda item: item[3], reverse=True)
        hits: List[RetrievalHit] = []
        for idx, dense_score, lexical_score, hybrid_score in combined[:top_k]:
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=float(dense_score),
                    lexical_score=float(lexical_score),
                    hybrid_score=float(hybrid_score),
                )
            )
        return hits


class SemanticCache:
    def __init__(self, embedder: HashingEmbedder, similarity_threshold: float = 0.92):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.entries: List[Tuple[str, np.ndarray, str]] = []
        self.exact: Dict[str, str] = {}

    def lookup(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str], float]:
        if query in self.exact:
            return True, self.exact[query], 1.0
        if not self.entries:
            return False, None, 0.0
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        best_score = -1.0
        best_answer: Optional[str] = None
        for _, emb, answer in self.entries:
            score = float(np.dot(query_embedding, emb))
            if score > best_score:
                best_score = score
                best_answer = answer
        if best_score >= self.similarity_threshold:
            return True, best_answer, best_score
        return False, None, best_score

    def put(self, query: str, answer: str, query_embedding: Optional[np.ndarray] = None) -> None:
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        self.exact[query] = answer
        self.entries.append((query, query_embedding.astype(np.float32), answer))


class MockLLM:
    def __init__(self, base_latency_ms: float = 100.0, ms_per_token: float = 3.0, target_tokens: int = 80):
        self.base_latency_ms = base_latency_ms
        self.ms_per_token = ms_per_token
        self.target_tokens = target_tokens

    def generate(self, query: str, context: str) -> Tuple[str, int]:
        preview = context[:240].replace("\n", " ").strip()
        answer = (
            f"Answer: {query.strip()} | grounded by: {preview if preview else 'no retrieved context available'}"
        )
        tokens = min(self.target_tokens, max(24, len(answer.split()) + len(query.split()) // 2))
        time.sleep((self.base_latency_ms + tokens * self.ms_per_token) / 1000.0)
        return answer, tokens


class TinyLocalLLM:
    def __init__(self, corpus_texts: Sequence[str], max_tokens: int = 48):
        self.max_tokens = max_tokens
        self.transitions: Dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()
        for text in corpus_texts:
            words = tokenize(text)
            if not words:
                continue
            self.global_counts.update(words)
            for left, right in zip(words, words[1:]):
                self.transitions[left][right] += 1
        if not self.global_counts:
            self.global_counts.update(["context", "answer", "grounded"])

    def generate(self, query: str, context: str) -> Tuple[str, int]:
        query_words = tokenize(query)
        context_words = tokenize(context)
        seed = query_words[:4] + context_words[:4]
        if not seed:
            seed = ["answer"]
        generated = list(seed[: min(len(seed), 8)])
        current = generated[-1]
        for _ in range(self.max_tokens - len(generated)):
            candidates = self.transitions.get(current)
            if candidates:
                next_token = max(candidates.items(), key=lambda item: (item[1], item[0]))[0]
            else:
                next_token = max(self.global_counts.items(), key=lambda item: (item[1], item[0]))[0]
            generated.append(next_token)
            current = next_token
        answer = " ".join(generated[: self.max_tokens])
        return answer, len(generated[: self.max_tokens])


class HFLLM:
    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
        device: str = "cpu",
        max_new_tokens: int = 96,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, query: str, context: str) -> Tuple[str, int]:
        import torch

        prompt = f"Use the provided context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = outputs[0]
        input_len = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True).strip()
        tokens = max(1, len(answer.split()))
        return answer, tokens


@dataclass
class BenchmarkSummary:
    engine: str
    scenario: str
    count: int
    cache_hit_rate: float
    semantic_cache_lookup_ms_avg: float
    retrieval_ms_avg: float
    memory_load_ms_avg: float
    memory_store_ms_avg: float
    llm_generation_ms_avg: float
    total_ms_avg: float
    total_ms_p50: float
    total_ms_p95: float
    tokens_generated_avg: float


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.array(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, p))


def summarize_metrics(rows: Sequence[QueryMetrics]) -> List[BenchmarkSummary]:
    groups: Dict[Tuple[str, str], List[QueryMetrics]] = defaultdict(list)
    for row in rows:
        groups[(row.engine, row.scenario)].append(row)
    summaries: List[BenchmarkSummary] = []
    for (engine, scenario), bucket in sorted(groups.items()):
        summaries.append(
            BenchmarkSummary(
                engine=engine,
                scenario=scenario,
                count=len(bucket),
                cache_hit_rate=sum(1 for item in bucket if item.cache_hit) / max(len(bucket), 1),
                semantic_cache_lookup_ms_avg=mean(item.semantic_cache_lookup_ms for item in bucket),
                retrieval_ms_avg=mean(item.retrieval_ms for item in bucket),
                memory_load_ms_avg=mean(item.memory_load_ms for item in bucket),
                memory_store_ms_avg=mean(item.memory_store_ms for item in bucket),
                llm_generation_ms_avg=mean(item.llm_generation_ms for item in bucket),
                total_ms_avg=mean(item.total_ms for item in bucket),
                total_ms_p50=percentile([item.total_ms for item in bucket], 50),
                total_ms_p95=percentile([item.total_ms for item in bucket], 95),
                tokens_generated_avg=mean(item.tokens_generated for item in bucket),
            )
        )
    return summaries


def write_query_metrics_csv(path: Path, rows: Sequence[QueryMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else list(QueryMetrics.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_csv(path: Path, rows: Sequence[BenchmarkSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else list(BenchmarkSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_json(path: Path, query_rows: Sequence[QueryMetrics], summary_rows: Sequence[BenchmarkSummary]) -> None:
    payload = {
        "query_metrics": [asdict(row) for row in query_rows],
        "summary": [asdict(row) for row in summary_rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_corpus(input_path: str, max_chars: int, overlap_chars: int, file_glob: str = "*") -> List[CorpusChunk]:
    raw_pairs = load_raw_documents(input_path=input_path, file_glob=file_glob, rank=0, world_size=1)
    chunks: List[CorpusChunk] = []
    for doc_idx, (text, metadata) in enumerate(raw_pairs):
        cleaned = clean_text(text)
        for chunk_idx, chunk in enumerate(chunk_text(cleaned, max_chars=max_chars, overlap_chars=overlap_chars)):
            chunks.append(
                CorpusChunk(
                    chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                    text=chunk,
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )
    return chunks


def generate_query_cases(chunks: Sequence[CorpusChunk], count: int = 8) -> Dict[str, List[QueryCase]]:
    if not chunks:
        raise ValueError("No corpus chunks were generated.")
    seed_chunks = list(chunks[: max(count, 4)])

    def focus_text(chunk: CorpusChunk) -> str:
        words = chunk.text.split()
        return " ".join(words[: min(18, len(words))]).strip() or chunk.text[:120]

    semantic = [
        QueryCase(
            query_id=f"semantic_{idx}",
            query=f"Summarize this content: {focus_text(chunk)}",
            allow_cache=True,
            expected_cache_hit=True,
            tags=["semantic-cache"],
        )
        for idx, chunk in enumerate(seed_chunks[:count])
    ]
    retrieval = [
        QueryCase(
            query_id=f"retrieval_{idx}",
            query=f"Find the most relevant context for: {focus_text(chunk)}",
            allow_cache=False,
            tags=["retrieval"],
        )
        for idx, chunk in enumerate(seed_chunks[:count])
    ]
    generation = [
        QueryCase(
            query_id=f"generation_{idx}",
            query=f"Answer a grounded question about: {focus_text(chunk)}",
            allow_cache=False,
            tags=["generation"],
        )
        for idx, chunk in enumerate(seed_chunks[:count])
    ]
    complex_cases = [
        QueryCase(
            query_id=f"complex_{idx}",
            query=(
                f"Compare the main claims, supporting evidence, and caveats in these materials: {focus_text(chunk)}. "
                f"Provide a non-cached grounded answer with tradeoffs."
            ),
            complex_query=True,
            allow_cache=False,
            tags=["complex", "non-cached"],
        )
        for idx, chunk in enumerate(seed_chunks[:count])
    ]
    return {
        "semantic_cache_lookup": semantic,
        "retrieval_hybrid": retrieval,
        "llm_generation": generation,
        "non_cached_complex_query": complex_cases,
    }
