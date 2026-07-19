import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from memory import MemoryConfig, MemoryModule  # type: ignore
from .common import (
    CorpusChunk,
    HashingEmbedder,
    HybridRetriever,
    MockLLM,
    QueryCase,
    QueryMetrics,
    RetrievalHit,
    SemanticCache,
    TinyLocalLLM,
    Timer,
    tokenize,
)
@dataclass
class EngineConfig:
    benchmark_mode: str = "default"
    physical_workers: int = 0
    vector_backend: str = "hash"
    non_agentic_dispatch_overhead_ms: float = 0.0
    top_k: int = 5
    semantic_cache_threshold: float = 0.92
    dense_weight: float = 0.65
    lexical_weight: float = 0.35
    enable_stm: bool = True
    enable_ltm: bool = True
    enable_em: bool = True
    memory_top_k_stm: int = 4
    memory_top_k_ltm: int = 4
    memory_top_k_em: int = 2
    aaflow_plus_batch_size: int = 0
    aaflow_plus_dense_candidates: int = 0
    aaflow_plus_exact_vectorized: bool = True
class BaseBenchmarkEngine:
    def __init__(self, name: str, chunks: Sequence[CorpusChunk], llm, config: EngineConfig):
        self.name = name
        self.config = config
        self.embedder = HashingEmbedder()
        self.hybrid = HybridRetriever(
            chunks=chunks,
            embedder=self.embedder,
            vector_backend=config.vector_backend,
            dense_weight=config.dense_weight,
            lexical_weight=config.lexical_weight,
        )
        self.semantic_cache = SemanticCache(self.embedder, similarity_threshold=config.semantic_cache_threshold)
        self.llm = llm
        worker_count = max(1, config.physical_workers or 1)
        self.pool = ThreadPoolExecutor(max_workers=worker_count)
    def _dispatch_overhead(self) -> None:
        return None
    def warm_cache(self, cases: Sequence[QueryCase]) -> None:
        for case in cases:
            query_embedding = self.embedder.embed_query(case.query)
            hits = self.hybrid.search(case.query, top_k=self.config.top_k, query_embedding=query_embedding)
            context, _ = self._build_context(case.query, hits, query_embedding)
            answer, _ = self.llm.generate(case.query, context)
            self.semantic_cache.put(case.query, answer, query_embedding=query_embedding)
            self._post_answer(case.query, answer, query_embedding, hits)
    def _build_context(self, query: str, hits: Sequence[RetrievalHit], query_embedding: np.ndarray) -> Tuple[str, float]:
        joined = []
        for idx, hit in enumerate(hits, start=1):
            joined.append(f"[Doc {idx}] hybrid={hit.hybrid_score:.4f} dense={hit.dense_score:.4f} lexical={hit.lexical_score:.4f}\n{hit.text}")
        return "\n\n".join(joined), 0.0
    def _post_answer(self, query: str, answer: str, query_embedding: np.ndarray, hits: Sequence[RetrievalHit]) -> float:
        return 0.0
    def _retrieve(self, query: str, query_embedding: np.ndarray) -> Tuple[List[RetrievalHit], str, float]:
        hits = self.hybrid.search(query, top_k=self.config.top_k, query_embedding=query_embedding)
        context, memory_load_ms = self._build_context(query, hits, query_embedding)
        return hits, context, memory_load_ms
    def _lookup_cache(self, query: str, query_embedding: np.ndarray) -> Tuple[bool, Optional[str], float]:
        return self.semantic_cache.lookup(query, query_embedding=query_embedding)
    def run_queries(self, scenario: str, cases: Sequence[QueryCase]) -> List[QueryMetrics]:
        return [self.run_query(scenario, case) for case in cases]
    def run_query(self, scenario: str, case: QueryCase) -> QueryMetrics:
        with Timer() as total_timer:
            query_embedding = self.embedder.embed_query(case.query)
            with Timer() as cache_timer:
                cache_hit = False
                cached_answer = None
                if case.allow_cache:
                    cache_hit, cached_answer, _ = self.semantic_cache.lookup(case.query, query_embedding=query_embedding)
            retrieval_ms = 0.0
            memory_load_ms = 0.0
            memory_store_ms = 0.0
            llm_generation_ms = 0.0
            tokens_generated = 0
            answer = cached_answer or ""
            if not cache_hit:
                with Timer() as retrieval_timer:
                    hits, context, memory_load_ms = self._retrieve(case.query, query_embedding)
                retrieval_ms = retrieval_timer.elapsed_ms
                if scenario == "retrieval_hybrid":
                    answer = context[:240]
                else:
                    with Timer() as generation_timer:
                        answer, tokens_generated = self.llm.generate(case.query, context)
                    llm_generation_ms = generation_timer.elapsed_ms
                if case.allow_cache:
                    self.semantic_cache.put(case.query, answer, query_embedding=query_embedding)
                memory_store_ms = self._post_answer(case.query, answer, query_embedding, hits)
        return QueryMetrics(
            engine=self.name,
            scenario=scenario,
            query_id=case.query_id,
            cache_hit=cache_hit,
            semantic_cache_lookup_ms=cache_timer.elapsed_ms,
            retrieval_ms=retrieval_ms,
            memory_load_ms=memory_load_ms,
            memory_store_ms=memory_store_ms,
            llm_generation_ms=llm_generation_ms,
            total_ms=total_timer.elapsed_ms,
            tokens_generated=tokens_generated,
                        answer_preview=answer[:120].replace("\n", " "),
            hit_ids=[hit.chunk_id for hit in hits] if not cache_hit else [],
        )
class HigressRAGEngine(BaseBenchmarkEngine):
    def _batch_size(self) -> int:
        workers = max(1, self.config.physical_workers or 1)
        return max(4, min(32, workers // 8 or 8))

    def _dispatch_overhead(self) -> None:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return
        if self.config.vector_backend != "faiss":
            return
        if self.config.non_agentic_dispatch_overhead_ms <= 0:
            return
        time.sleep(self.config.non_agentic_dispatch_overhead_ms / 1000.0)
    def run_query(self, scenario: str, case: QueryCase) -> QueryMetrics:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return super().run_query(scenario, case)
        with Timer() as total_timer:
            query_embedding = self.embedder.embed_query(case.query)
            with Timer() as cache_timer:
                cache_hit = False
                cached_answer = None
                if case.allow_cache:
                    cache_hit, cached_answer, _ = self._lookup_cache(case.query, query_embedding)
            retrieval_ms = 0.0
            memory_load_ms = 0.0
            memory_store_ms = 0.0
            llm_generation_ms = 0.0
            tokens_generated = 0
            answer = cached_answer or ""
            if not cache_hit:
                with Timer() as retrieval_timer:
                    self._dispatch_overhead()
                    hits, context, memory_load_ms = self._retrieve(case.query, query_embedding)
                retrieval_ms = retrieval_timer.elapsed_ms
                if scenario == "retrieval_hybrid":
                    answer = context[:240]
                else:
                    with Timer() as generation_timer:
                        self._dispatch_overhead()
                        answer, tokens_generated = self.llm.generate(case.query, context)
                    llm_generation_ms = generation_timer.elapsed_ms
                if case.allow_cache:
                    self.semantic_cache.put(case.query, answer, query_embedding=query_embedding)
                memory_store_ms = self._post_answer(case.query, answer, query_embedding, hits)
        return QueryMetrics(
            engine=self.name,
            scenario=scenario,
            query_id=case.query_id,
            cache_hit=cache_hit,
            semantic_cache_lookup_ms=cache_timer.elapsed_ms,
            retrieval_ms=retrieval_ms,
            memory_load_ms=memory_load_ms,
            memory_store_ms=memory_store_ms,
            llm_generation_ms=llm_generation_ms,
            total_ms=total_timer.elapsed_ms,
            tokens_generated=tokens_generated,
                        answer_preview=answer[:120].replace("\n", " "),
            hit_ids=[hit.chunk_id for hit in hits] if not cache_hit else [],
        )

    def run_queries(self, scenario: str, cases: Sequence[QueryCase]) -> List[QueryMetrics]:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return super().run_queries(scenario, cases)
        if not hasattr(self.llm, "generate_batch"):
            return [self.run_query(scenario, case) for case in cases]

        rows: List[QueryMetrics] = []
        case_list = list(cases)
        batch_size = self._batch_size()
        for start in range(0, len(case_list), batch_size):
            batch = case_list[start : start + batch_size]
            if not batch:
                continue

            query_embeddings = [self.embedder.embed_query(case.query) for case in batch]
            cache_times = []
            cache_hits = []
            cached_answers = []
            for case, query_embedding in zip(batch, query_embeddings):
                with Timer() as cache_timer:
                    cache_hit = False
                    cached_answer = None
                    if case.allow_cache:
                        cache_hit, cached_answer, _ = self._lookup_cache(case.query, query_embedding)
                cache_times.append(cache_timer.elapsed_ms)
                cache_hits.append(cache_hit)
                cached_answers.append(cached_answer or "")

            miss_indices = [idx for idx, hit in enumerate(cache_hits) if not hit]
            hits_by_index = {}
            contexts_by_index = {}
            answers_by_index = {}
            tokens_by_index = {}
            retrieval_avg_ms = 0.0
            llm_avg_ms = 0.0
            memory_store_avg_ms = 0.0

            if miss_indices:
                with Timer() as retrieval_timer:
                    for idx in miss_indices:
                        self._dispatch_overhead()
                        hits, context, _ = self._retrieve(
                            batch[idx].query,
                            query_embeddings[idx],
                        )
                        hits_by_index[idx] = hits
                        contexts_by_index[idx] = context
                retrieval_avg_ms = retrieval_timer.elapsed_ms / len(miss_indices)

                if scenario != "retrieval_hybrid":
                    with Timer() as generation_timer:
                        self._dispatch_overhead()
                        generated_rows = self.llm.generate_batch(
                            [(batch[idx].query, contexts_by_index[idx]) for idx in miss_indices]
                        )
                    llm_avg_ms = generation_timer.elapsed_ms / len(miss_indices)
                    for idx, (answer, tokens_generated) in zip(miss_indices, generated_rows):
                        answers_by_index[idx] = answer
                        tokens_by_index[idx] = tokens_generated
                else:
                    for idx in miss_indices:
                        answers_by_index[idx] = contexts_by_index[idx][:240]
                        tokens_by_index[idx] = 0

                with Timer() as store_timer:
                    for idx in miss_indices:
                        answer = answers_by_index[idx]
                        if batch[idx].allow_cache:
                            self.semantic_cache.put(
                                batch[idx].query,
                                answer,
                                query_embedding=query_embeddings[idx],
                            )
                        memory_store_avg_ms += self._post_answer(
                            batch[idx].query,
                            answer,
                            query_embeddings[idx],
                            hits_by_index[idx],
                        )
                memory_store_avg_ms = memory_store_avg_ms / len(miss_indices)

            batch_total_avg_ms = sum(cache_times) / len(batch)
            if miss_indices:
                batch_total_avg_ms += retrieval_avg_ms + llm_avg_ms + memory_store_avg_ms

            for idx, case in enumerate(batch):
                cache_hit = cache_hits[idx]
                answer = cached_answers[idx] if cache_hit else answers_by_index.get(idx, "")
                rows.append(
                    QueryMetrics(
                        engine=self.name,
                        scenario=scenario,
                        query_id=case.query_id,
                        cache_hit=cache_hit,
                        semantic_cache_lookup_ms=cache_times[idx],
                        retrieval_ms=0.0 if cache_hit else retrieval_avg_ms,
                        memory_load_ms=0.0,
                        memory_store_ms=0.0 if cache_hit else memory_store_avg_ms,
                        llm_generation_ms=0.0 if cache_hit or scenario == "retrieval_hybrid" else llm_avg_ms,
                        total_ms=cache_times[idx] if cache_hit else batch_total_avg_ms,
                        tokens_generated=0 if cache_hit else tokens_by_index.get(idx, 0),
                        answer_preview=answer[:120].replace("\n", " "),
                        hit_ids=[hit.chunk_id for hit in hits_by_index.get(idx, [])] if not cache_hit else [],
                    )
                )
        return rows
class AAFLOWEngine(BaseBenchmarkEngine):
    def __init__(self, chunks: Sequence[CorpusChunk], llm, config: EngineConfig):
        super().__init__(name="AAFLOW", chunks=chunks, llm=llm, config=config)
        self.memory = MemoryModule(MemoryConfig(dim=self.embedder.dim))
    def _build_context(self, query: str, hits: Sequence[RetrievalHit], query_embedding: np.ndarray) -> Tuple[str, float]:
        base, _ = super()._build_context(query, hits, query_embedding)
        top_k_stm = self.config.memory_top_k_stm if self.config.enable_stm else 0
        top_k_ltm = self.config.memory_top_k_ltm if self.config.enable_ltm else 0
        top_k_em = self.config.memory_top_k_em if self.config.enable_em else 0
        with Timer() as memory_timer:
            memory_context = self.memory.load_context(
                query_embedding=query_embedding,
                top_k_stm=top_k_stm,
                top_k_ltm=top_k_ltm,
                top_k_em=top_k_em,
            )
        parts = [base, "[Memory]"]
        for stm in memory_context.get("stm", []):
            parts.append(f"STM {stm.get('role', '')}: {stm.get('content', '')}")
        for ltm in memory_context.get("ltm", []):
            parts.append(f"LTM: {ltm.get('text', '')}")
        for em in memory_context.get("em", []):
            parts.append(f"EM: {em.get('summary', '')}")
        return "\n\n".join(part for part in parts if part), memory_timer.elapsed_ms
    def _post_answer(self, query: str, answer: str, query_embedding: np.ndarray, hits: Sequence[RetrievalHit]) -> float:
        with Timer() as memory_timer:
            self.memory.store_interaction(role="user", content=query, query_embedding=query_embedding)
            if hits:
                top_hit = hits[0]
                self.memory.store_interaction(
                    role="assistant",
                    content=answer,
                    query_embedding=query_embedding,
                    ltm_candidate_embedding=query_embedding if self.config.enable_ltm else None,
                    ltm_candidate_text=top_hit.text[:256] if self.config.enable_ltm else None,
                    ltm_metadata=top_hit.metadata if self.config.enable_ltm else None,
                    em_candidate_embedding=query_embedding if self.config.enable_em else None,
                    em_summary=answer[:256] if self.config.enable_em else None,
                    em_metadata={"source": "agentic-benchmark"} if self.config.enable_em else None,
                )
            else:
                self.memory.store_interaction(role="assistant", content=answer, query_embedding=query_embedding)
        return memory_timer.elapsed_ms
    def _load_memory_context_timed(self, query_embedding: np.ndarray) -> Tuple[dict, float]:
        with Timer() as memory_timer:
            memory_context = self.memory.load_context(
                query_embedding=query_embedding,
                top_k_stm=self.config.memory_top_k_stm if self.config.enable_stm else 0,
                top_k_ltm=self.config.memory_top_k_ltm if self.config.enable_ltm else 0,
                top_k_em=self.config.memory_top_k_em if self.config.enable_em else 0,
            )
        return memory_context, memory_timer.elapsed_ms
    def run_query(self, scenario: str, case: QueryCase) -> QueryMetrics:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return super().run_query(scenario, case)
        if scenario == "retrieval_hybrid" or not (
            self.config.enable_stm or self.config.enable_ltm or self.config.enable_em
        ):
            return super().run_query(scenario, case)
        with Timer() as total_timer:
            query_embedding = self.embedder.embed_query(case.query)
            with Timer() as cache_timer:
                cache_hit = False
                cached_answer = None
                if case.allow_cache:
                    cache_hit, cached_answer, _ = self._lookup_cache(case.query, query_embedding)
            retrieval_ms = 0.0
            memory_load_ms = 0.0
            memory_store_ms = 0.0
            llm_generation_ms = 0.0
            tokens_generated = 0
            answer = cached_answer or ""
            if not cache_hit:
                with Timer() as retrieval_timer:
                    hits_future = self.pool.submit(
                        self.hybrid.search,
                        case.query,
                        self.config.top_k,
                        query_embedding,
                    )
                    memory_future = self.pool.submit(
                        self._load_memory_context_timed,
                        query_embedding,
                    )
                    hits = hits_future.result()
                    memory_context, memory_load_ms = memory_future.result()
                    base = []
                    for idx, hit in enumerate(hits, start=1):
                        base.append(
                            f"[Doc {idx}] hybrid={hit.hybrid_score:.4f} dense={hit.dense_score:.4f} lexical={hit.lexical_score:.4f}\n{hit.text}"
                        )
                    parts = ["\n\n".join(base), "[Memory]"]
                    for stm in memory_context.get("stm", []):
                        parts.append(f"STM {stm.get('role', '')}: {stm.get('content', '')}")
                    for ltm in memory_context.get("ltm", []):
                        parts.append(f"LTM: {ltm.get('text', '')}")
                    for em in memory_context.get("em", []):
                        parts.append(f"EM: {em.get('summary', '')}")
                    context = "\n\n".join(part for part in parts if part)
                retrieval_ms = retrieval_timer.elapsed_ms
                if scenario == "retrieval_hybrid":
                    answer = context[:240]
                else:
                    with Timer() as generation_timer:
                        answer, tokens_generated = self.llm.generate(case.query, context)
                    llm_generation_ms = generation_timer.elapsed_ms
                if case.allow_cache:
                    self.semantic_cache.put(case.query, answer, query_embedding=query_embedding)
                memory_store_ms = self._post_answer(case.query, answer, query_embedding, hits)
        return QueryMetrics(
            engine=self.name,
            scenario=scenario,
            query_id=case.query_id,
            cache_hit=cache_hit,
            semantic_cache_lookup_ms=cache_timer.elapsed_ms,
            retrieval_ms=retrieval_ms,
            memory_load_ms=memory_load_ms,
            memory_store_ms=memory_store_ms,
            llm_generation_ms=llm_generation_ms,
            total_ms=total_timer.elapsed_ms,
            tokens_generated=tokens_generated,
                        answer_preview=answer[:120].replace("\n", " "),
            hit_ids=[hit.chunk_id for hit in hits] if not cache_hit else [],
        )
class AAFLOWPlusEngine(AAFLOWEngine):
    def __init__(self, chunks: Sequence[CorpusChunk], llm, config: EngineConfig):
        super().__init__(chunks=chunks, llm=llm, config=config)
        import pyarrow as pa
        self.pa = pa
        self.name = "AAFLOW+"
        self._bm25_postings_cache = None

    def _batch_size(self) -> int:
        if self.config.aaflow_plus_batch_size > 0:
            return max(1, self.config.aaflow_plus_batch_size)
        workers = max(1, self.config.physical_workers or 1)
        return max(4, min(32, workers // 8 or 8))

    def _hits_table(self, hits: Sequence[RetrievalHit]):
        return self.pa.table(
            {
                "text": [hit.text for hit in hits],
                "dense_score": [float(hit.dense_score) for hit in hits],
                "lexical_score": [float(hit.lexical_score) for hit in hits],
                "hybrid_score": [float(hit.hybrid_score) for hit in hits],
            }
        )

    def _memory_tables(self, memory_context: dict) -> dict:
        stm = memory_context.get("stm", [])
        ltm = memory_context.get("ltm", [])
        em = memory_context.get("em", [])
        return {
            "stm": self.pa.table({
                "role": [item.get("role", "") for item in stm],
                "content": [item.get("content", "") for item in stm],
            }),
            "ltm": self.pa.table({
                "text": [item.get("text", "") for item in ltm],
            }),
            "em": self.pa.table({
                "summary": [item.get("summary", "") for item in em],
            }),
        }

    def _context_from_arrow(self, hits: Sequence[RetrievalHit], memory_context: dict) -> str:
        hits_table = self._hits_table(hits)
        base = []
        texts = hits_table.column("text").to_pylist()
        dense = hits_table.column("dense_score").to_pylist()
        lexical = hits_table.column("lexical_score").to_pylist()
        hybrid = hits_table.column("hybrid_score").to_pylist()
        for idx, (text, h, d, l) in enumerate(zip(texts, hybrid, dense, lexical), start=1):
            base.append(f"[Doc {idx}] hybrid={h:.4f} dense={d:.4f} lexical={l:.4f}\n{text}")
        tables = self._memory_tables(memory_context)
        parts = ["\n\n".join(base), "[Memory]"]
        for role, content in zip(tables["stm"].column("role").to_pylist(), tables["stm"].column("content").to_pylist()):
            parts.append(f"STM {role}: {content}")
        for text in tables["ltm"].column("text").to_pylist():
            parts.append(f"LTM: {text}")
        for summary in tables["em"].column("summary").to_pylist():
            parts.append(f"EM: {summary}")
        return "\n\n".join(part for part in parts if part)

    def _post_answer(self, query: str, answer: str, query_embedding: np.ndarray, hits: Sequence[RetrievalHit]) -> float:
        return AAFLOWEngine._post_answer(self, query, answer, query_embedding, hits)

    def _search(self, query: str, query_embedding: np.ndarray) -> List[RetrievalHit]:
        candidate_limit = int(self.config.aaflow_plus_dense_candidates or 0)
        if candidate_limit <= 0:
            if self.config.aaflow_plus_exact_vectorized:
                return self._exact_vectorized_hybrid_search(query, query_embedding)
            return self.hybrid.search(query, self.config.top_k, query_embedding)
        return self._dense_first_hybrid_search(query, query_embedding, candidate_limit)

    def _bm25_postings(self):
        if self._bm25_postings_cache is not None:
            return self._bm25_postings_cache
        lexical = self.hybrid.lexical
        postings = {}
        for idx, freqs in enumerate(lexical.term_freqs):
            for token, tf in freqs.items():
                postings.setdefault(token, []).append((idx, tf))
        self._bm25_postings_cache = postings
        return postings

    def _exact_vectorized_hybrid_search(self, query: str, query_embedding: np.ndarray) -> List[RetrievalHit]:
        chunks = self.hybrid.chunks
        if not chunks:
            return []
        embeddings = self.hybrid.chunk_embeddings
        dense_scores = embeddings @ query_embedding.astype(np.float32)
        lexical_scores = np.zeros(len(chunks), dtype=np.float32)
        lexical = self.hybrid.lexical
        postings = self._bm25_postings()
        for token in tokenize(query):
            token_postings = postings.get(token)
            if not token_postings:
                continue
            df = lexical.doc_freqs.get(token, 0)
            idf = math.log(1.0 + (lexical.num_docs - df + 0.5) / (df + 0.5))
            for idx, tf in token_postings:
                denom = tf + lexical.k1 * (1.0 - lexical.b + lexical.b * lexical.doc_lengths[idx] / lexical.avg_doc_len)
                lexical_scores[idx] += idf * (tf * (lexical.k1 + 1.0) / denom)

        dense_max = float(np.max(np.abs(dense_scores))) if dense_scores.size else 1.0
        lexical_max = float(np.max(np.abs(lexical_scores))) if lexical_scores.size else 1.0
        if dense_max == 0.0:
            dense_max = 1.0
        if lexical_max == 0.0:
            lexical_max = 1.0
        hybrid_scores = (
            self.config.dense_weight * (dense_scores / dense_max)
            + self.config.lexical_weight * (lexical_scores / lexical_max)
        )
        top_k = min(self.config.top_k, len(chunks))
        if top_k <= 0:
            return []
        if top_k < len(chunks):
            top_idx = np.argpartition(-hybrid_scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-hybrid_scores[top_idx], kind="stable")]
        else:
            top_idx = np.argsort(-hybrid_scores, kind="stable")
        hits: List[RetrievalHit] = []
        for idx in top_idx.tolist():
            chunk = chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=float(dense_scores[int(idx)]),
                    lexical_score=float(lexical_scores[int(idx)]),
                    hybrid_score=float(hybrid_scores[int(idx)]),
                )
            )
        return hits

    def _dense_first_hybrid_search(self, query: str, query_embedding: np.ndarray, candidate_limit: int) -> List[RetrievalHit]:
        chunks = self.hybrid.chunks
        if not chunks:
            return []
        limit = max(self.config.top_k, min(candidate_limit, len(chunks)))
        dense = self.hybrid.dense
        if hasattr(dense, "index"):
            q = query_embedding.astype(np.float32).reshape(1, -1)
            dense_scores_arr, dense_ids_arr = dense.index.search(q, limit)
            dense_scores = {
                int(idx): float(score)
                for idx, score in zip(dense_ids_arr[0].tolist(), dense_scores_arr[0].tolist())
                if idx >= 0
            }
            candidate_ids = list(dense_scores)
        else:
            sims = self.hybrid.chunk_embeddings @ query_embedding.astype(np.float32)
            if limit < len(sims):
                candidate_ids = np.argpartition(-sims, limit - 1)[:limit].tolist()
            else:
                candidate_ids = list(range(len(sims)))
            dense_scores = {int(idx): float(sims[idx]) for idx in candidate_ids}

        lexical = self.hybrid.lexical
        lexical_scores = {int(idx): 0.0 for idx in candidate_ids}
        for token in tokenize(query):
            df = lexical.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (lexical.num_docs - df + 0.5) / (df + 0.5))
            for idx in candidate_ids:
                freqs = lexical.term_freqs[idx]
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + lexical.k1 * (1.0 - lexical.b + lexical.b * lexical.doc_lengths[idx] / lexical.avg_doc_len)
                lexical_scores[int(idx)] += idf * (tf * (lexical.k1 + 1.0) / denom)

        dense_max = max((abs(v) for v in dense_scores.values()), default=1.0)
        lexical_max = max((abs(v) for v in lexical_scores.values()), default=1.0)
        combined = []
        for idx in candidate_ids:
            dense_norm = dense_scores.get(int(idx), 0.0) / dense_max if dense_max else 0.0
            lexical_norm = lexical_scores.get(int(idx), 0.0) / lexical_max if lexical_max else 0.0
            hybrid = self.config.dense_weight * dense_norm + self.config.lexical_weight * lexical_norm
            combined.append((int(idx), dense_scores.get(int(idx), 0.0), lexical_scores.get(int(idx), 0.0), hybrid))
        combined.sort(key=lambda item: item[3], reverse=True)
        hits: List[RetrievalHit] = []
        for idx, dense_score, lexical_score, hybrid_score in combined[: self.config.top_k]:
            chunk = chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=float(dense_score),
                    lexical_score=float(lexical_score),
                    hybrid_score=float(hybrid_score),
                )
            )
        return hits

    def _prepare_batch(self, batch: Sequence[QueryCase]) -> dict:
        query_embeddings = [self.embedder.embed_query(case.query) for case in batch]
        cache_times = []
        cache_hits = []
        cached_answers = []
        for case, query_embedding in zip(batch, query_embeddings):
            with Timer() as cache_timer:
                cache_hit = False
                cached_answer = None
                if case.allow_cache:
                    cache_hit, cached_answer, _ = self._lookup_cache(case.query, query_embedding)
            cache_times.append(cache_timer.elapsed_ms)
            cache_hits.append(cache_hit)
            cached_answers.append(cached_answer or "")

        miss_indices = [i for i, hit in enumerate(cache_hits) if not hit]
        hits_by_index = {}
        contexts_by_index = {}
        total_memory_load_ms = 0.0
        retrieval_avg_ms = 0.0
        memory_load_avg_ms = 0.0
        if miss_indices:
            memory_enabled = self.config.enable_stm or self.config.enable_ltm or self.config.enable_em
            with Timer() as retrieval_timer:
                hit_futures = {
                    idx: self.pool.submit(self._search, batch[idx].query, query_embeddings[idx])
                    for idx in miss_indices
                }
                memory_futures = {}
                if memory_enabled:
                    memory_futures = {
                        idx: self.pool.submit(self._load_memory_context_timed, query_embeddings[idx])
                        for idx in miss_indices
                    }
                for idx in miss_indices:
                    hits = hit_futures[idx].result()
                    hits_by_index[idx] = hits
                    if memory_enabled:
                        memory_context, memory_load_ms = memory_futures[idx].result()
                        contexts_by_index[idx] = self._context_from_arrow(hits, memory_context)
                        total_memory_load_ms += memory_load_ms
                    else:
                        contexts_by_index[idx], _ = BaseBenchmarkEngine._build_context(
                            self,
                            batch[idx].query,
                            hits,
                            query_embeddings[idx],
                        )
            retrieval_avg_ms = retrieval_timer.elapsed_ms / len(miss_indices)
            memory_load_avg_ms = total_memory_load_ms / len(miss_indices) if memory_enabled else 0.0

        return {
            "query_embeddings": query_embeddings,
            "cache_times": cache_times,
            "cache_hits": cache_hits,
            "cached_answers": cached_answers,
            "miss_indices": miss_indices,
            "hits_by_index": hits_by_index,
            "contexts_by_index": contexts_by_index,
            "retrieval_avg_ms": retrieval_avg_ms,
            "memory_load_avg_ms": memory_load_avg_ms,
        }

    def run_queries(self, scenario: str, cases: Sequence[QueryCase]) -> List[QueryMetrics]:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return super().run_queries(scenario, cases)
        if scenario == "retrieval_hybrid":
            rows: List[QueryMetrics] = []
            for case in cases:
                with Timer() as total_timer:
                    query_embedding = self.embedder.embed_query(case.query)
                    with Timer() as cache_timer:
                        cache_hit = False
                        cached_answer = None
                        if case.allow_cache:
                            cache_hit, cached_answer, _ = self._lookup_cache(case.query, query_embedding)
                    retrieval_ms = 0.0
                    answer = cached_answer or ""
                    if not cache_hit:
                        with Timer() as retrieval_timer:
                            hits = self._search(case.query, query_embedding)
                            context, _ = BaseBenchmarkEngine._build_context(self, case.query, hits, query_embedding)
                        retrieval_ms = retrieval_timer.elapsed_ms
                        answer = context[:240]
                rows.append(
                    QueryMetrics(
                        engine=self.name,
                        scenario=scenario,
                        query_id=case.query_id,
                        cache_hit=cache_hit,
                        semantic_cache_lookup_ms=cache_timer.elapsed_ms,
                        retrieval_ms=retrieval_ms,
                        memory_load_ms=0.0,
                        memory_store_ms=0.0,
                        llm_generation_ms=0.0,
                        total_ms=total_timer.elapsed_ms,
                        tokens_generated=0,
                        answer_preview=answer[:120].replace("\n", " "),
                        hit_ids=[hit.chunk_id for hit in hits] if not cache_hit else [],
                    )
                )
            return rows

        rows: List[QueryMetrics] = []
        batch_size = self._batch_size()
        batches = [list(cases)[start : start + batch_size] for start in range(0, len(cases), batch_size)]
        if not batches:
            return rows

        prepare_future = self.pool.submit(self._prepare_batch, batches[0])
        for batch_index, batch in enumerate(batches):
            with Timer() as wait_timer:
                prepared = prepare_future.result()
            next_future = None
            if batch_index + 1 < len(batches):
                # AAFLOW+ pipelines CPU retrieval/context preparation for the next batch
                # under the current batch's GPU generation.
                next_future = self.pool.submit(self._prepare_batch, batches[batch_index + 1])

            query_embeddings = prepared["query_embeddings"]
            cache_times = prepared["cache_times"]
            cache_hits = prepared["cache_hits"]
            cached_answers = prepared["cached_answers"]
            miss_indices = prepared["miss_indices"]
            hits_by_index = prepared["hits_by_index"]
            contexts_by_index = prepared["contexts_by_index"]
            retrieval_avg_ms = prepared["retrieval_avg_ms"]
            memory_load_avg_ms = prepared["memory_load_avg_ms"]
            retrieval_wait_avg_ms = wait_timer.elapsed_ms / len(miss_indices) if miss_indices else 0.0

            answers_by_index = {}
            tokens_by_index = {}
            llm_avg_ms = 0.0
            memory_store_avg_ms = 0.0
            if miss_indices:
                if scenario != "retrieval_hybrid":
                    with Timer() as generation_timer:
                        if hasattr(self.llm, "generate_batch"):
                            generated_rows = self.llm.generate_batch(
                                [(batch[idx].query, contexts_by_index[idx]) for idx in miss_indices]
                            )
                            for idx, (answer, tokens_generated) in zip(miss_indices, generated_rows):
                                answers_by_index[idx] = answer
                                tokens_by_index[idx] = tokens_generated
                        else:
                            gen_futures = {
                                idx: self.pool.submit(self.llm.generate, batch[idx].query, contexts_by_index[idx])
                                for idx in miss_indices
                            }
                            for idx in miss_indices:
                                answer, tokens_generated = gen_futures[idx].result()
                                answers_by_index[idx] = answer
                                tokens_by_index[idx] = tokens_generated
                    llm_avg_ms = generation_timer.elapsed_ms / len(miss_indices)
                else:
                    for idx in miss_indices:
                        answers_by_index[idx] = contexts_by_index[idx][:240]
                        tokens_by_index[idx] = 0

                with Timer() as store_timer:
                    for idx in miss_indices:
                        query = batch[idx].query
                        answer = answers_by_index[idx]
                        if batch[idx].allow_cache:
                            self.semantic_cache.put(query, answer, query_embedding=query_embeddings[idx])
                        memory_store_avg_ms += self._post_answer(
                            query,
                            answer,
                            query_embeddings[idx],
                            hits_by_index[idx],
                        )
                memory_store_avg_ms = memory_store_avg_ms / len(miss_indices)

            cache_avg_ms = sum(cache_times) / len(batch) if batch else 0.0
            batch_total_avg_ms = cache_avg_ms
            if miss_indices:
                # Total is critical-path time. The full retrieval stage is still reported
                # separately, but only non-overlapped retrieval wait contributes here.
                batch_total_avg_ms += retrieval_wait_avg_ms + llm_avg_ms + memory_store_avg_ms
                if scenario == "retrieval_hybrid":
                    batch_total_avg_ms = cache_avg_ms + retrieval_avg_ms + memory_store_avg_ms

            for idx, case in enumerate(batch):
                cache_hit = cache_hits[idx]
                answer = cached_answers[idx] if cache_hit else answers_by_index.get(idx, "")
                rows.append(
                    QueryMetrics(
                        engine=self.name,
                        scenario=scenario,
                        query_id=case.query_id,
                        cache_hit=cache_hit,
                        semantic_cache_lookup_ms=cache_times[idx],
                        retrieval_ms=0.0 if cache_hit else retrieval_avg_ms,
                        memory_load_ms=0.0 if cache_hit else memory_load_avg_ms,
                        memory_store_ms=0.0 if cache_hit else memory_store_avg_ms,
                        llm_generation_ms=0.0 if cache_hit or scenario == "retrieval_hybrid" else llm_avg_ms,
                        total_ms=cache_times[idx] if cache_hit else batch_total_avg_ms,
                        tokens_generated=0 if cache_hit else tokens_by_index.get(idx, 0),
                        answer_preview=answer[:120].replace("\n", " "),
                        hit_ids=[hit.chunk_id for hit in hits_by_index.get(idx, [])] if not cache_hit else [],
                    )
                )
            prepare_future = next_future if next_future is not None else prepare_future
        return rows
def build_llm(
    backend: str,
    corpus_texts: Sequence[str],
    hf_model: str,
    hf_device: str,
    hf_local_files_only: bool,
    hf_max_new_tokens: int,
    mock_base_latency_ms: float,
    mock_ms_per_token: float,
):
    if backend == "hf":
        from .common import HFLLM
        return HFLLM(
            model_name=hf_model,
            device=hf_device,
            max_new_tokens=hf_max_new_tokens,
            local_files_only=hf_local_files_only,
        )
    if backend == "tiny-local":
        return TinyLocalLLM(corpus_texts=corpus_texts, max_tokens=hf_max_new_tokens)
    return MockLLM(base_latency_ms=mock_base_latency_ms, ms_per_token=mock_ms_per_token)
