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
        )
class HigressRAGEngine(BaseBenchmarkEngine):
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
        )
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
        )
class AAFLOWPlusEngine(AAFLOWEngine):
    def __init__(self, chunks: Sequence[CorpusChunk], llm, config: EngineConfig):
        super().__init__(chunks=chunks, llm=llm, config=config)
        import pyarrow as pa
        self.pa = pa
        self.name = "AAFLOW+"
    def _batch_size(self) -> int:
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
    def run_queries(self, scenario: str, cases: Sequence[QueryCase]) -> List[QueryMetrics]:
        if self.config.benchmark_mode != "fair_parallelism_plus_overlap":
            return super().run_queries(scenario, cases)
        if not (self.config.enable_stm or self.config.enable_ltm or self.config.enable_em):
            return super().run_queries(scenario, cases)
        rows: List[QueryMetrics] = []
        batch_size = self._batch_size()
        case_list = list(cases)
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
            miss_indices = [i for i, hit in enumerate(cache_hits) if not hit]
            hits_by_index = {}
            contexts_by_index = {}
            tokens_by_index = {}
            answers_by_index = {}
            retrieval_avg_ms = 0.0
            memory_load_avg_ms = 0.0
            llm_avg_ms = 0.0
            memory_store_avg_ms = 0.0
            if miss_indices:
                with Timer() as retrieval_timer:
                    hit_futures = {
                        idx: self.pool.submit(self.hybrid.search, batch[idx].query, self.config.top_k, query_embeddings[idx])
                        for idx in miss_indices
                    }
                    memory_futures = {
                        idx: self.pool.submit(self._load_memory_context_timed, query_embeddings[idx])
                        for idx in miss_indices
                    }
                    total_memory_load_ms = 0.0
                    for idx in miss_indices:
                        hits = hit_futures[idx].result()
                        memory_context, memory_load_ms = memory_futures[idx].result()
                        hits_by_index[idx] = hits
                        contexts_by_index[idx] = self._context_from_arrow(hits, memory_context)
                        total_memory_load_ms += memory_load_ms
                retrieval_avg_ms = retrieval_timer.elapsed_ms / len(miss_indices)
                memory_load_avg_ms = total_memory_load_ms / len(miss_indices)
                if scenario != "retrieval_hybrid":
                    with Timer() as generation_timer:
                        gen_futures = {
                            idx: self.pool.submit(self.llm.generate, batch[idx].query, contexts_by_index[idx])
                            for idx in miss_indices
                        }
                        total_tokens = 0
                        for idx in miss_indices:
                            answer, tokens_generated = gen_futures[idx].result()
                            answers_by_index[idx] = answer
                            tokens_by_index[idx] = tokens_generated
                            total_tokens += tokens_generated
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
            batch_total_avg_ms = 0.0
            if batch:
                # throughput-oriented attribution: amortize batch wall time over the batch size
                batch_total_avg_ms = sum(cache_times) / len(batch)
                if miss_indices:
                    batch_total_avg_ms += retrieval_avg_ms + llm_avg_ms + memory_store_avg_ms
            for i, case in enumerate(batch):
                cache_hit = cache_hits[i]
                answer = cached_answers[i] if cache_hit else answers_by_index.get(i, "")
                rows.append(
                    QueryMetrics(
                        engine=self.name,
                        scenario=scenario,
                        query_id=case.query_id,
                        cache_hit=cache_hit,
                        semantic_cache_lookup_ms=cache_times[i],
                        retrieval_ms=0.0 if cache_hit else retrieval_avg_ms,
                        memory_load_ms=0.0 if cache_hit else memory_load_avg_ms,
                        memory_store_ms=0.0 if cache_hit else memory_store_avg_ms,
                        llm_generation_ms=0.0 if cache_hit or scenario == "retrieval_hybrid" else llm_avg_ms,
                        total_ms=cache_times[i] if cache_hit else batch_total_avg_ms,
                        tokens_generated=0 if cache_hit else tokens_by_index.get(i, 0),
                        answer_preview=answer[:120].replace("\n", " "),
                    )
                )
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
