#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from preprocessing import clean_text, chunk_text

from .common import CorpusChunk, HashingEmbedder, HybridRetriever, tokenize


logger = logging.getLogger(__name__)
_WORKER_SEARCHER = None
_WORKER_TOP_K = 0

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "were", "which",
    "their", "about", "would", "there", "after", "before", "into", "when", "what",
    "where", "while", "whose", "them", "they", "been", "also", "than", "then",
    "because", "could", "should", "such", "these", "those", "your", "into", "over",
    "under", "between", "during", "through", "being", "it", "is", "a", "an", "of",
    "to", "in", "on", "as", "by", "or", "at", "be", "are", "was", "if", "but",
}


@dataclass
class RetrievalCase:
    case_id: str
    seed_query: str
    followup_query: str
    expected_chunk_id: str
    source_file: str
    keywords: List[str]


@dataclass
class RetrievalResult:
    engine: str
    case_id: str
    expected_chunk_id: str
    retrieved_chunk_ids: str
    top1_correct: int
    hit_at_k: int
    reciprocal_rank: float
    effective_query: str


def _top_keywords(text: str, k: int = 4) -> List[str]:
    freqs = Counter(tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) >= 5)
    if not freqs:
        freqs = Counter(tok for tok in tokenize(text) if len(tok) >= 4)
    return [token for token, _ in freqs.most_common(k)]


def _build_corpus(input_path: str, file_glob: str, max_chars: int, overlap_chars: int, sample_files: int) -> List[CorpusChunk]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(p for p in path.glob(file_glob) if p.is_file())
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError("Input path does not exist: %s" % input_path)
    if sample_files > 0:
        files = files[:sample_files]

    chunks: List[CorpusChunk] = []
    for doc_idx, file_path in enumerate(files):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(text)
        if not cleaned:
            continue
        for chunk_idx, chunk in enumerate(chunk_text(cleaned, max_chars=max_chars, overlap_chars=overlap_chars)):
            chunks.append(
                CorpusChunk(
                    chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                    text=chunk,
                    metadata={"source_file": str(file_path)},
                )
            )
    return chunks


def _select_cases(chunks: Sequence[CorpusChunk], num_cases: int) -> List[RetrievalCase]:
    cases: List[RetrievalCase] = []
    used_sources = set()
    for idx, chunk in enumerate(chunks):
        words = chunk.text.split()
        if len(words) < 24:
            continue
        keywords = _top_keywords(chunk.text, k=4)
        if len(keywords) < 2:
            continue
        source_file = chunk.metadata.get("source_file", "")
        key = (source_file, tuple(keywords[:2]))
        if key in used_sources:
            continue
        used_sources.add(key)
        preview = " ".join(words[:18])
        seed_query = (
            f"I am reading a passage about {keywords[0]} and {keywords[1]}. "
            f"Use this text as the current topic: {preview}"
        )
        followup_query = (
            "What else does it say about this topic? "
            "Answer the follow-up using the same subject as the previous turn."
        )
        cases.append(
            RetrievalCase(
                case_id=f"case_{idx}",
                seed_query=seed_query,
                followup_query=followup_query,
                expected_chunk_id=chunk.chunk_id,
                source_file=source_file,
                keywords=keywords[:4],
            )
        )
        if len(cases) >= num_cases:
            break
    if not cases:
        raise RuntimeError("Could not build conversational retrieval cases from the input corpus.")
    return cases


def _rank_hit_ids(hit_ids: Sequence[str], expected_chunk_id: str) -> Tuple[int, int, float]:
    top1 = 1 if hit_ids and hit_ids[0] == expected_chunk_id else 0
    hit_k = 0
    rr = 0.0
    for rank, chunk_id in enumerate(hit_ids, start=1):
        if chunk_id == expected_chunk_id:
            hit_k = 1
            rr = 1.0 / rank
            break
    return top1, hit_k, rr


class FastHybridSearcher:
    def __init__(self, chunks: Sequence[CorpusChunk]):
        self.chunks = list(chunks)
        self.retriever = HybridRetriever(self.chunks, HashingEmbedder())
        self.embedder = self.retriever.embedder
        self.embeddings = self.retriever.chunk_embeddings
        self.lexical = self.retriever.lexical
        self.dense_weight = self.retriever.dense_weight
        self.lexical_weight = self.retriever.lexical_weight

    def search_ids(self, query: str, top_k: int) -> List[str]:
        query_embedding = self.embedder.embed_query(query)
        dense_scores = self.embeddings @ query_embedding.astype(np.float32)
        dense_max = float(np.max(np.abs(dense_scores))) if dense_scores.size else 1.0
        if dense_max == 0.0:
            dense_max = 1.0
        hybrid = self.dense_weight * (dense_scores / dense_max)

        lexical_scores = self.lexical.score_query(query)
        if lexical_scores:
            lexical_max = max(abs(v) for v in lexical_scores.values()) or 1.0
            lexical_scale = self.lexical_weight / lexical_max
            for idx, score in lexical_scores.items():
                hybrid[idx] += score * lexical_scale

        k = min(top_k, hybrid.shape[0])
        if k <= 0:
            return []
        candidate_idx = np.argpartition(-hybrid, k - 1)[:k]
        ordered_idx = candidate_idx[np.argsort(-hybrid[candidate_idx])]
        return [self.chunks[idx].chunk_id for idx in ordered_idx]


def _init_worker(chunks: Sequence[CorpusChunk], top_k: int) -> None:
    global _WORKER_SEARCHER, _WORKER_TOP_K
    prepared = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = chunk.chunk_id
        prepared.append(CorpusChunk(chunk_id=chunk.chunk_id, text=chunk.text, metadata=metadata))
    _WORKER_SEARCHER = FastHybridSearcher(prepared)
    _WORKER_TOP_K = top_k


def _process_case(case: RetrievalCase) -> List[RetrievalResult]:
    global _WORKER_SEARCHER, _WORKER_TOP_K
    assert _WORKER_SEARCHER is not None
    results: List[RetrievalResult] = []
    for engine_name, effective_query in (
        ("HigressRAG", case.followup_query),
        ("AgenticDRC", f"{case.seed_query}\nFollow-up: {case.followup_query}"),
    ):
        hit_ids = _WORKER_SEARCHER.search_ids(effective_query, top_k=_WORKER_TOP_K)
        top1, hit_k, rr = _rank_hit_ids(hit_ids, case.expected_chunk_id)
        results.append(
            RetrievalResult(
                engine=engine_name,
                case_id=case.case_id,
                expected_chunk_id=case.expected_chunk_id,
                retrieved_chunk_ids="|".join(hit_ids),
                top1_correct=top1,
                hit_at_k=hit_k,
                reciprocal_rank=rr,
                effective_query=effective_query,
            )
        )
    return results


def _run_benchmark(
    chunks: Sequence[CorpusChunk],
    cases: Sequence[RetrievalCase],
    top_k: int,
    progress_every: int,
    num_workers: int,
) -> List[RetrievalResult]:
    results: List[RetrievalResult] = []
    total_steps = len(cases) * 2

    if num_workers <= 1:
        _init_worker(chunks, top_k)
        for case_idx, case in enumerate(cases, start=1):
            case_results = _process_case(case)
            results.extend(case_results)
            step = case_idx * 2
            if progress_every > 0 and (step % progress_every == 0 or step == total_steps):
                logger.info("Progress %d/%d (case %d/%d)", step, total_steps, case_idx, len(cases))
        return results

    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    chunksize = max(1, len(cases) // max(num_workers * 4, 1))
    completed_steps = 0
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(list(chunks), top_k),
    ) as pool:
        for case_idx, case_results in enumerate(pool.imap_unordered(_process_case, cases, chunksize=chunksize), start=1):
            results.extend(case_results)
            completed_steps += len(case_results)
            if progress_every > 0 and (completed_steps % progress_every == 0 or completed_steps == total_steps):
                logger.info(
                    "Progress %d/%d (completed cases %d/%d)",
                    completed_steps,
                    total_steps,
                    case_idx,
                    len(cases),
                )
    return results


def _summarize(results: Sequence[RetrievalResult]) -> List[Dict[str, float]]:
    groups: Dict[str, List[RetrievalResult]] = defaultdict(list)
    for row in results:
        groups[row.engine].append(row)
    summaries: List[Dict[str, float]] = []
    for engine, rows in sorted(groups.items()):
        summaries.append(
            {
                "engine": engine,
                "count": len(rows),
                "top1_accuracy": mean(row.top1_correct for row in rows),
                "hit_at_k_accuracy": mean(row.hit_at_k for row in rows),
                "mrr": mean(row.reciprocal_rank for row in rows),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conversational retrieval accuracy benchmark for HigressRAG vs AgenticDRC")
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "wikitext_eval"))
    p.add_argument("--file-glob", type=str, default="*")
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overlap-chars", type=int, default=120)
    p.add_argument("--sample-files", type=int, default=0, help="If >0, only load the first N source files/documents.")
    p.add_argument("--num-cases", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=100, help="Log progress every N engine-query evaluations.")
    p.add_argument("--num-workers", type=int, default=0, help="CPU worker processes. 0 means auto-detect.")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "conversational_outputs"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s", args.data_dir)
    chunks = _build_corpus(
        input_path=args.data_dir,
        file_glob=args.file_glob,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        sample_files=args.sample_files,
    )
    logger.info("Built %d chunks", len(chunks))
    cases = _select_cases(chunks, num_cases=args.num_cases)
    logger.info("Selected %d retrieval cases", len(cases))
    num_workers = args.num_workers if args.num_workers > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    num_workers = max(1, num_workers)
    logger.info("Using %d worker process(es)", num_workers)
    results = _run_benchmark(
        chunks,
        cases,
        top_k=args.top_k,
        progress_every=args.progress_every,
        num_workers=num_workers,
    )
    summaries = _summarize(results)

    _write_csv(output_dir / "retrieval_cases.csv", [asdict(case) for case in cases])
    _write_csv(output_dir / "retrieval_results.csv", [asdict(row) for row in results])
    _write_csv(output_dir / "retrieval_summary.csv", summaries)
    (output_dir / "retrieval_summary.json").write_text(
        json.dumps(
            {
                "cases": [asdict(case) for case in cases],
                "results": [asdict(row) for row in results],
                "summary": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Cases: {len(cases)}")
    for row in summaries:
        print(
            f"{row['engine']:10s} top1={row['top1_accuracy']:.3f} "
            f"hit@{args.top_k}={row['hit_at_k_accuracy']:.3f} mrr={row['mrr']:.3f}"
        )
    print(f"Wrote {output_dir / 'retrieval_summary.csv'}")


if __name__ == "__main__":
    main()
