#!/usr/bin/env python3
import argparse
from pathlib import Path

from .common import (
    build_corpus,
    generate_query_cases,
    summarize_metrics,
    write_query_metrics_csv,
    write_summary_csv,
    write_summary_json,
)
from .engines import AgenticRAGEngine, EngineConfig, HigressRAGEngine, build_llm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Higress-style RAG vs Agentic RAG")
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--file-glob", type=str, default="*")
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overlap-chars", type=int, default=120)
    p.add_argument("--query-count", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--semantic-cache-threshold", type=float, default=0.92)
    p.add_argument("--dense-weight", type=float, default=0.65)
    p.add_argument("--lexical-weight", type=float, default=0.35)
    p.add_argument("--llm-backend", choices=["mock", "hf", "tiny-local"], default="mock")
    p.add_argument("--hf-model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--hf-device", type=str, default="cpu")
    p.add_argument("--hf-local-files-only", action="store_true")
    p.add_argument("--hf-max-new-tokens", type=int, default=64)
    p.add_argument("--mock-base-latency-ms", type=float, default=100.0)
    p.add_argument("--mock-ms-per-token", type=float, default=3.0)
    p.add_argument("--disable-stm", action="store_true")
    p.add_argument("--disable-ltm", action="store_true")
    p.add_argument("--disable-em", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_corpus(
        input_path=args.data_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        file_glob=args.file_glob,
    )
    query_sets = generate_query_cases(chunks, count=args.query_count)

    config = EngineConfig(
        top_k=args.top_k,
        semantic_cache_threshold=args.semantic_cache_threshold,
        dense_weight=args.dense_weight,
        lexical_weight=args.lexical_weight,
        enable_stm=not args.disable_stm,
        enable_ltm=not args.disable_ltm,
        enable_em=not args.disable_em,
    )
    corpus_texts = [chunk.text for chunk in chunks]
    higress_llm = build_llm(
        args.llm_backend,
        corpus_texts,
        args.hf_model,
        args.hf_device,
        args.hf_local_files_only,
        args.hf_max_new_tokens,
        args.mock_base_latency_ms,
        args.mock_ms_per_token,
    )
    agentic_llm = build_llm(
        args.llm_backend,
        corpus_texts,
        args.hf_model,
        args.hf_device,
        args.hf_local_files_only,
        args.hf_max_new_tokens,
        args.mock_base_latency_ms,
        args.mock_ms_per_token,
    )

    higress = HigressRAGEngine(name="HigressRAG", chunks=chunks, llm=higress_llm, config=config)
    agentic = AgenticRAGEngine(chunks=chunks, llm=agentic_llm, config=config)

    # Warm semantic cache using the same semantic-cache scenario queries.
    warm_cases = query_sets["semantic_cache_lookup"]
    higress.warm_cache(warm_cases)
    agentic.warm_cache(warm_cases)

    query_rows = []
    for scenario, cases in query_sets.items():
        for case in cases:
            query_rows.append(higress.run_query(scenario, case))
            query_rows.append(agentic.run_query(scenario, case))

    summaries = summarize_metrics(query_rows)

    write_query_metrics_csv(output_dir / "query_metrics.csv", query_rows)
    write_summary_csv(output_dir / "summary.csv", summaries)
    write_summary_json(output_dir / "summary.json", query_rows, summaries)

    print(f"Corpus chunks: {len(chunks)}")
    print(f"Wrote {output_dir / 'query_metrics.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print()
    for row in summaries:
        print(
            f"{row.engine:12s} {row.scenario:24s} count={row.count:2d} "
            f"cache_hit_rate={row.cache_hit_rate:.2f} total_ms_avg={row.total_ms_avg:.2f} "
            f"retrieval_ms_avg={row.retrieval_ms_avg:.2f} memory_load_ms_avg={row.memory_load_ms_avg:.2f} "
            f"memory_store_ms_avg={row.memory_store_ms_avg:.2f} llm_ms_avg={row.llm_generation_ms_avg:.2f}"
        )


if __name__ == "__main__":
    main()
