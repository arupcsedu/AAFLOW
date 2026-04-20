#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

from .common import (
    BenchmarkSummary,
    QueryMetrics,
    build_corpus,
    generate_query_cases,
    summarize_metrics,
    write_summary_csv,
)
from .engines import AAFLOWEngine, EngineConfig, HigressRAGEngine, build_llm


def _barrier(run_dir: Path, name: str, rank: int, world: int, poll_s: float = 0.1) -> None:
    barrier_dir = run_dir / "barriers" / name
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"arrive_{rank:06d}").write_text("1", encoding="utf-8")
    release = barrier_dir / "release"
    if rank == 0:
        while len(list(barrier_dir.glob("arrive_*"))) < world:
            time.sleep(poll_s)
        release.write_text("1", encoding="utf-8")
    else:
        while not release.exists():
            time.sleep(poll_s)


def _write_full_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "repeat_index",
        "engine",
        "scenario",
        "count",
        "cache_hit_rate",
        "semantic_cache_lookup_ms_avg",
        "retrieval_ms_avg",
        "memory_load_ms_avg",
        "memory_store_ms_avg",
        "llm_generation_ms_avg",
        "total_ms_avg",
        "total_ms_p50",
        "total_ms_p95",
        "tokens_generated_avg",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_query_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "repeat_index",
        "engine",
        "scenario",
        "query_id",
        "cache_hit",
        "semantic_cache_lookup_ms",
        "retrieval_ms",
        "memory_load_ms",
        "memory_store_ms",
        "llm_generation_ms",
        "total_ms",
        "tokens_generated",
        "answer_preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _median_summary(rows: Sequence[BenchmarkSummary]) -> BenchmarkSummary:
    sample = rows[0]
    return BenchmarkSummary(
        engine=sample.engine,
        scenario=sample.scenario,
        count=int(median(row.count for row in rows)),
        cache_hit_rate=float(median(row.cache_hit_rate for row in rows)),
        semantic_cache_lookup_ms_avg=float(median(row.semantic_cache_lookup_ms_avg for row in rows)),
        retrieval_ms_avg=float(median(row.retrieval_ms_avg for row in rows)),
        memory_load_ms_avg=float(median(row.memory_load_ms_avg for row in rows)),
        memory_store_ms_avg=float(median(row.memory_store_ms_avg for row in rows)),
        llm_generation_ms_avg=float(median(row.llm_generation_ms_avg for row in rows)),
        total_ms_avg=float(median(row.total_ms_avg for row in rows)),
        total_ms_p50=float(median(row.total_ms_p50 for row in rows)),
        total_ms_p95=float(median(row.total_ms_p95 for row in rows)),
        tokens_generated_avg=float(median(row.tokens_generated_avg for row in rows)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Slurm-task Higress vs Agentic benchmark")
    p.add_argument("--benchmark-mode", choices=["default", "fair_parallelism", "fair_parallelism_plus_overlap"], default="fair_parallelism_plus_overlap")
    p.add_argument("--vector-backend", choices=["hash", "faiss"], default="faiss")
    p.add_argument("--physical-workers", type=int, required=True)
    p.add_argument("--non-agentic-dispatch-overhead-ms", type=float, default=0.0)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--file-glob", type=str, default="*.txt")
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overlap-chars", type=int, default=120)
    p.add_argument("--query-count", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--semantic-cache-threshold", type=float, default=0.92)
    p.add_argument("--dense-weight", type=float, default=0.65)
    p.add_argument("--lexical-weight", type=float, default=0.35)
    p.add_argument("--llm-backend", choices=["mock", "hf", "tiny-local"], default="hf")
    p.add_argument("--hf-model", type=str, default="distilgpt2")
    p.add_argument("--hf-device", type=str, default="cpu")
    p.add_argument("--hf-local-files-only", action="store_true")
    p.add_argument("--hf-max-new-tokens", type=int, default=32)
    p.add_argument("--mock-base-latency-ms", type=float, default=20.0)
    p.add_argument("--mock-ms-per-token", type=float, default=0.5)
    p.add_argument("--disable-stm", action="store_true")
    p.add_argument("--disable-ltm", action="store_true")
    p.add_argument("--disable-em", action="store_true")
    p.add_argument("--engine-filter", type=str, default="")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_corpus(
        input_path=args.data_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        file_glob=args.file_glob,
        rank=rank,
        world_size=world,
    )
    if rank == 0:
        (output_dir / "allocation.txt").write_text(
            "\n".join(
                [
                    f"job_id={os.environ.get('SLURM_JOB_ID', '')}",
                    f"physical_workers={args.physical_workers}",
                    f"world_size={world}",
                    f"benchmark_mode={args.benchmark_mode}",
                    f"vector_backend={args.vector_backend}",
                    f"query_count={args.query_count}",
                    f"repeat={args.repeat}",
                    f"data_dir={args.data_dir}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    _barrier(output_dir, "corpus_ready", rank, world)

    query_sets = generate_query_cases(chunks, count=args.query_count) if chunks else {}
    config = EngineConfig(
        benchmark_mode=args.benchmark_mode,
        physical_workers=args.physical_workers,
        vector_backend=args.vector_backend,
        non_agentic_dispatch_overhead_ms=args.non_agentic_dispatch_overhead_ms,
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
    engine_specs = [
        ("HigressRAG", lambda: HigressRAGEngine(name="HigressRAG", chunks=chunks, llm=higress_llm, config=config)),
        ("AAFLOW", lambda: AAFLOWEngine(chunks=chunks, llm=agentic_llm, config=config)),
    ]
    wanted = {item.strip() for item in args.engine_filter.split(",") if item.strip()}
    engine_specs = [item for item in engine_specs if not wanted or item[0] in wanted]

    local_query_rows: List[Dict[str, object]] = []
    repeat_count = max(1, args.repeat)
    for repeat_index in range(repeat_count):
        engines = [factory() for _, factory in engine_specs]
        if chunks:
            warm_cases = query_sets["semantic_cache_lookup"]
            for engine in engines:
                engine.warm_cache(warm_cases)
            for scenario, cases in query_sets.items():
                for case in cases:
                    for engine in engines:
                        row = asdict(engine.run_query(scenario, case))
                        row["repeat_index"] = repeat_index
                        local_query_rows.append(row)

    rank_results_dir = output_dir / "rank_results"
    rank_results_dir.mkdir(parents=True, exist_ok=True)
    (rank_results_dir / f"rank_{rank:06d}.json").write_text(json.dumps(local_query_rows, indent=2), encoding="utf-8")
    _barrier(output_dir, "rank_results_ready", rank, world)

    if rank == 0:
        all_query_rows: List[Dict[str, object]] = []
        for path in sorted(rank_results_dir.glob("rank_*.json")):
            all_query_rows.extend(json.loads(path.read_text(encoding="utf-8")))

        per_repeat_summaries: Dict[int, List[BenchmarkSummary]] = {}
        full_summary_rows: List[Dict[str, object]] = []
        for repeat_index in range(repeat_count):
            repeat_query_rows = [
                QueryMetrics(
                    engine=str(row["engine"]),
                    scenario=str(row["scenario"]),
                    query_id=str(row["query_id"]),
                    cache_hit=bool(row["cache_hit"]),
                    semantic_cache_lookup_ms=float(row["semantic_cache_lookup_ms"]),
                    retrieval_ms=float(row["retrieval_ms"]),
                    memory_load_ms=float(row["memory_load_ms"]),
                    memory_store_ms=float(row["memory_store_ms"]),
                    llm_generation_ms=float(row["llm_generation_ms"]),
                    total_ms=float(row["total_ms"]),
                    tokens_generated=int(row["tokens_generated"]),
                    answer_preview=str(row["answer_preview"]),
                )
                for row in all_query_rows
                if int(row.get("repeat_index", 0)) == repeat_index
            ]
            summaries = summarize_metrics(repeat_query_rows)
            per_repeat_summaries[repeat_index] = summaries
            for summary in summaries:
                full_row = asdict(summary)
                full_row["repeat_index"] = repeat_index
                full_summary_rows.append(full_row)

        grouped: Dict[Tuple[str, str], List[BenchmarkSummary]] = defaultdict(list)
        for repeat_index in range(repeat_count):
            for summary in per_repeat_summaries[repeat_index]:
                grouped[(summary.engine, summary.scenario)].append(summary)
        median_rows = [
            _median_summary(rows)
            for _, rows in sorted(grouped.items(), key=lambda item: item[0])
        ]

        _write_query_metrics_csv(output_dir / "query_metrics.csv", all_query_rows)
        _write_full_summary_csv(output_dir / "full_summary.csv", full_summary_rows)
        write_summary_csv(output_dir / "summary.csv", median_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "summary": [asdict(row) for row in median_rows],
                    "full_summary": full_summary_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Local chunks per rank vary; rank0 chunks={len(chunks)}")
        for row in median_rows:
            print(
                f"{row.engine:10s} {row.scenario:24s} count={row.count:4d} "
                f"cache_hit_rate={row.cache_hit_rate:.2f} total_ms_avg={row.total_ms_avg:.2f} "
                f"retrieval_ms_avg={row.retrieval_ms_avg:.2f} memory_load_ms_avg={row.memory_load_ms_avg:.2f} "
                f"memory_store_ms_avg={row.memory_store_ms_avg:.2f} llm_ms_avg={row.llm_generation_ms_avg:.2f}"
            )
        print(f"Wrote {output_dir / 'summary.csv'}")
        print(f"Wrote {output_dir / 'full_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
