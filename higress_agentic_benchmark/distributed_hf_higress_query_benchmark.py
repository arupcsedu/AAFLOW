#!/usr/bin/env python3
"""GPU HF query-scenario benchmark for HigressRAG vs AAFLOW+.

This preserves the historical Higress benchmark metrics:
semantic_cache_lookup, retrieval_hybrid, llm_generation, and
non_cached_complex_query, while using local HuggingFace Llama/Mistral generation
on one GPU per Slurm task.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.benchmark_hf_models import MODEL_ALIASES, resolve_hub_cache  # noqa: E402
from higress_agentic_benchmark.common import (  # noqa: E402
    BenchmarkSummary,
    QueryMetrics,
    build_corpus,
    generate_query_cases,
    summarize_metrics,
    write_summary_csv,
)
from higress_agentic_benchmark.engines import (  # noqa: E402
    AAFLOWPlusEngine,
    EngineConfig,
    HigressRAGEngine,
)


class GPUHFLLM:
    def __init__(
        self,
        model_id: str,
        cache_dir: str,
        dtype_name: str,
        attn_implementation: str,
        max_new_tokens: int,
        max_input_tokens: int,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda")
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.dtype = getattr(torch, dtype_name)
        self.lock = threading.Lock()
        hub_cache = resolve_hub_cache(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hub_cache, local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs = {"cache_dir": hub_cache, "local_files_only": True, "dtype": self.dtype}
        if attn_implementation != "default":
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(self.device)
        self.model.eval()
        torch.cuda.synchronize()

    def generate(self, query: str, context: str) -> Tuple[str, int]:
        prompt = f"Use the provided context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        with self.lock, torch.inference_mode():
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            ).to(self.device)
            torch.cuda.synchronize()
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
        generated = outputs[0, encoded["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, int(generated.numel())

    def generate_batch(self, items: Sequence[Tuple[str, str]]) -> List[Tuple[str, int]]:
        if not items:
            return []
        prompts = [
            f"Use the provided context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
            for query, context in items
        ]
        with self.lock, torch.inference_mode():
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_width = encoded["input_ids"].shape[1]
            torch.cuda.synchronize()
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
        rows: List[Tuple[str, int]] = []
        for row_idx in range(len(items)):
            generated = outputs[row_idx, prompt_width:]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            rows.append((answer, int(generated.numel())))
        return rows


def barrier(run_dir: Path, name: str, rank: int, world: int, poll_s: float = 0.1) -> None:
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


def write_full_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "repeat_index", "engine", "scenario", "count", "cache_hit_rate",
        "semantic_cache_lookup_ms_avg", "retrieval_ms_avg", "memory_load_ms_avg",
        "memory_store_ms_avg", "llm_generation_ms_avg", "total_ms_avg",
        "total_ms_p50", "total_ms_p95", "tokens_generated_avg",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if isinstance(payload.get("hit_ids"), list):
                payload["hit_ids"] = "|".join(str(item) for item in payload["hit_ids"])
            writer.writerow(payload)


def write_query_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "repeat_index", "engine", "scenario", "query_id", "cache_hit",
        "semantic_cache_lookup_ms", "retrieval_ms", "memory_load_ms",
        "memory_store_ms", "llm_generation_ms", "total_ms", "tokens_generated",
        "answer_preview", "hit_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_retrieval_quality(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[int, str, str], Dict[str, List[str]]] = defaultdict(dict)
    for row in rows:
        engine = str(row.get("engine", ""))
        if engine not in {"HigressRAG", "AAFLOW+"}:
            continue
        hit_ids = row.get("hit_ids", [])
        if isinstance(hit_ids, str):
            hits = [item for item in hit_ids.split("|") if item]
        else:
            hits = [str(item) for item in hit_ids]
        if not hits:
            continue
        key = (int(row.get("repeat_index", 0)), str(row.get("scenario", "")), str(row.get("query_id", "")))
        by_key[key][engine] = hits

    grouped: Dict[Tuple[int, str], List[Tuple[float, float, int, int, int]]] = defaultdict(list)
    for (repeat_index, scenario, _query_id), engine_hits in by_key.items():
        baseline = engine_hits.get("HigressRAG")
        candidate = engine_hits.get("AAFLOW+")
        if not baseline or not candidate:
            continue
        baseline_set = set(baseline)
        candidate_set = set(candidate)
        intersection = len(baseline_set & candidate_set)
        recall = intersection / len(baseline_set) if baseline_set else 0.0
        jaccard = intersection / len(baseline_set | candidate_set) if (baseline_set or candidate_set) else 0.0
        grouped[(repeat_index, scenario)].append((recall, jaccard, intersection, len(baseline_set), len(candidate_set)))

    quality_rows: List[Dict[str, object]] = []
    for (repeat_index, scenario), vals in sorted(grouped.items()):
        quality_rows.append({
            "repeat_index": repeat_index,
            "scenario": scenario,
            "count": len(vals),
            "topk_recall_avg": mean(v[0] for v in vals),
            "topk_jaccard_avg": mean(v[1] for v in vals),
            "overlap_count_avg": mean(v[2] for v in vals),
            "baseline_k_avg": mean(v[3] for v in vals),
            "candidate_k_avg": mean(v[4] for v in vals),
        })
    return quality_rows


def write_retrieval_quality_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "repeat_index", "scenario", "count", "topk_recall_avg", "topk_jaccard_avg",
        "overlap_count_avg", "baseline_k_avg", "candidate_k_avg",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def median_summary(rows: Sequence[BenchmarkSummary]) -> BenchmarkSummary:
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--file-glob", default="*.txt")
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overlap-chars", type=int, default=120)
    p.add_argument("--query-count", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--benchmark-mode", choices=["default", "fair_parallelism", "fair_parallelism_plus_overlap"], default="fair_parallelism_plus_overlap")
    p.add_argument("--vector-backend", choices=["hash", "faiss"], default="faiss")
    p.add_argument("--physical-workers", type=int, default=2)
    p.add_argument("--aaflow-plus-batch-size", type=int, default=0)
    p.add_argument("--aaflow-plus-dense-candidates", type=int, default=0)
    p.add_argument("--non-agentic-dispatch-overhead-ms", type=float, default=0.0)
    p.add_argument("--semantic-cache-threshold", type=float, default=0.92)
    p.add_argument("--dense-weight", type=float, default=0.65)
    p.add_argument("--lexical-weight", type=float, default=0.35)
    p.add_argument("--disable-stm", action="store_true")
    p.add_argument("--disable-ltm", action="store_true")
    p.add_argument("--disable-em", action="store_true")
    p.add_argument("--engine-filter", default="HigressRAG,AAFLOW+")
    p.add_argument("--scenario-filter", default="")
    p.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME", "/scratch/djy8hg/huggingface"))
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn-implementation", choices=["default", "eager", "sdpa", "flash_attention_2"], default="flash_attention_2")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = MODEL_ALIASES.get(args.model, args.model)

    chunks = build_corpus(args.data_dir, args.max_chars, args.overlap_chars, args.file_glob, rank=rank, world_size=world)
    if rank == 0:
        (output_dir / "allocation.txt").write_text(
            "\n".join([
                f"job_id={os.environ.get('SLURM_JOB_ID', '')}",
                f"model={args.model}",
                f"model_id={model_id}",
                f"world_size={world}",
                f"query_count={args.query_count}",
                f"repeat={args.repeat}",
                f"data_dir={args.data_dir}",
                f"attn_implementation={args.attn_implementation}",
                f"aaflow_plus_batch_size={args.aaflow_plus_batch_size}",
                f"aaflow_plus_dense_candidates={args.aaflow_plus_dense_candidates}",
            ]) + "\n",
            encoding="utf-8",
        )
    barrier(output_dir, "corpus_ready", rank, world)

    query_sets = generate_query_cases(chunks, count=args.query_count) if chunks else {}
    scenario_filter = args.scenario_filter.replace(":", ",")
    wanted_scenarios = {item.strip() for item in scenario_filter.split(",") if item.strip()}
    if wanted_scenarios:
        query_sets = {name: cases for name, cases in query_sets.items() if name in wanted_scenarios}
    config = EngineConfig(
        benchmark_mode=args.benchmark_mode,
        physical_workers=args.physical_workers,
        aaflow_plus_batch_size=args.aaflow_plus_batch_size,
        aaflow_plus_dense_candidates=args.aaflow_plus_dense_candidates,
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
    llm = GPUHFLLM(model_id, args.hf_cache_dir, args.dtype, args.attn_implementation, args.max_new_tokens, args.max_input_tokens)
    engine_specs = [
        ("HigressRAG", lambda: HigressRAGEngine(name="HigressRAG", chunks=chunks, llm=llm, config=config)),
        ("AAFLOW+", lambda: AAFLOWPlusEngine(chunks=chunks, llm=llm, config=config)),
    ]
    wanted = {item.strip() for item in args.engine_filter.split(",") if item.strip()}
    engine_specs = [item for item in engine_specs if not wanted or item[0] in wanted]

    local_query_rows: List[Dict[str, object]] = []
    for repeat_index in range(max(1, args.repeat)):
        engines = [factory() for _, factory in engine_specs]
        if chunks:
            warm_cases = query_sets.get("semantic_cache_lookup", [])
            if warm_cases:
                for engine in engines:
                    engine.warm_cache(warm_cases)
            for scenario, cases in query_sets.items():
                for engine in engines:
                    for row in engine.run_queries(scenario, cases):
                        payload = asdict(row)
                        payload["repeat_index"] = repeat_index
                        local_query_rows.append(payload)

    rank_results_dir = output_dir / "rank_results"
    rank_results_dir.mkdir(parents=True, exist_ok=True)
    (rank_results_dir / f"rank_{rank:06d}.json").write_text(json.dumps(local_query_rows, indent=2), encoding="utf-8")
    barrier(output_dir, "rank_results_ready", rank, world)

    if rank == 0:
        all_query_rows: List[Dict[str, object]] = []
        for path in sorted(rank_results_dir.glob("rank_*.json")):
            all_query_rows.extend(json.loads(path.read_text(encoding="utf-8")))
        per_repeat: Dict[int, List[BenchmarkSummary]] = {}
        full_summary_rows: List[Dict[str, object]] = []
        for repeat_index in range(max(1, args.repeat)):
            repeat_rows = [
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
                    hit_ids=list(row.get("hit_ids", [])),
                )
                for row in all_query_rows
                if int(row.get("repeat_index", 0)) == repeat_index
            ]
            summaries = summarize_metrics(repeat_rows)
            per_repeat[repeat_index] = summaries
            for summary in summaries:
                full = asdict(summary)
                full["repeat_index"] = repeat_index
                full_summary_rows.append(full)

        grouped: Dict[Tuple[str, str], List[BenchmarkSummary]] = defaultdict(list)
        for summaries in per_repeat.values():
            for summary in summaries:
                grouped[(summary.engine, summary.scenario)].append(summary)
        median_rows = [median_summary(rows) for _, rows in sorted(grouped.items(), key=lambda item: item[0])]
        quality_rows = compute_retrieval_quality(all_query_rows)
        write_query_metrics_csv(output_dir / "query_metrics.csv", all_query_rows)
        write_retrieval_quality_csv(output_dir / "retrieval_quality.csv", quality_rows)
        write_full_summary_csv(output_dir / "full_summary.csv", full_summary_rows)
        write_summary_csv(output_dir / "summary.csv", median_rows)
        (output_dir / "summary.json").write_text(
            json.dumps({"summary": [asdict(row) for row in median_rows], "retrieval_quality": quality_rows, "query_metrics": all_query_rows}, indent=2),
            encoding="utf-8",
        )
        for row in median_rows:
            print(
                f"{row.engine:12s} {row.scenario:24s} count={row.count:4d} "
                f"cache_hit_rate={row.cache_hit_rate:.2f} total_ms_avg={row.total_ms_avg:.2f} "
                f"retrieval_ms_avg={row.retrieval_ms_avg:.2f} memory_load_ms_avg={row.memory_load_ms_avg:.2f} "
                f"memory_store_ms_avg={row.memory_store_ms_avg:.2f} llm_ms_avg={row.llm_generation_ms_avg:.2f}"
            )
        for row in quality_rows:
            print(
                f"retrieval_quality {row['scenario']:24s} count={int(row['count']):4d} "
                f"topk_recall={float(row['topk_recall_avg']):.3f} "
                f"jaccard={float(row['topk_jaccard_avg']):.3f} "
                f"overlap={float(row['overlap_count_avg']):.2f}"
            )

        summary_by_key = {(row.engine, row.scenario): row for row in median_rows}
        quality_by_scenario = {str(row["scenario"]): row for row in quality_rows}
        comparable_scenarios = [
            scenario for scenario in ("retrieval_hybrid", "llm_generation", "non_cached_complex_query")
            if ("HigressRAG", scenario) in summary_by_key and ("AAFLOW+", scenario) in summary_by_key
        ]
        if comparable_scenarios:
            print("")
            print("Comparison Table")
            print("Scenario                 HigressRAG ms  AAFLOW+ ms  Improvement  Top-k Recall  Jaccard  Avg Overlap")
            print("-----------------------  ------------  ----------  -----------  ------------  -------  -----------")
            for scenario in comparable_scenarios:
                higress = summary_by_key[("HigressRAG", scenario)]
                aaflow = summary_by_key[("AAFLOW+", scenario)]
                improvement = (higress.total_ms_avg - aaflow.total_ms_avg) / higress.total_ms_avg * 100.0 if higress.total_ms_avg else 0.0
                quality = quality_by_scenario.get(scenario, {})
                recall = float(quality.get("topk_recall_avg", 0.0))
                jaccard = float(quality.get("topk_jaccard_avg", 0.0))
                overlap = float(quality.get("overlap_count_avg", 0.0))
                baseline_k = float(quality.get("baseline_k_avg", 0.0))
                print(
                    f"{scenario:23s}  {higress.total_ms_avg:12.2f}  {aaflow.total_ms_avg:10.2f}  "
                    f"{improvement:10.2f}%  {recall:12.3f}  {jaccard:7.3f}  {overlap:.2f} / {baseline_k:.0f}"
                )
        print(f"Wrote {output_dir / 'summary.csv'}")
        print(f"Wrote {output_dir / 'full_summary.csv'}")
        print(f"Wrote {output_dir / 'retrieval_quality.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
