#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from .common import BenchmarkConfig, PipelineMetrics, ensure_synthetic_corpus, median_metrics, write_metrics
from .runners import RUNNERS


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


def _make_rank_shard_dir(run_dir: Path, rank: int, local_files: List[Path]) -> Path:
    shard_dir = run_dir / "rank_shards" / f"rank_{rank:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for src in local_files:
        dst = shard_dir / src.name
        if dst.exists():
            continue
        os.symlink(src, dst)
    return shard_dir


def _aggregate_rows(rows: List[Dict[str, object]], framework: str) -> PipelineMetrics:
    return PipelineMetrics(
        framework=framework,
        runtime_mode="distributed",
        documents_loaded=sum(int(r["documents_loaded"]) for r in rows),
        chunks=sum(int(r["chunks"]) for r in rows),
        generated_prompts=sum(int(r["generated_prompts"]) for r in rows),
        generated_tokens=sum(int(r["generated_tokens"]) for r in rows),
        load_s=max(float(r["load_s"]) for r in rows) if rows else 0.0,
        transform_s=max(float(r["transform_s"]) for r in rows) if rows else 0.0,
        generation_s=max(float(r["generation_s"]) for r in rows) if rows else 0.0,
        tokens_per_second=(
            sum(float(r["generated_tokens"]) for r in rows) / max(float(max(float(r["generation_s"]) for r in rows)), 1e-9)
            if rows
            else 0.0
        ),
        embed_s=max(float(r["embed_s"]) for r in rows) if rows else 0.0,
        upsert_s=max(float(r["upsert_s"]) for r in rows) if rows else 0.0,
        total_s=max(float(r["total_s"]) for r in rows) if rows else 0.0,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Slurm-task framework RAG benchmark")
    p.add_argument("--benchmark-mode", choices=["default", "fair_parallelism", "fair_parallelism_plus_overlap"], default="fair_parallelism")
    p.add_argument("--embedding-backend", choices=["fake", "transformers"], default="fake")
    p.add_argument("--generation-backend", choices=["fake", "transformers"], default="fake")
    p.add_argument("--vector-backend", choices=["fake", "chroma", "faiss"], default="faiss")
    p.add_argument("--generation-cost-mode", choices=["linear", "fixed"], default="linear")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--generation-model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--chroma-path", type=str, default=None)
    p.add_argument("--faiss-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--file-glob", type=str, default="*.txt")
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--files", type=int, default=64)
    p.add_argument("--node-chars", type=int, default=900)
    p.add_argument("--chunk-tokens", type=int, default=180)
    p.add_argument("--chunk-overlap", type=int, default=20)
    p.add_argument("--generation-samples", type=int, default=128)
    p.add_argument("--generation-output-tokens", type=int, default=64)
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--transform-workers", type=int, default=8)
    p.add_argument("--async-workers", type=int, default=16)
    p.add_argument("--physical-workers", type=int, required=True)
    p.add_argument("--embed-workers", type=int, default=None)
    p.add_argument("--upsert-workers", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--upsert-batch-size", type=int, default=64)
    p.add_argument("--agentic-queue-size", type=int, default=0)
    p.add_argument("--agentic-upsert-coalesce-target", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--embed-overhead-ms", type=float, default=20.0)
    p.add_argument("--embed-per-item-ms", type=float, default=1.5)
    p.add_argument("--upsert-overhead-ms", type=float, default=8.0)
    p.add_argument("--upsert-per-item-ms", type=float, default=0.5)
    p.add_argument("--generate-overhead-ms", type=float, default=30.0)
    p.add_argument("--generate-ms-per-token", type=float, default=0.6)
    p.add_argument("--framework-filter", type=str, default="")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = BenchmarkConfig(
        benchmark_mode=args.benchmark_mode,
        embedding_backend=args.embedding_backend,
        generation_backend=args.generation_backend,
        vector_backend=args.vector_backend,
        generation_cost_mode=args.generation_cost_mode,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        chroma_path=str((out_dir / "chroma_store" / f"rank_{rank:06d}")) if args.vector_backend == "chroma" else args.chroma_path,
        faiss_path=str((out_dir / "faiss_store" / f"rank_{rank:06d}")) if args.vector_backend == "faiss" else args.faiss_path,
        data_dir=args.data_dir,
        file_glob=args.file_glob,
        nodes=args.nodes,
        files=args.files,
        node_chars=args.node_chars,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        generation_samples=args.generation_samples,
        generation_output_tokens=args.generation_output_tokens,
        load_workers=min(max(1, args.load_workers), max(1, args.physical_workers)),
        transform_workers=min(max(1, args.transform_workers), max(1, args.physical_workers)),
        async_workers=min(max(1, args.async_workers), max(1, args.physical_workers)),
        physical_workers=args.physical_workers,
        embed_workers=None if args.embed_workers is None else min(max(1, args.embed_workers), max(1, args.physical_workers)),
        upsert_workers=None if args.upsert_workers is None else min(max(1, args.upsert_workers), max(1, args.physical_workers)),
        embed_dim=args.embed_dim,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        agentic_queue_size=args.agentic_queue_size,
        agentic_upsert_coalesce_target=args.agentic_upsert_coalesce_target,
        seed=args.seed,
        embed_overhead_ms=args.embed_overhead_ms,
        embed_per_item_ms=args.embed_per_item_ms,
        upsert_overhead_ms=args.upsert_overhead_ms,
        upsert_per_item_ms=args.upsert_per_item_ms,
        generate_overhead_ms=args.generate_overhead_ms,
        generate_ms_per_token=args.generate_ms_per_token,
    )

    if rank == 0:
        ensure_synthetic_corpus(base_config)
    _barrier(out_dir, "corpus_ready", rank, world)

    all_files = sorted(Path(args.data_dir).glob(args.file_glob))[: args.files]
    local_files = [p for i, p in enumerate(all_files) if i % world == rank]
    shard_dir = _make_rank_shard_dir(out_dir, rank, local_files)
    shard_nodes = args.nodes // world + (1 if rank < (args.nodes % world) else 0)
    local_config = BenchmarkConfig(
        **{
            **base_config.__dict__,
            "data_dir": str(shard_dir),
            "files": len(local_files),
            "nodes": shard_nodes,
        }
    )

    wanted = {item.strip() for item in args.framework_filter.split(",") if item.strip()}
    selected_runners = [runner_cls for runner_cls in RUNNERS if not wanted or runner_cls.framework in wanted]

    local_rows: List[Dict[str, object]] = []
    repeat_count = max(1, args.repeat)
    for repeat_index in range(repeat_count):
        for runner_cls in selected_runners:
            runner = runner_cls(local_config)
            row = runner.run() if local_files else PipelineMetrics(
                framework=runner_cls.framework,
                runtime_mode="distributed",
                documents_loaded=0,
                chunks=0,
                generated_prompts=0,
                generated_tokens=0,
                load_s=0.0,
                transform_s=0.0,
                generation_s=0.0,
                tokens_per_second=0.0,
                embed_s=0.0,
                upsert_s=0.0,
                total_s=0.0,
            )
            full_row = row.__dict__.copy()
            full_row["repeat_index"] = repeat_index
            local_rows.append(full_row)

    rank_results_dir = out_dir / "rank_results"
    rank_results_dir.mkdir(parents=True, exist_ok=True)
    (rank_results_dir / f"rank_{rank:06d}.json").write_text(json.dumps(local_rows, indent=2), encoding="utf-8")
    _barrier(out_dir, "rank_results_ready", rank, world)

    if rank == 0:
        aggregated: List[PipelineMetrics] = []
        by_framework_repeat: Dict[str, Dict[int, List[Dict[str, object]]]] = {
            runner.framework: {repeat_index: [] for repeat_index in range(repeat_count)} for runner in selected_runners
        }
        full_rows: List[Dict[str, object]] = []
        for path in sorted(rank_results_dir.glob("rank_*.json")):
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                framework = row["framework"]
                repeat_index = int(row.get("repeat_index", 0))
                if framework in by_framework_repeat and repeat_index in by_framework_repeat[framework]:
                    by_framework_repeat[framework][repeat_index].append(row)
        for runner in selected_runners:
            repeat_rows: List[PipelineMetrics] = []
            for repeat_index in range(repeat_count):
                row = _aggregate_rows(by_framework_repeat[runner.framework][repeat_index], runner.framework)
                repeat_rows.append(row)
                full_row = row.__dict__.copy()
                full_row["repeat_index"] = repeat_index
                full_rows.append(full_row)
            row = median_metrics(repeat_rows)
            aggregated.append(row)
            print(
                f"{row.framework:10s} mode={row.runtime_mode:11s} docs={row.documents_loaded:6d} chunks={row.chunks:8d} "
                f"load={row.load_s:.3f}s transform={row.transform_s:.3f}s gen={row.generation_s:.3f}s "
                f"tok/s={row.tokens_per_second:.1f} embed={row.embed_s:.3f}s upsert={row.upsert_s:.3f}s total={row.total_s:.3f}s"
            )
        write_metrics(out_dir, aggregated, full_rows=full_rows)
        print(f"Wrote {out_dir / 'summary.csv'}")
        print(f"Wrote {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
