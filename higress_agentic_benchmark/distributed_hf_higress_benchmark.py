#!/usr/bin/env python3
"""GPU Hugging Face Higress-vs-AAFLOW RAG benchmark.

This benchmark mirrors the GPU HF pipeline/framework benchmarks, but narrows the
comparison to the Higress-agentic benchmark family: HigressRAG, AAFLOW, and
AAFLOW+.  It keeps the semantic workload fixed and uses one Slurm task per GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.benchmark_hf_models import MODEL_ALIASES  # noqa: E402
from framework_rag_pipeline_benchmark.distributed_hf_framework_benchmark import (  # noqa: E402
    FrameworkResult,
    HuggingFaceEngine,
    barrier,
    resolve_rank_corpus,
    run_aaflow,
    run_framework,
    split_document,
    read_file,
)

ENGINE_ORDER = ("HigressRAG", "AAFLOW", "AAFLOW+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    parser.add_argument("--chunks-per-gpu", type=int, default=16000)
    parser.add_argument("--files-per-gpu", type=int, default=64)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--corpus-root")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--aaflow-embed-batch-size", type=int, default=128)
    parser.add_argument("--aaflow-tokenizer-prefetch", action="store_true")
    parser.add_argument("--aaflow-defer-vector-transfer", action="store_true")
    parser.add_argument("--aaflow-tf32", action="store_true")
    parser.add_argument("--aaflow-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--aaflow-token-budget", type=int, default=0)
    parser.add_argument("--aaflow-bucket-sizes", default="64,128")
    parser.add_argument("--upsert-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--generation-samples-per-gpu", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["default", "eager", "sdpa", "flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME", "/scratch/djy8hg/huggingface"))
    parser.add_argument("--engine-filter", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def aggregate(run_dir: Path, world: int, model_name: str, model_id: str, model_load_s: float) -> None:
    rank_rows: list[FrameworkResult] = []
    for path in sorted((run_dir / "rank_results").glob("rank_*.json")):
        if path.name.endswith("_meta.json"):
            continue
        rank_rows.extend(FrameworkResult(**row) for row in json.loads(path.read_text(encoding="utf-8")))

    summary: list[dict[str, object]] = []
    for engine in ENGINE_ORDER:
        group = [row for row in rank_rows if row.framework == engine]
        if not group:
            continue
        total_s = max(row.total_s for row in group)
        chunks = sum(row.chunks for row in group)
        generated_tokens = sum(row.generated_tokens for row in group)
        generate_s = max(row.generate_s for row in group)
        summary.append(
            {
                "model": model_name,
                "model_id": model_id,
                "engine": engine,
                "runtime_mode": "gpu-hf",
                "gpus": world,
                "documents_loaded": sum(row.documents_loaded for row in group),
                "chunks": chunks,
                "generated_prompts": sum(row.generated_prompts for row in group),
                "generated_tokens": generated_tokens,
                "model_load_s_max": model_load_s,
                "load_s": max(row.load_s for row in group),
                "transform_s": max(row.transform_s for row in group),
                "generation_s": generate_s,
                "tokens_per_second": generated_tokens / max(generate_s, 1e-9),
                "embed_s": max(row.embed_s for row in group),
                "upsert_s": max(row.upsert_s for row in group),
                "total_s": total_s,
                "chunks_per_s": chunks / max(total_s, 1e-9),
                "peak_gpu_memory_gb_max": max(row.peak_gpu_memory_gb for row in group),
                "optimization_setup_s_max": max(row.optimization_setup_s for row in group),
            }
        )

    if not summary:
        raise RuntimeError("No rank results were produced")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = (
        ("Engine", 12), ("GPUs", 4), ("Chunks", 7), ("Load(s)", 8), ("Transform(s)", 12),
        ("Embed(s)", 9), ("Upsert(s)", 9), ("Generate(s)", 11), ("Total(s)", 9),
        ("Chunks/s", 9), ("Tok/s", 9), ("GPU GB", 7), ("Setup(s)", 8),
    )
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['engine']:<12.12}  {world:>4}  {row['chunks']:>7}  {row['load_s']:>8.3f}  "
            f"{row['transform_s']:>12.3f}  {row['embed_s']:>9.3f}  {row['upsert_s']:>9.3f}  "
            f"{row['generation_s']:>11.3f}  {row['total_s']:>9.3f}  {row['chunks_per_s']:>9.2f}  "
            f"{row['tokens_per_second']:>9.2f}  {row['peak_gpu_memory_gb_max']:>7.2f}  {row['optimization_setup_s_max']:>8.2f}"
        )


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model
    model_id = MODEL_ALIASES.get(model_name, model_name)

    corpus_dir, corpus_prep_s = resolve_rank_corpus(args, run_dir, rank)
    paths = sorted(corpus_dir.glob("*.txt"))
    engine = HuggingFaceEngine(model_id, args.hf_cache_dir, args.dtype, args.attn_implementation)
    sample = split_document(read_file(paths[0]))[0]
    engine.warmup(sample, args.max_input_tokens)

    compile_chunks: list[str] = []
    for path in paths:
        compile_chunks.extend(split_document(read_file(path)))
        if len(compile_chunks) >= args.aaflow_embed_batch_size:
            compile_chunks = compile_chunks[: args.aaflow_embed_batch_size]
            break
    if len(compile_chunks) < args.aaflow_embed_batch_size:
        raise RuntimeError("Not enough chunks to prepare the AAFLOW compiled embedding graph")
    engine.prepare_compiled_embed(
        compile_chunks,
        args.max_input_tokens,
        use_cudagraphs=True,
        compile_mode=args.aaflow_compile_mode,
    )
    torch.cuda.reset_peak_memory_stats()
    barrier(run_dir, "model_ready", rank, world)

    wanted = {item.strip() for item in args.engine_filter.split(",") if item.strip()}
    engines = [name for name in ENGINE_ORDER if not wanted or name in wanted]
    results: list[FrameworkResult] = []
    for name in engines:
        if name == "HigressRAG":
            result = run_framework(engine, paths, args, rank, "HigressRAG")
            result.framework = "HigressRAG"
            results.append(result)
        elif name == "AAFLOW":
            results.append(run_aaflow(engine, paths, args, rank))
        elif name == "AAFLOW+":
            result = run_aaflow(engine, paths, args, rank)
            result.framework = "AAFLOW+"
            results.append(result)

    results_dir = run_dir / "rank_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"rank_{rank:04d}.json").write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    (results_dir / f"rank_{rank:04d}_meta.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "model": model_name,
                "model_id": model_id,
                "model_load_s": engine.model_load_s,
                "corpus_prep_s": corpus_prep_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    barrier(run_dir, "rank_results_ready", rank, world)
    if rank == 0:
        model_load_s = engine.model_load_s
        meta_paths = sorted((run_dir / "rank_results").glob("rank_*_meta.json"))
        if meta_paths:
            model_load_s = max(json.loads(path.read_text(encoding="utf-8"))["model_load_s"] for path in meta_paths)
        aggregate(run_dir, world, model_name, model_id, model_load_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
