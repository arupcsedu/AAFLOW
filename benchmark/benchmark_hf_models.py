#!/usr/bin/env python3
"""Benchmark real Hugging Face causal models in an AAFLOW-style RAG pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ALIASES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
CHUNK_DELIMITER = "\n<AAFLOW_CHUNK>\n"


@dataclass
class ModelResult:
    model: str
    model_id: str
    dtype: str
    device: str
    model_load_s: float
    load_s: float
    transform_s: float
    embed_s: float
    upsert_s: float
    generate_s: float
    total_s: float
    chunks_embedded: int
    embedding_dim: int
    embed_items_per_s: float
    generation_prompts: int
    generated_tokens: int
    generation_tokens_per_s: float
    peak_gpu_memory_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real Hugging Face embedding, FAISS upsert, and generation stages "
            "for Llama 3 8B and Mistral 7B."
        )
    )
    parser.add_argument(
        "--models",
        default="llama3-8b,mistral-7b",
        help="Comma-separated aliases or Hugging Face model IDs.",
    )
    parser.add_argument("--data-dir", default="", help="Optional existing text corpus.")
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--generation-samples", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("HF_HOME", "/scratch/djy8hg/huggingface"),
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prepare_synthetic_corpus(
    root: Path,
    files: int,
    chunks: int,
    chunk_chars: int,
    seed: int,
) -> Path:
    corpus_dir = root / f"corpus_{chunks}chunks_{files}files_{chunk_chars}chars_seed{seed}"
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        return corpus_dir

    corpus_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    vocabulary = (
        "distributed retrieval embedding vector memory agent workflow context "
        "benchmark pipeline scalable asynchronous semantic generation model "
        "document query index transform inference throughput latency"
    ).split()
    counts = [chunks // files] * files
    for index in range(chunks % files):
        counts[index] += 1

    for file_index, chunk_count in enumerate(counts):
        records: list[str] = []
        for chunk_index in range(chunk_count):
            prefix = f"document {file_index} chunk {chunk_index}. "
            words: list[str] = []
            while len(prefix) + sum(len(word) + 1 for word in words) < chunk_chars:
                words.append(rng.choice(vocabulary))
            records.append((prefix + " ".join(words))[:chunk_chars])
        (corpus_dir / f"doc_{file_index:06d}.txt").write_text(
            CHUNK_DELIMITER.join(records),
            encoding="utf-8",
        )

    manifest.write_text(
        json.dumps(
            {
                "files": files,
                "chunks": chunks,
                "chunk_chars": chunk_chars,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus_dir


def load_documents(data_dir: Path) -> tuple[list[str], float]:
    started = time.perf_counter()
    documents = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(data_dir.glob("*.txt"))
    ]
    return documents, time.perf_counter() - started


def transform_documents(documents: Sequence[str], limit: int) -> tuple[list[str], float]:
    started = time.perf_counter()
    chunks: list[str] = []
    for document in documents:
        chunks.extend(part.strip() for part in document.split(CHUNK_DELIMITER) if part.strip())
        if len(chunks) >= limit:
            break
    return chunks[:limit], time.perf_counter() - started


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=1)


def embed_chunks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    chunks: Sequence[str],
    batch_size: int,
    max_input_tokens: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    base_model = getattr(model, "model", model)
    vectors: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches(chunks, batch_size):
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            outputs = base_model(**encoded, use_cache=False, return_dict=True)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(vectors, axis=0), elapsed


def upsert_faiss(vectors: np.ndarray) -> float:
    import faiss

    started = time.perf_counter()
    contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(contiguous.shape[1])
    index.add(contiguous)
    if index.ntotal != contiguous.shape[0]:
        raise RuntimeError(f"FAISS inserted {index.ntotal} of {contiguous.shape[0]} vectors")
    return time.perf_counter() - started


def format_generation_prompts(tokenizer: AutoTokenizer, chunks: Sequence[str]) -> list[str]:
    prompts: list[str] = []
    for chunk in chunks:
        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following RAG context in two concise sentences:\n\n"
                    f"{chunk}"
                ),
            }
        ]
        if getattr(tokenizer, "chat_template", None):
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            prompts.append(messages[0]["content"])
    return prompts


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[int, float]:
    generated_tokens = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches(prompts, batch_size):
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_tokens += int(output.shape[0] * (output.shape[1] - encoded["input_ids"].shape[1]))
    synchronize(device)
    return generated_tokens, time.perf_counter() - started


def resolve_model(model: str) -> tuple[str, str]:
    alias = model.strip()
    return alias, MODEL_ALIASES.get(alias, alias)


def resolve_hub_cache(cache_root: str) -> str:
    cache_path = Path(cache_root).expanduser().resolve()
    hub_path = cache_path / "hub"
    return str(hub_path if hub_path.is_dir() else cache_path)


def benchmark_model(
    model_name: str,
    model_id: str,
    args: argparse.Namespace,
    data_dir: Path,
) -> ModelResult:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    hub_cache = resolve_hub_cache(args.hf_cache_dir)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hub_cache,
        local_files_only=not args.allow_download,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=hub_cache,
        local_files_only=not args.allow_download,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    synchronize(device)
    model_load_s = time.perf_counter() - load_started

    documents, load_s = load_documents(data_dir)
    chunks, transform_s = transform_documents(documents, args.chunks)
    if not chunks:
        raise RuntimeError(f"No chunks were loaded from {data_dir}")

    warmup_text = chunks[0][: args.chunk_chars]
    embed_chunks(model, tokenizer, [warmup_text], 1, min(32, args.max_input_tokens), device)
    generate_text(model, tokenizer, [warmup_text], 1, min(32, args.max_input_tokens), 1, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    vectors, embed_s = embed_chunks(
        model,
        tokenizer,
        chunks,
        args.embed_batch_size,
        args.max_input_tokens,
        device,
    )
    upsert_s = upsert_faiss(vectors)
    generation_chunks = chunks[: min(args.generation_samples, len(chunks))]
    prompts = format_generation_prompts(tokenizer, generation_chunks)
    generated_tokens, generate_s = generate_text(
        model,
        tokenizer,
        prompts,
        args.generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
        device,
    )
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )

    result = ModelResult(
        model=model_name,
        model_id=model_id,
        dtype=args.dtype,
        device=str(device),
        model_load_s=model_load_s,
        load_s=load_s,
        transform_s=transform_s,
        embed_s=embed_s,
        upsert_s=upsert_s,
        generate_s=generate_s,
        total_s=load_s + transform_s + embed_s + upsert_s + generate_s,
        chunks_embedded=len(chunks),
        embedding_dim=int(vectors.shape[1]),
        embed_items_per_s=len(chunks) / embed_s if embed_s else math.inf,
        generation_prompts=len(prompts),
        generated_tokens=generated_tokens,
        generation_tokens_per_s=generated_tokens / generate_s if generate_s else math.inf,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
    )

    del vectors, model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_results(output_dir: Path, args: argparse.Namespace, results: Sequence[ModelResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    config = vars(args).copy()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    headers = (
        "Model",
        "LoadModel(s)",
        "Load(s)",
        "Transform(s)",
        "Embed(s)",
        "Upsert(s)",
        "Generate(s)",
        "Total(s)",
        "Embed/s",
        "Gen tok/s",
        "GPU GB",
    )
    print(" | ".join(headers))
    print("-" * 132)
    for result in results:
        print(
            f"{result.model} | {result.model_load_s:.3f} | {result.load_s:.3f} | "
            f"{result.transform_s:.3f} | {result.embed_s:.3f} | {result.upsert_s:.3f} | "
            f"{result.generate_s:.3f} | {result.total_s:.3f} | "
            f"{result.embed_items_per_s:.2f} | {result.generation_tokens_per_s:.2f} | "
            f"{result.peak_gpu_memory_gb:.2f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        prep_started = time.perf_counter()
        data_dir = prepare_synthetic_corpus(
            output_dir,
            args.files,
            args.chunks,
            args.chunk_chars,
            args.seed,
        )
        prep_s = time.perf_counter() - prep_started
        print(f"CorpusPrep(s) [excluded from model pipeline totals]: {prep_s:.3f}")

    models = [resolve_model(item) for item in args.models.split(",") if item.strip()]
    results = [
        benchmark_model(model_name, model_id, args, data_dir)
        for model_name, model_id in models
    ]
    write_results(output_dir, args, results)


if __name__ == "__main__":
    main()
