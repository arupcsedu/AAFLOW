#!/usr/bin/env python3
"""Weak-scale AAFLOW ingestion/query pipelines with one HF model replica per GPU."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import dask
import faiss
import numpy as np
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_hf_models import MODEL_ALIASES, resolve_hub_cache


DELIMITER = "\n<AAFLOW_CHUNK>\n"
CONFIG_ORDER = (
    "AsyncParallelOnly",
    "DaskScalableRAG",
    "RayDataScalableRAG",
    "BulkSynchronousParallelRAG",
    "HigressRAG",
    "AAFLOW-baseline",
    "AAFLOW-embed-batched",
    "AAFLOW-generation-batched",
    "AAFLOW-prefetch",
    "AAFLOW-prefetch-static-cache",
    "AAFLOW-s2-tokenizer-prefetch",
    "AAFLOW-s3-faiss-overlap",
    "AAFLOW-s4-compile",
    "AAFLOW-s5-compile-cudagraph",
    "AAFLOW-s6-token-budget",
    "AAFLOW-s7-bucketed-compile-cudagraph",
    "AAFLOW-s8-length-bucket-eager",
    "AAFLOW-s9-length-bucket-compile",
    "AAFLOW-s10-length-bucket-compile-cudagraph",
    "AAFLOW-s11-true-bucket-compile",
    "AAFLOW-s12-true-bucket-compile-cudagraph",
    "AAFLOW-s13-deferred-transfer",
    "AAFLOW-s14-reduced-embed-tokens",
    "AAFLOW-s15-compile-reduce-overhead",
    "AAFLOW-s16-compile-max-autotune",
    "AAFLOW",
)


@dataclass
class RankResult:
    config: str
    rank: int
    chunks: int
    generated_tokens: int
    load_s: float
    transform_s: float
    embed_s: float
    upsert_s: float
    generate_s: float
    total_s: float
    peak_gpu_memory_gb: float
    optimization_setup_s: float = 0.0
    ttft_s: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    parser.add_argument("--chunks-per-gpu", type=int, default=128)
    parser.add_argument("--files-per-gpu", type=int, default=16)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--upsert-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--aaflow-embed-batch-size", type=int, default=32)
    parser.add_argument("--aaflow-generation-batch-size", type=int, default=8)
    parser.add_argument("--generation-samples-per-gpu", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        default="default",
        help="Attention backend passed to Hugging Face model loading.",
    )
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME", "/scratch/djy8hg/huggingface"))
    parser.add_argument(
        "--corpus-root",
        help="Prebuilt corpus containing rank_0000, rank_0001, ... directories.",
    )
    parser.add_argument(
        "--aaflow-tuning-only",
        action="store_true",
        help="Run AAFLOW baseline and batching ablations without competitor configurations.",
    )
    parser.add_argument(
        "--aaflow-optimized-only",
        action="store_true",
        help="Run only the configured optimized AAFLOW path.",
    )
    parser.add_argument(
        "--aaflow-strategy",
        choices=[
            "default",
            "s2_prefetch",
            "s3_faiss_overlap",
            "s4_compile",
            "s5_compile_cudagraph",
            "s6_token_budget",
            "s7_bucketed_compile_cudagraph",
            "s8_length_bucket_eager",
            "s9_length_bucket_compile",
            "s10_length_bucket_compile_cudagraph",
            "s11_true_bucket_compile",
            "s12_true_bucket_compile_cudagraph",
            "s13_deferred_transfer",
            "s14_reduced_embed_tokens",
            "s15_compile_reduce_overhead",
            "s16_compile_max_autotune",
        ],
        default="default",
        help="AAFLOW-only optimization strategy label for one-at-a-time ablation runs.",
    )
    parser.add_argument(
        "--aaflow-token-budget",
        type=int,
        default=4096,
        help="Maximum token budget per AAFLOW token-budget embedding batch.",
    )
    parser.add_argument(
        "--aaflow-reduced-embed-tokens",
        type=int,
        default=96,
        help="Reduced AAFLOW-only max input tokens for aggressive embedding ablations.",
    )
    parser.add_argument(
        "--aaflow-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode for AAFLOW compiled embedding ablations.",
    )
    parser.add_argument(
        "--aaflow-bucket-sizes",
        default="32,64,96,128",
        help="Comma-separated embedding token buckets used by AAFLOW bucketed strategies.",
    )
    parser.add_argument(
        "--aaflow-disable-faiss-overlap",
        action="store_true",
        help="Disable AAFLOW producer/consumer overlap and insert embeddings synchronously.",
    )
    parser.add_argument(
        "--disable-bsp",
        action="store_true",
        help="Keep BSP implemented but omit it from this benchmark run.",
    )
    parser.add_argument(
        "--aaflow-embed-backend",
        choices=["eager", "compile", "compile_cudagraph"],
        default="eager",
        help="AAFLOW-only embedding execution backend.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def barrier(run_dir: Path, name: str, rank: int, world: int) -> None:
    barrier_dir = run_dir / "barriers" / name
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"rank_{rank:04d}").touch()
    release = barrier_dir / "release"
    if rank == 0:
        while len(list(barrier_dir.glob("rank_*"))) < world:
            time.sleep(0.05)
        release.touch()
    else:
        while not release.exists():
            time.sleep(0.05)


def chunk_counts(total: int, files: int) -> list[int]:
    counts = [total // files] * files
    for index in range(total % files):
        counts[index] += 1
    return counts


def prepare_rank_corpus(
    run_dir: Path,
    rank: int,
    chunks: int,
    files: int,
    chunk_chars: int,
    seed: int,
) -> tuple[Path, float]:
    started = time.perf_counter()
    corpus_dir = run_dir / "rank_corpora" / f"rank_{rank:04d}"
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        return corpus_dir, time.perf_counter() - started

    corpus_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + rank * 1_000_003)
    vocabulary = (
        "distributed retrieval embedding vector memory agent workflow context benchmark "
        "pipeline scalable asynchronous semantic generation model document query index "
        "transform inference throughput latency"
    ).split()
    for file_index, count in enumerate(chunk_counts(chunks, files)):
        records: list[str] = []
        for chunk_index in range(count):
            prefix = f"rank {rank} document {file_index} chunk {chunk_index}. "
            words: list[str] = []
            length = len(prefix)
            while length < chunk_chars:
                word = rng.choice(vocabulary)
                words.append(word)
                length += len(word) + 1
            records.append((prefix + " ".join(words))[:chunk_chars])
        (corpus_dir / f"doc_{file_index:04d}.txt").write_text(
            DELIMITER.join(records),
            encoding="utf-8",
        )
    manifest.write_text(
        json.dumps(
            {"rank": rank, "chunks": chunks, "files": files, "chunk_chars": chunk_chars},
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus_dir, time.perf_counter() - started


def resolve_rank_corpus(
    args: argparse.Namespace,
    run_dir: Path,
    rank: int,
) -> tuple[Path, float]:
    if not args.corpus_root:
        return prepare_rank_corpus(
            run_dir,
            rank,
            args.chunks_per_gpu,
            args.files_per_gpu,
            args.chunk_chars,
            args.seed,
        )

    started = time.perf_counter()
    corpus_root = Path(args.corpus_root).resolve()
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prebuilt corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chunks_per_rank") != args.chunks_per_gpu:
        raise ValueError(
            f"Corpus has {manifest.get('chunks_per_rank')} chunks/rank, "
            f"but benchmark requested {args.chunks_per_gpu}"
        )
    corpus_dir = corpus_root / f"rank_{rank:04d}"
    paths = list(corpus_dir.glob("*.txt"))
    if len(paths) != args.files_per_gpu:
        raise ValueError(
            f"Corpus rank {rank} has {len(paths)} files, expected {args.files_per_gpu}"
        )
    return corpus_dir, time.perf_counter() - started


def batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_bucket_sizes(spec: str, max_input_tokens: int) -> list[int]:
    # Slurm --export uses commas as separators, so allow ':' for env-provided lists.
    normalized = spec.replace(":", ",")
    buckets = sorted({int(part) for part in normalized.split(",") if part.strip()})
    buckets = [bucket for bucket in buckets if bucket > 0]
    if max_input_tokens not in buckets:
        buckets.append(max_input_tokens)
    return sorted({min(bucket, max_input_tokens) for bucket in buckets})


def choose_bucket(length: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]


def length_bucketed_batches(
    engine: "HuggingFaceEngine",
    chunks: Sequence[str],
    max_batch_size: int,
    max_input_tokens: int,
    bucket_sizes: Sequence[int],
    token_budget: int | None = None,
    fixed_batch_size: bool = False,
) -> list[tuple[int, list[str], int]]:
    lengths = engine.token_lengths(chunks, max_input_tokens)
    groups: dict[int, list[tuple[int, str]]] = {bucket: [] for bucket in bucket_sizes}
    for chunk, length in zip(chunks, lengths):
        groups[choose_bucket(length, bucket_sizes)].append((length, chunk))

    batches: list[tuple[int, list[str], int]] = []
    for bucket in bucket_sizes:
        items = sorted(groups.get(bucket, []), key=lambda item: item[0])
        if not items:
            continue
        if fixed_batch_size:
            for start in range(0, len(items), max_batch_size):
                texts = [text for _, text in items[start : start + max_batch_size]]
                original_count = len(texts)
                if original_count < max_batch_size:
                    texts.extend([texts[-1]] * (max_batch_size - original_count))
                batches.append((bucket, texts, original_count))
            continue

        current: list[str] = []
        current_tokens = 0
        budget = token_budget or max_batch_size * bucket
        for length, text in items:
            should_flush = current and (
                len(current) >= max_batch_size or current_tokens + length > budget
            )
            if should_flush:
                batches.append((bucket, current, len(current)))
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += length
        if current:
            batches.append((bucket, current, len(current)))
    return batches


def true_bucketed_compile_batches(
    engine: "HuggingFaceEngine",
    chunks: Sequence[str],
    base_batch_size: int,
    max_input_tokens: int,
    bucket_sizes: Sequence[int],
    token_budget: int,
) -> list[tuple[int, int, list[str], int]]:
    """Create fixed-shape batches keyed by (token bucket, batch size).

    The previous compiled path used one max-length graph for all batches. This
    variant compiles the real bucket shapes and increases batch size for shorter
    buckets while keeping an approximately constant token budget per GPU launch.
    Last batches are padded to keep compiled graph shapes static.
    """
    lengths = engine.token_lengths(chunks, max_input_tokens)
    groups: dict[int, list[tuple[int, str]]] = {bucket: [] for bucket in bucket_sizes}
    for chunk, length in zip(chunks, lengths):
        groups[choose_bucket(length, bucket_sizes)].append((length, chunk))

    batches: list[tuple[int, int, list[str], int]] = []
    for bucket in bucket_sizes:
        items = sorted(groups.get(bucket, []), key=lambda item: item[0])
        if not items:
            continue
        batch_size = max(base_batch_size, token_budget // max(bucket, 1))
        batch_size = max(1, batch_size)
        for start in range(0, len(items), batch_size):
            texts = [text for _, text in items[start : start + batch_size]]
            original_count = len(texts)
            if original_count < batch_size:
                texts.extend([texts[-1]] * (batch_size - original_count))
            batches.append((bucket, batch_size, texts, original_count))
    return batches


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_document(document: str) -> list[str]:
    return [part.strip() for part in document.split(DELIMITER) if part.strip()]


def ray_read_record(record: dict[str, str]) -> dict[str, str]:
    return {"document": read_file(Path(record["path"]))}


def ray_split_record(record: dict[str, str]) -> list[dict[str, str]]:
    return [{"chunk": chunk} for chunk in split_document(record["document"])]


class HuggingFaceEngine:
    def __init__(
        self,
        model_id: str,
        cache_dir: str,
        dtype_name: str,
        attn_implementation: str = "default",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda")
        self.dtype = getattr(torch, dtype_name)
        self._lock = threading.Lock()
        hub_cache = resolve_hub_cache(cache_dir)

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=hub_cache,
            local_files_only=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {
            "cache_dir": hub_cache,
            "local_files_only": True,
            "dtype": self.dtype,
        }
        if attn_implementation != "default":
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        torch.cuda.synchronize()
        self.model_load_s = time.perf_counter() - started
        self.dim = int(self.model.config.hidden_size)
        self.compiled_embed_model: torch.nn.Module | None = None
        self.compiled_embed_models: dict[int | tuple[int, int], torch.nn.Module] = {}
        self.compiled_embed_setup_s = 0.0

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def embed(self, texts: Sequence[str], max_input_tokens: int) -> np.ndarray:
        with self._lock, torch.inference_mode():
            encoded = self.tokenize(texts, max_input_tokens)
            return self.embed_encoded(encoded)

    def tokenize(
        self,
        texts: Sequence[str],
        max_input_tokens: int,
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )

    def tokenize_fixed(
        self,
        texts: Sequence[str],
        max_input_tokens: int,
    ) -> dict[str, torch.Tensor]:
        return self.tokenize_fixed_length(texts, max_input_tokens)

    def tokenize_fixed_length(
        self,
        texts: Sequence[str],
        length: int,
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="pt",
        )

    def token_lengths(self, texts: Sequence[str], max_input_tokens: int) -> list[int]:
        encoded = self.tokenizer(
            list(texts),
            padding=False,
            truncation=True,
            max_length=max_input_tokens,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is not None:
            return [min(int(length), max_input_tokens) for length in lengths]
        return [min(len(input_ids), max_input_tokens) for input_ids in encoded["input_ids"]]

    def embed_encoded(self, encoded: dict[str, torch.Tensor]) -> np.ndarray:
        return self.embed_encoded_tensor(encoded).float().cpu().numpy()

    def embed_encoded_tensor(
        self,
        encoded: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        base_model = getattr(self.model, "model", self.model)
        outputs = base_model(**encoded, use_cache=False, return_dict=True)
        return self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    def prepare_compiled_embed(
        self,
        sample_texts: Sequence[str],
        max_input_tokens: int,
        use_cudagraphs: bool,
        graph_length: int | None = None,
        graph_batch_size: int | None = None,
        compile_mode: str = "default",
    ) -> float:
        engine = self
        base_model = getattr(self.model, "model", self.model)
        compile_length = graph_length or max_input_tokens
        compile_key: int | tuple[int, int] = (
            (compile_length, graph_batch_size)
            if graph_batch_size is not None
            else compile_length
        )

        class EmbedGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                return engine._mean_pool(outputs.last_hidden_state, attention_mask)

        encoded_cpu = self.tokenize_fixed_length(sample_texts, compile_length)
        encoded = {name: tensor.to(self.device) for name, tensor in encoded_cpu.items()}
        eager_graph = EmbedGraph().to(self.device).eval()
        with torch.inference_mode():
            reference = eager_graph(
                encoded["input_ids"],
                encoded["attention_mask"],
            ).float()
        torch.cuda.synchronize()

        options = {
            "triton.cudagraphs": use_cudagraphs,
            "triton.cudagraph_trees": use_cudagraphs,
        }
        started = time.perf_counter()
        compile_kwargs = {
            "backend": "inductor",
            "fullgraph": False,
            "dynamic": False,
        }
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        else:
            compile_kwargs["options"] = options
        compiled = torch.compile(eager_graph, **compile_kwargs)
        with torch.inference_mode():
            candidate = None
            for _ in range(3):
                if use_cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                candidate = compiled(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - started
        if candidate is None:
            raise RuntimeError("Compiled embedding graph produced no output")
        max_error = float((candidate.float() - reference).abs().max().item())
        cosine = torch.nn.functional.cosine_similarity(
            candidate.float(),
            reference,
            dim=1,
        )
        min_cosine = float(cosine.min().item())
        if max_error > 1e-2 or min_cosine < 0.999:
            raise RuntimeError(
                "Compiled embedding graph failed correctness check: "
                f"max error {max_error}, min cosine {min_cosine}"
            )
        self.compiled_embed_models[compile_key] = compiled
        if graph_length is None:
            self.compiled_embed_model = compiled
        self.compiled_embed_setup_s += setup_s
        return setup_s

    def embed_compiled_encoded(
        self,
        encoded: dict[str, torch.Tensor],
        use_cudagraphs: bool,
        graph_length: int | None = None,
        graph_batch_size: int | None = None,
    ) -> np.ndarray:
        return self.embed_compiled_encoded_tensor(
            encoded,
            use_cudagraphs,
            graph_length,
            graph_batch_size,
        ).float().cpu().numpy()

    def embed_compiled_encoded_tensor(
        self,
        encoded: dict[str, torch.Tensor],
        use_cudagraphs: bool,
        graph_length: int | None = None,
        graph_batch_size: int | None = None,
    ) -> torch.Tensor:
        compile_key: int | tuple[int, int] | None = (
            (graph_length, graph_batch_size)
            if graph_length is not None and graph_batch_size is not None
            else graph_length
        )
        compiled_model = (
            self.compiled_embed_models.get(compile_key)
            if compile_key is not None
            else self.compiled_embed_model
        )
        if compiled_model is None:
            raise RuntimeError("Compiled embedding backend was not prepared")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        if use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        pooled = compiled_model(
            encoded["input_ids"],
            encoded["attention_mask"],
        )
        return pooled

    def generate(
        self,
        chunks: Sequence[str],
        batch_size: int,
        max_input_tokens: int,
        max_new_tokens: int,
        prefetch_tokenization: bool = False,
        static_cache: bool = False,
    ) -> tuple[int, float]:
        prompts: list[str] = []
        for chunk in chunks:
            messages = [
                {
                    "role": "user",
                    "content": f"Summarize this RAG context in two concise sentences:\n\n{chunk}",
                }
            ]
            if getattr(self.tokenizer, "chat_template", None):
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                prompts.append(messages[0]["content"])

        generated_tokens = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            prompt_batches = list(batched(prompts, batch_size))

            def tokenize_batch(batch: Sequence[str]) -> dict[str, torch.Tensor]:
                return self.tokenize(batch, max_input_tokens)

            def generate_encoded(encoded: dict[str, torch.Tensor]) -> int:
                encoded = {
                    name: tensor.to(self.device) for name, tensor in encoded.items()
                }
                generate_args = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if static_cache:
                    generate_args["cache_implementation"] = "static"
                outputs = self.model.generate(**encoded, **generate_args)
                generated = outputs[:, encoded["input_ids"].shape[1] :]
                count = 0
                for row in generated:
                    eos_positions = (row == self.tokenizer.eos_token_id).nonzero(
                        as_tuple=False
                    )
                    count += (
                        int(eos_positions[0].item()) + 1
                        if len(eos_positions)
                        else int(row.numel())
                    )
                return count

            if prefetch_tokenization and prompt_batches:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(tokenize_batch, prompt_batches[0])
                    for batch_index in range(len(prompt_batches)):
                        encoded = pending.result()
                        if batch_index + 1 < len(prompt_batches):
                            pending = pool.submit(
                                tokenize_batch,
                                prompt_batches[batch_index + 1],
                            )
                        generated_tokens += generate_encoded(encoded)
            else:
                for batch in prompt_batches:
                    generated_tokens += generate_encoded(tokenize_batch(batch))
        torch.cuda.synchronize()
        return generated_tokens, time.perf_counter() - started

    def measure_ttft(
        self,
        chunks: Sequence[str],
        batch_size: int,
        max_input_tokens: int,
    ) -> float:
        if not chunks:
            return 0.0
        prompts: list[str] = []
        for chunk in chunks[:batch_size]:
            messages = [
                {
                    "role": "user",
                    "content": f"Summarize this RAG context in two concise sentences:\n\n{chunk}",
                }
            ]
            if getattr(self.tokenizer, "chat_template", None):
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                prompts.append(messages[0]["content"])
        with self._lock, torch.inference_mode():
            encoded = self.tokenize(prompts, max_input_tokens)
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            self.model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            return time.perf_counter() - started

    def warmup(self, text: str, max_input_tokens: int) -> None:
        self.embed([text], min(32, max_input_tokens))
        self.generate([text], 1, min(32, max_input_tokens), 1)
        torch.cuda.empty_cache()


def load_sequential(paths: Sequence[Path]) -> tuple[list[str], float]:
    started = time.perf_counter()
    return [read_file(path) for path in paths], time.perf_counter() - started


def load_threaded(paths: Sequence[Path], workers: int = 4) -> tuple[list[str], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        documents = list(pool.map(read_file, paths))
    return documents, time.perf_counter() - started


def transform_sequential(documents: Sequence[str]) -> tuple[list[str], float]:
    started = time.perf_counter()
    chunks = [chunk for document in documents for chunk in split_document(document)]
    return chunks, time.perf_counter() - started


def transform_threaded(documents: Sequence[str], workers: int = 4) -> tuple[list[str], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(split_document, documents))
    return [chunk for group in parts for chunk in group], time.perf_counter() - started


def embed_batches(
    engine: HuggingFaceEngine,
    chunks: Sequence[str],
    batch_size: int,
    max_input_tokens: int,
) -> tuple[list[np.ndarray], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    vectors = [
        engine.embed(batch, max_input_tokens)
        for batch in batched(chunks, batch_size)
    ]
    torch.cuda.synchronize()
    return vectors, time.perf_counter() - started


async def embed_async_batches(
    engine: HuggingFaceEngine,
    chunks: Sequence[str],
    batch_size: int,
    max_input_tokens: int,
) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    tasks = [
        asyncio.to_thread(engine.embed, batch, max_input_tokens)
        for batch in batched(chunks, batch_size)
    ]
    vectors = await asyncio.gather(*tasks)
    torch.cuda.synchronize()
    return list(vectors), time.perf_counter() - started


def upsert_batches(vectors: Sequence[np.ndarray], dim: int) -> float:
    started = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    for batch in vectors:
        index.add(np.ascontiguousarray(batch, dtype=np.float32))
    expected = sum(batch.shape[0] for batch in vectors)
    if index.ntotal != expected:
        raise RuntimeError(f"FAISS inserted {index.ntotal} of {expected} vectors")
    return time.perf_counter() - started


def run_standard_config(
    config: str,
    load_fn: Callable[[Sequence[Path]], tuple[list[str], float]],
    transform_fn: Callable[[Sequence[str]], tuple[list[str], float]],
    embed_fn: Callable[[Sequence[str]], tuple[list[np.ndarray], float]],
    engine: HuggingFaceEngine,
    paths: Sequence[Path],
    generation_samples: int,
    generation_batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    rank: int,
) -> RankResult:
    total_started = time.perf_counter()
    documents, load_s = load_fn(paths)
    chunks, transform_s = transform_fn(documents)
    vectors, embed_s = embed_fn(chunks)
    upsert_s = upsert_batches(vectors, engine.dim)
    generated_tokens, generate_s = engine.generate(
        chunks[:generation_samples],
        generation_batch_size,
        max_input_tokens,
        max_new_tokens,
    )
    total_s = time.perf_counter() - total_started
    ttft_s = engine.measure_ttft(
        chunks[:generation_samples],
        generation_batch_size,
        max_input_tokens,
    )
    return RankResult(
        config=config,
        rank=rank,
        chunks=len(chunks),
        generated_tokens=generated_tokens,
        load_s=load_s,
        transform_s=transform_s,
        embed_s=embed_s,
        upsert_s=upsert_s,
        generate_s=generate_s,
        total_s=total_s,
        peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / (1024**3),
        ttft_s=ttft_s,
    )


def run_dask_config(
    engine: HuggingFaceEngine,
    paths: Sequence[Path],
    args: argparse.Namespace,
    rank: int,
) -> RankResult:
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    documents = list(
        dask.compute(
            *[dask.delayed(read_file)(path) for path in paths],
            scheduler="threads",
            num_workers=4,
        )
    )
    load_s = time.perf_counter() - load_started

    transform_started = time.perf_counter()
    groups = dask.compute(
        *[dask.delayed(split_document)(document) for document in documents],
        scheduler="threads",
        num_workers=4,
    )
    chunks = [chunk for group in groups for chunk in group]
    transform_s = time.perf_counter() - transform_started

    embed_started = time.perf_counter()
    vectors = list(
        dask.compute(
            *[
                dask.delayed(engine.embed)(batch, args.max_input_tokens)
                for batch in batched(chunks, args.embed_batch_size)
            ],
            scheduler="threads",
            num_workers=4,
        )
    )
    torch.cuda.synchronize()
    embed_s = time.perf_counter() - embed_started
    upsert_s = upsert_batches(vectors, engine.dim)
    generated_tokens, generate_s = engine.generate(
        chunks[: args.generation_samples_per_gpu],
        args.generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
    )
    total_s = time.perf_counter() - total_started
    ttft_s = engine.measure_ttft(
        chunks[: args.generation_samples_per_gpu],
        args.generation_batch_size,
        args.max_input_tokens,
    )
    return RankResult(
        "DaskScalableRAG",
        rank,
        len(chunks),
        generated_tokens,
        load_s,
        transform_s,
        embed_s,
        upsert_s,
        generate_s,
        total_s,
        torch.cuda.max_memory_allocated() / (1024**3),
        ttft_s=ttft_s,
    )


def run_ray_config(
    engine: HuggingFaceEngine,
    paths: Sequence[Path],
    args: argparse.Namespace,
    rank: int,
) -> RankResult:
    total_started = time.perf_counter()
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    ray_temp = Path(os.environ.get("TMPDIR", "/tmp")) / f"aaflow_ray_{job_id}_{rank}"

    load_started = time.perf_counter()
    ray.init(
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        logging_level="ERROR",
        _temp_dir=str(ray_temp),
    )
    try:
        documents = [
            record["document"]
            for record in (
                ray.data.from_items([{"path": str(path)} for path in paths])
                .map(ray_read_record, concurrency=4)
                .take_all()
            )
        ]
        load_s = time.perf_counter() - load_started

        transform_started = time.perf_counter()
        chunks = [
            record["chunk"]
            for record in (
                ray.data.from_items([{"document": document} for document in documents])
                .flat_map(ray_split_record, concurrency=4)
                .take_all()
            )
        ]
        transform_s = time.perf_counter() - transform_started

        vectors, embed_s = embed_batches(
            engine,
            chunks,
            args.embed_batch_size,
            args.max_input_tokens,
        )
        upsert_s = upsert_batches(vectors, engine.dim)
        generated_tokens, generate_s = engine.generate(
            chunks[: args.generation_samples_per_gpu],
            args.generation_batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        total_s = time.perf_counter() - total_started
        ttft_s = engine.measure_ttft(
            chunks[: args.generation_samples_per_gpu],
            args.generation_batch_size,
            args.max_input_tokens,
        )
        return RankResult(
            "RayDataScalableRAG",
            rank,
            len(chunks),
            generated_tokens,
            load_s,
            transform_s,
            embed_s,
            upsert_s,
            generate_s,
            total_s,
            torch.cuda.max_memory_allocated() / (1024**3),
            ttft_s=ttft_s,
        )
    finally:
        ray.shutdown()


def run_aaflow(
    engine: HuggingFaceEngine,
    paths: Sequence[Path],
    args: argparse.Namespace,
    rank: int,
    config: str = "AAFLOW",
    embed_batch_size: int | None = None,
    generation_batch_size: int | None = None,
    length_bucket: bool = True,
    prefetch_tokenization: bool = False,
    static_cache: bool = False,
    embed_backend: str = "eager",
    overlap_faiss: bool = True,
    embedding_schedule: str = "fixed",
    defer_embedding_transfer: bool = False,
    embed_max_input_tokens: int | None = None,
) -> RankResult:
    total_started = time.perf_counter()
    documents, load_s = load_threaded(paths)
    chunks, transform_s = transform_threaded(documents)
    work_queue: queue.Queue[np.ndarray | None] | None = (
        queue.Queue(maxsize=2) if overlap_faiss and not defer_embedding_transfer else None
    )
    index = faiss.IndexFlatIP(engine.dim)
    upsert_active_s = 0.0
    consumer_errors: list[BaseException] = []
    deferred_gpu_vectors: list[torch.Tensor] = []

    def add_vectors(batch: np.ndarray) -> None:
        nonlocal upsert_active_s
        started = time.perf_counter()
        index.add(np.ascontiguousarray(batch, dtype=np.float32))
        upsert_active_s += time.perf_counter() - started

    def consume() -> None:
        if work_queue is None:
            return
        while True:
            batch = work_queue.get()
            if batch is None:
                return
            if consumer_errors:
                continue
            try:
                add_vectors(batch)
            except BaseException as error:
                consumer_errors.append(error)

    consumer: threading.Thread | None = None
    if work_queue is not None:
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
    torch.cuda.synchronize()
    embed_started = time.perf_counter()
    embed_batch_size = embed_batch_size or args.aaflow_embed_batch_size
    generation_batch_size = generation_batch_size or args.aaflow_generation_batch_size
    embed_max_input_tokens = embed_max_input_tokens or args.max_input_tokens
    ordered_chunks = sorted(chunks, key=len) if length_bucket else list(chunks)
    bucket_sizes = parse_bucket_sizes(args.aaflow_bucket_sizes, embed_max_input_tokens)
    if embedding_schedule == "token_budget":
        scheduled_batches = [
            (bucket, None, batch, original_count)
            for bucket, batch, original_count in length_bucketed_batches(
                engine,
                ordered_chunks,
                embed_batch_size,
                embed_max_input_tokens,
                bucket_sizes,
                token_budget=args.aaflow_token_budget,
                fixed_batch_size=False,
            )
        ]
    elif embedding_schedule == "bucketed_compile":
        scheduled_batches = [
            (bucket, embed_batch_size, batch, original_count)
            for bucket, batch, original_count in length_bucketed_batches(
                engine,
                ordered_chunks,
                embed_batch_size,
                embed_max_input_tokens,
                bucket_sizes,
                fixed_batch_size=True,
            )
        ]
    elif embedding_schedule == "true_bucketed_compile":
        scheduled_batches = true_bucketed_compile_batches(
            engine,
            ordered_chunks,
            embed_batch_size,
            embed_max_input_tokens,
            bucket_sizes,
            token_budget=args.aaflow_token_budget,
        )
    else:
        scheduled_batches = [
            (embed_max_input_tokens, None, list(batch), len(batch))
            for batch in batched(ordered_chunks, embed_batch_size)
        ]
    if prefetch_tokenization and scheduled_batches:
        with ThreadPoolExecutor(max_workers=1) as pool, torch.inference_mode():
            def tokenize_scheduled(
                item: tuple[int, int | None, list[str], int],
            ) -> tuple[int, int | None, int, dict[str, torch.Tensor]]:
                bucket, batch_size, batch, original_count = item
                if embedding_schedule in {"bucketed_compile", "true_bucketed_compile"}:
                    encoded = engine.tokenize_fixed_length(batch, bucket)
                elif embed_backend != "eager":
                    encoded = engine.tokenize_fixed(batch, embed_max_input_tokens)
                else:
                    encoded = engine.tokenize(batch, embed_max_input_tokens)
                return bucket, batch_size, original_count, encoded

            pending = pool.submit(
                tokenize_scheduled,
                scheduled_batches[0],
            )
            for batch_index in range(len(scheduled_batches)):
                bucket, batch_size, original_count, encoded = pending.result()
                if batch_index + 1 < len(scheduled_batches):
                    pending = pool.submit(
                        tokenize_scheduled,
                        scheduled_batches[batch_index + 1],
                    )
                if embed_backend == "eager":
                    if defer_embedding_transfer:
                        vectors = engine.embed_encoded_tensor(encoded)
                    else:
                        vectors = engine.embed_encoded(encoded)
                else:
                    graph_length = (
                        bucket
                        if embedding_schedule in {
                            "bucketed_compile",
                            "true_bucketed_compile",
                        }
                        else None
                    )
                    graph_batch_size = (
                        batch_size
                        if embedding_schedule == "true_bucketed_compile"
                        else None
                    )
                    if defer_embedding_transfer:
                        vectors = engine.embed_compiled_encoded_tensor(
                            encoded,
                            use_cudagraphs=embed_backend == "compile_cudagraph",
                            graph_length=graph_length,
                            graph_batch_size=graph_batch_size,
                        )
                    else:
                        vectors = engine.embed_compiled_encoded(
                            encoded,
                            use_cudagraphs=embed_backend == "compile_cudagraph",
                            graph_length=graph_length,
                            graph_batch_size=graph_batch_size,
                        )
                vectors = vectors[:original_count]
                if defer_embedding_transfer:
                    # CUDA graph outputs are overwritten on replay; clone before
                    # retaining them for one bulk host transfer.
                    deferred_gpu_vectors.append(vectors.detach().clone())
                elif work_queue is not None:
                    work_queue.put(vectors)
                else:
                    add_vectors(vectors)
    else:
        for bucket, batch_size, batch, original_count in scheduled_batches:
            if embedding_schedule in {"bucketed_compile", "true_bucketed_compile"}:
                encoded = engine.tokenize_fixed_length(batch, bucket)
                graph_batch_size = (
                    batch_size if embedding_schedule == "true_bucketed_compile" else None
                )
                if defer_embedding_transfer:
                    vectors = engine.embed_compiled_encoded_tensor(
                        encoded,
                        use_cudagraphs=embed_backend == "compile_cudagraph",
                        graph_length=bucket,
                        graph_batch_size=graph_batch_size,
                    )[:original_count]
                else:
                    vectors = engine.embed_compiled_encoded(
                        encoded,
                        use_cudagraphs=embed_backend == "compile_cudagraph",
                        graph_length=bucket,
                        graph_batch_size=graph_batch_size,
                    )[:original_count]
            else:
                if defer_embedding_transfer:
                    encoded = engine.tokenize(batch, embed_max_input_tokens)
                    vectors = engine.embed_encoded_tensor(encoded)
                else:
                    vectors = engine.embed(batch, embed_max_input_tokens)
            if defer_embedding_transfer:
                # CUDA graph outputs are overwritten on replay; clone before
                # retaining them for one bulk host transfer.
                deferred_gpu_vectors.append(vectors.detach().clone())
            elif work_queue is not None:
                work_queue.put(vectors)
            else:
                add_vectors(vectors)
    deferred_cpu_vectors: np.ndarray | None = None
    if defer_embedding_transfer:
        deferred_cpu_vectors = torch.cat(deferred_gpu_vectors, dim=0).float().cpu().numpy()
        deferred_gpu_vectors.clear()
    torch.cuda.synchronize()
    embed_s = time.perf_counter() - embed_started
    if work_queue is not None:
        work_queue.put(None)
    if consumer is not None:
        consumer.join()
    if consumer_errors:
        raise RuntimeError("AAFLOW FAISS consumer failed") from consumer_errors[0]
    if deferred_cpu_vectors is not None:
        add_vectors(deferred_cpu_vectors)
        del deferred_cpu_vectors
    if index.ntotal != len(chunks):
        raise RuntimeError(f"FAISS inserted {index.ntotal} of {len(chunks)} vectors")
    generated_tokens, generate_s = engine.generate(
        (
            sorted(chunks[: args.generation_samples_per_gpu], key=len)
            if length_bucket
            else chunks[: args.generation_samples_per_gpu]
        ),
        generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
        prefetch_tokenization=prefetch_tokenization,
        static_cache=static_cache,
    )
    total_s = time.perf_counter() - total_started
    generation_chunks = (
        sorted(chunks[: args.generation_samples_per_gpu], key=len)
        if length_bucket
        else chunks[: args.generation_samples_per_gpu]
    )
    ttft_s = engine.measure_ttft(
        generation_chunks,
        generation_batch_size,
        args.max_input_tokens,
    )
    return RankResult(
        config,
        rank,
        len(chunks),
        generated_tokens,
        load_s,
        transform_s,
        embed_s,
        upsert_active_s,
        generate_s,
        total_s,
        torch.cuda.max_memory_allocated() / (1024**3),
        engine.compiled_embed_setup_s if embed_backend != "eager" else 0.0,
        ttft_s=ttft_s,
    )


def aggregate(run_dir: Path, world: int, model_name: str, model_id: str, model_load_s: float) -> None:
    rank_rows: list[RankResult] = []
    result_paths = sorted(
        path
        for path in (run_dir / "rank_results").glob("rank_*.json")
        if not path.name.endswith("_meta.json")
    )
    for path in result_paths:
        rank_rows.extend(RankResult(**row) for row in json.loads(path.read_text(encoding="utf-8")))

    summary_rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        rows = [row for row in rank_rows if row.config == config]
        if not rows:
            continue
        total_s = max(row.total_s for row in rows)
        chunks = sum(row.chunks for row in rows)
        generated_tokens = sum(row.generated_tokens for row in rows)
        summary_rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "config": config,
                "gpus": world,
                "chunks": chunks,
                "chunks_per_gpu": chunks // world,
                "generated_tokens": generated_tokens,
                "model_load_s_max": model_load_s,
                "load_s": max(row.load_s for row in rows),
                "transform_s": max(row.transform_s for row in rows),
                "embed_s": max(row.embed_s for row in rows),
                "upsert_s": max(row.upsert_s for row in rows),
                "generate_s": max(row.generate_s for row in rows),
                "ttft_s": max(row.ttft_s for row in rows),
                "total_s": total_s,
                "chunks_per_s": chunks / total_s,
                "generation_tokens_per_s": generated_tokens / max(row.generate_s for row in rows),
                "peak_gpu_memory_gb_max": max(row.peak_gpu_memory_gb for row in rows),
                "optimization_setup_s_max": max(
                    row.optimization_setup_s for row in rows
                ),
            }
        )

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    columns = (
        ("Config", 24),
        ("GPUs", 4),
        ("Chunks", 7),
        ("Load(s)", 8),
        ("Transform(s)", 12),
        ("Embed(s)", 9),
        ("Upsert(s)", 9),
        ("Generate(s)", 11),
        ("Total(s)", 9),
        ("Chunks/s", 9),
        ("Gen tok/s", 10),
        ("GPU GB", 7),
        ("Setup(s)", 8),
    )
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['config']:<24.24}  "
            f"{world:>4}  "
            f"{row['chunks']:>7}  "
            f"{row['load_s']:>8.3f}  "
            f"{row['transform_s']:>12.3f}  "
            f"{row['embed_s']:>9.3f}  "
            f"{row['upsert_s']:>9.3f}  "
            f"{row['generate_s']:>11.3f}  "
            f"{row['total_s']:>9.3f}  "
            f"{row['chunks_per_s']:>9.2f}  "
            f"{row['generation_tokens_per_s']:>10.2f}  "
            f"{row['peak_gpu_memory_gb_max']:>7.2f}  "
            f"{row['optimization_setup_s_max']:>8.2f}"
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
    engine = HuggingFaceEngine(
        model_id,
        args.hf_cache_dir,
        args.dtype,
        args.attn_implementation,
    )
    sample = read_file(paths[0]).split(DELIMITER)[0]
    engine.warmup(sample, args.max_input_tokens)
    strategy_backend = {
        "s4_compile": "compile",
        "s5_compile_cudagraph": "compile_cudagraph",
        "s7_bucketed_compile_cudagraph": "compile_cudagraph",
        "s9_length_bucket_compile": "compile",
        "s10_length_bucket_compile_cudagraph": "compile_cudagraph",
        "s11_true_bucket_compile": "compile",
        "s12_true_bucket_compile_cudagraph": "compile_cudagraph",
        "s13_deferred_transfer": "compile_cudagraph",
        "s14_reduced_embed_tokens": "compile_cudagraph",
        "s15_compile_reduce_overhead": "compile_cudagraph",
        "s16_compile_max_autotune": "compile_cudagraph",
    }.get(args.aaflow_strategy, args.aaflow_embed_backend)
    strategy_compile_mode = {
        "s15_compile_reduce_overhead": "reduce-overhead",
        "s16_compile_max_autotune": "max-autotune",
    }.get(args.aaflow_strategy, args.aaflow_compile_mode)
    strategy_embed_max_input_tokens = (
        min(args.max_input_tokens, args.aaflow_reduced_embed_tokens)
        if args.aaflow_strategy == "s14_reduced_embed_tokens"
        else args.max_input_tokens
    )
    if args.aaflow_strategy in {
        "s7_bucketed_compile_cudagraph",
        "s11_true_bucket_compile",
        "s12_true_bucket_compile_cudagraph",
    }:
        compile_chunks: list[str] = []
        for path in paths:
            compile_chunks.extend(split_document(read_file(path)))
        bucket_sizes = parse_bucket_sizes(args.aaflow_bucket_sizes, args.max_input_tokens)
        if args.aaflow_strategy == "s7_bucketed_compile_cudagraph":
            compile_batches = [
                (bucket, None, batch)
                for bucket, batch, _ in length_bucketed_batches(
                    engine,
                    compile_chunks,
                    args.aaflow_embed_batch_size,
                    args.max_input_tokens,
                    bucket_sizes,
                    fixed_batch_size=True,
                )
            ]
        else:
            compile_batches = [
                (bucket, batch_size, batch)
                for bucket, batch_size, batch, _ in true_bucketed_compile_batches(
                    engine,
                    compile_chunks,
                    args.aaflow_embed_batch_size,
                    args.max_input_tokens,
                    bucket_sizes,
                    args.aaflow_token_budget,
                )
            ]
        prepared: set[int | tuple[int, int]] = set()
        for bucket, batch_size, batch in compile_batches:
            compile_key: int | tuple[int, int] = (
                (bucket, batch_size) if batch_size is not None else bucket
            )
            if compile_key in prepared:
                continue
            engine.prepare_compiled_embed(
                batch,
                args.max_input_tokens,
                use_cudagraphs=strategy_backend == "compile_cudagraph",
                graph_length=bucket,
                graph_batch_size=batch_size,
                compile_mode=strategy_compile_mode,
            )
            prepared.add(compile_key)
    elif strategy_backend != "eager":
        compile_chunks: list[str] = []
        for path in paths:
            compile_chunks.extend(split_document(read_file(path)))
            if len(compile_chunks) >= args.aaflow_embed_batch_size:
                compile_chunks = compile_chunks[: args.aaflow_embed_batch_size]
                break
        if len(compile_chunks) != args.aaflow_embed_batch_size:
            raise RuntimeError(
                "The first corpus file does not contain a full AAFLOW compile batch"
            )
        engine.prepare_compiled_embed(
            compile_chunks,
            strategy_embed_max_input_tokens,
            use_cudagraphs=strategy_backend == "compile_cudagraph",
            compile_mode=strategy_compile_mode,
        )
    torch.cuda.reset_peak_memory_stats()
    barrier(run_dir, "model_ready", rank, world)

    results: list[RankResult] = []
    strategy_config = "AAFLOW"
    strategy_backend = {
        "s4_compile": "compile",
        "s5_compile_cudagraph": "compile_cudagraph",
        "s7_bucketed_compile_cudagraph": "compile_cudagraph",
        "s9_length_bucket_compile": "compile",
        "s10_length_bucket_compile_cudagraph": "compile_cudagraph",
        "s11_true_bucket_compile": "compile",
        "s12_true_bucket_compile_cudagraph": "compile_cudagraph",
        "s13_deferred_transfer": "compile_cudagraph",
        "s14_reduced_embed_tokens": "compile_cudagraph",
        "s15_compile_reduce_overhead": "compile_cudagraph",
        "s16_compile_max_autotune": "compile_cudagraph",
    }.get(args.aaflow_strategy, args.aaflow_embed_backend)
    strategy_embedding_schedule = {
        "s6_token_budget": "token_budget",
        "s7_bucketed_compile_cudagraph": "bucketed_compile",
        "s11_true_bucket_compile": "true_bucketed_compile",
        "s12_true_bucket_compile_cudagraph": "true_bucketed_compile",
    }.get(args.aaflow_strategy, "fixed")
    strategy_length_bucket = args.aaflow_strategy in {
        "s8_length_bucket_eager",
        "s9_length_bucket_compile",
        "s10_length_bucket_compile_cudagraph",
        "s11_true_bucket_compile",
        "s12_true_bucket_compile_cudagraph",
        "s13_deferred_transfer",
        "s14_reduced_embed_tokens",
        "s15_compile_reduce_overhead",
        "s16_compile_max_autotune",
    }
    strategy_prefetch = args.aaflow_strategy in {
        "default",
        "s2_prefetch",
        "s3_faiss_overlap",
        "s4_compile",
        "s5_compile_cudagraph",
        "s6_token_budget",
        "s7_bucketed_compile_cudagraph",
        "s8_length_bucket_eager",
        "s9_length_bucket_compile",
        "s10_length_bucket_compile_cudagraph",
        "s11_true_bucket_compile",
        "s12_true_bucket_compile_cudagraph",
        "s13_deferred_transfer",
        "s14_reduced_embed_tokens",
        "s15_compile_reduce_overhead",
        "s16_compile_max_autotune",
    }
    strategy_overlap = args.aaflow_strategy in {
        "default",
        "s3_faiss_overlap",
        "s4_compile",
        "s5_compile_cudagraph",
        "s6_token_budget",
        "s7_bucketed_compile_cudagraph",
        "s8_length_bucket_eager",
        "s9_length_bucket_compile",
        "s10_length_bucket_compile_cudagraph",
        "s11_true_bucket_compile",
        "s12_true_bucket_compile_cudagraph",
        "s14_reduced_embed_tokens",
        "s15_compile_reduce_overhead",
        "s16_compile_max_autotune",
    } and not args.aaflow_disable_faiss_overlap
    strategy_defer_transfer = args.aaflow_strategy == "s13_deferred_transfer"
    strategy_embed_max_input_tokens = (
        min(args.max_input_tokens, args.aaflow_reduced_embed_tokens)
        if args.aaflow_strategy == "s14_reduced_embed_tokens"
        else args.max_input_tokens
    )
    if args.aaflow_optimized_only:
        results.append(
            run_aaflow(
                engine,
                paths,
                args,
                rank,
                config=strategy_config,
                length_bucket=strategy_length_bucket,
                prefetch_tokenization=strategy_prefetch,
                embed_backend=strategy_backend,
                overlap_faiss=strategy_overlap,
                embedding_schedule=strategy_embedding_schedule,
                defer_embedding_transfer=strategy_defer_transfer,
                embed_max_input_tokens=strategy_embed_max_input_tokens,
            )
        )
    elif args.aaflow_tuning_only:
        results.extend(
            [
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    config="AAFLOW-baseline",
                    embed_batch_size=args.embed_batch_size,
                    generation_batch_size=args.generation_batch_size,
                    length_bucket=False,
                ),
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    config="AAFLOW-embed-batched",
                    embed_batch_size=args.aaflow_embed_batch_size,
                    generation_batch_size=args.generation_batch_size,
                    length_bucket=True,
                ),
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    config="AAFLOW-generation-batched",
                    embed_batch_size=args.embed_batch_size,
                    generation_batch_size=args.aaflow_generation_batch_size,
                    length_bucket=True,
                ),
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    config="AAFLOW-prefetch",
                    embed_batch_size=args.embed_batch_size,
                    generation_batch_size=args.generation_batch_size,
                    length_bucket=False,
                    prefetch_tokenization=True,
                ),
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    config="AAFLOW-prefetch-static-cache",
                    embed_batch_size=args.embed_batch_size,
                    generation_batch_size=args.generation_batch_size,
                    length_bucket=False,
                    prefetch_tokenization=True,
                    static_cache=True,
                ),
                run_aaflow(
                    engine,
                    paths,
                    args,
                    rank,
                    length_bucket=False,
                    prefetch_tokenization=True,
                    embed_backend=args.aaflow_embed_backend,
                ),
            ]
        )
    else:
        results.append(
            run_standard_config(
                "AsyncParallelOnly",
                load_sequential,
                transform_sequential,
                lambda chunks: asyncio.run(
                    embed_async_batches(
                        engine,
                        chunks,
                        args.embed_batch_size,
                        args.max_input_tokens,
                    )
                ),
                engine,
                paths,
                args.generation_samples_per_gpu,
                args.generation_batch_size,
                args.max_input_tokens,
                args.max_new_tokens,
                rank,
            )
        )
        results.append(run_dask_config(engine, paths, args, rank))
        results.append(run_ray_config(engine, paths, args, rank))
        if not args.disable_bsp:
            results.append(
                run_standard_config(
                    "BulkSynchronousParallelRAG",
                    load_threaded,
                    transform_threaded,
                    lambda chunks: embed_batches(
                        engine,
                        chunks,
                        args.embed_batch_size,
                        args.max_input_tokens,
                    ),
                    engine,
                    paths,
                    args.generation_samples_per_gpu,
                    args.generation_batch_size,
                    args.max_input_tokens,
                    args.max_new_tokens,
                    rank,
                )
            )
        results.append(
            run_standard_config(
                "HigressRAG",
                load_threaded,
                transform_threaded,
                lambda chunks: embed_batches(
                    engine,
                    chunks,
                    args.embed_batch_size,
                    args.max_input_tokens,
                ),
                engine,
                paths,
                args.generation_samples_per_gpu,
                args.generation_batch_size,
                args.max_input_tokens,
                args.max_new_tokens,
                rank,
            )
        )
        results.append(
            run_aaflow(
                engine,
                paths,
                args,
                rank,
                config=strategy_config,
                length_bucket=strategy_length_bucket,
                prefetch_tokenization=strategy_prefetch,
                embed_backend=strategy_backend,
                overlap_faiss=strategy_overlap,
                embedding_schedule=strategy_embedding_schedule,
            )
        )

    results_dir = run_dir / "rank_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"rank_{rank:04d}.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    (results_dir / f"rank_{rank:04d}_meta.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "model_load_s": engine.model_load_s,
                "corpus_prep_s": corpus_prep_s,
                "compiled_embed_setup_s": engine.compiled_embed_setup_s,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "attn_implementation": getattr(
                    engine.model.config,
                    "_attn_implementation",
                    args.attn_implementation,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    barrier(run_dir, "results_ready", rank, world)

    if rank == 0:
        load_times = [
            json.loads(path.read_text(encoding="utf-8"))["model_load_s"]
            for path in sorted(results_dir.glob("rank_*_meta.json"))
        ]
        aggregate(run_dir, world, model_name, model_id, max(load_times))
        (run_dir / "config.json").write_text(
            json.dumps({**vars(args), "world_size": world, "model_id": model_id}, indent=2),
            encoding="utf-8",
        )

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
