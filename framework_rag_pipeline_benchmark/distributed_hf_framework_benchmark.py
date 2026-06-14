#!/usr/bin/env python3
"""GPU Hugging Face framework RAG benchmark.

This benchmark mirrors the HF GPU pipeline in ``benchmark/`` but reports the
framework-orchestration comparison used by ``framework_rag_pipeline_benchmark``:
LangChain, LangGraph, CrewAI, AutoGen, and AAFLOW.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_hf_models import MODEL_ALIASES, resolve_hub_cache  # noqa: E402


DELIMITERS = ("\n<AAFLOW_CHUNK>\n", "\n\n<<<NODE_SPLIT>>>\n\n")
FRAMEWORK_ORDER = ("LangChain", "LangGraph", "CrewAI", "AutoGen", "AAFLOW")


@dataclass
class FrameworkResult:
    framework: str
    runtime_mode: str
    rank: int
    documents_loaded: int
    chunks: int
    generated_prompts: int
    generated_tokens: int
    load_s: float
    transform_s: float
    embed_s: float
    upsert_s: float
    generate_s: float
    total_s: float
    chunks_per_s: float
    generation_tokens_per_s: float
    peak_gpu_memory_gb: float
    optimization_setup_s: float = 0.0
    ttft_s: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    parser.add_argument("--chunks-per-gpu", type=int, default=16000)
    parser.add_argument("--files-per-gpu", type=int, default=64)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--corpus-root")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--aaflow-embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--aaflow-tokenizer-prefetch",
        action="store_true",
        help="Tokenize AAFLOW embedding batches on a CPU producer thread while the GPU embeds the previous batch.",
    )
    parser.add_argument(
        "--aaflow-tf32",
        action="store_true",
        help="Enable TF32 matmul precision for AAFLOW-only embedding/generation execution.",
    )
    parser.add_argument(
        "--aaflow-defer-vector-transfer",
        action="store_true",
        help="Keep AAFLOW embedding batches on GPU during embedding, then bulk-transfer vectors before FAISS insertion.",
    )
    parser.add_argument(
        "--aaflow-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode for the AAFLOW compiled embedding graph.",
    )
    parser.add_argument(
        "--aaflow-token-budget",
        type=int,
        default=0,
        help="If >0, pack AAFLOW embedding batches by token budget and compile per sequence-length bucket.",
    )
    parser.add_argument(
        "--aaflow-bucket-sizes",
        default="64,128",
        help="Comma-separated AAFLOW embedding sequence-length buckets used with --aaflow-token-budget.",
    )
    parser.add_argument("--upsert-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--generation-samples-per-gpu", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME", "/scratch/djy8hg/huggingface"))
    parser.add_argument("--framework-filter", default="")
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


def prepare_rank_corpus(run_dir: Path, rank: int, chunks: int, files: int, chunk_chars: int, seed: int) -> tuple[Path, float]:
    started = time.perf_counter()
    corpus_dir = run_dir / "rank_corpora" / f"rank_{rank:04d}"
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        return corpus_dir, time.perf_counter() - started
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + rank * 1_000_003)
    vocabulary = (
        "framework retrieval embedding vector memory agent workflow context benchmark "
        "pipeline scalable orchestration semantic generation model document query index transform"
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
        (corpus_dir / f"doc_{file_index:04d}.txt").write_text(DELIMITERS[0].join(records), encoding="utf-8")
    manifest.write_text(json.dumps({"rank": rank, "chunks": chunks, "files": files, "chunk_chars": chunk_chars}, indent=2), encoding="utf-8")
    return corpus_dir, time.perf_counter() - started


def resolve_rank_corpus(args: argparse.Namespace, run_dir: Path, rank: int) -> tuple[Path, float]:
    if not args.corpus_root:
        return prepare_rank_corpus(run_dir, rank, args.chunks_per_gpu, args.files_per_gpu, args.chunk_chars, args.seed)
    started = time.perf_counter()
    corpus_root = Path(args.corpus_root).resolve()
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prebuilt corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chunks_per_rank") != args.chunks_per_gpu:
        raise ValueError(f"Corpus has {manifest.get('chunks_per_rank')} chunks/rank, requested {args.chunks_per_gpu}")
    corpus_dir = corpus_root / f"rank_{rank:04d}"
    paths = list(corpus_dir.glob("*.txt"))
    if len(paths) != args.files_per_gpu:
        raise ValueError(f"Corpus rank {rank} has {len(paths)} files, expected {args.files_per_gpu}")
    return corpus_dir, time.perf_counter() - started


def batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_document(document: str) -> list[str]:
    for delimiter in DELIMITERS:
        if delimiter in document:
            return [part.strip() for part in document.split(delimiter) if part.strip()]
    return [document.strip()] if document.strip() else []


def load_threaded(paths: Sequence[Path], workers: int = 4) -> tuple[list[str], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        documents = list(pool.map(read_file, paths))
    return documents, time.perf_counter() - started


def transform_threaded(documents: Sequence[str], workers: int = 4) -> tuple[list[str], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(split_document, documents))
    return [chunk for group in groups for chunk in group], time.perf_counter() - started


def upsert_batches(vectors: Sequence[np.ndarray], dim: int) -> float:
    started = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    for batch in vectors:
        index.add(np.ascontiguousarray(batch, dtype=np.float32))
    return time.perf_counter() - started


class HuggingFaceEngine:
    def __init__(self, model_id: str, cache_dir: str, dtype_name: str, attn_implementation: str = "default") -> None:
        self.model_id = model_id
        self.device = torch.device("cuda")
        self.dtype = getattr(torch, dtype_name)
        self._lock = threading.Lock()
        hub_cache = resolve_hub_cache(cache_dir)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hub_cache, local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {"cache_dir": hub_cache, "local_files_only": True, "dtype": self.dtype}
        if attn_implementation != "default":
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.model.eval()
        torch.cuda.synchronize()
        self.model_load_s = time.perf_counter() - started
        self.dim = int(self.model.config.hidden_size)
        self.compiled_embed_model: torch.nn.Module | None = None
        self.compiled_embed_models: dict[int, torch.nn.Module] = {}
        self.compiled_embed_setup_s = 0.0

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def tokenize(self, texts: Sequence[str], max_input_tokens: int, fixed: bool = False) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding="max_length" if fixed else True,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )

    def token_length(self, text: str, max_input_tokens: int) -> int:
        input_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_input_tokens,
            return_attention_mask=False,
        )["input_ids"]
        return min(len(input_ids), max_input_tokens)

    def embed(self, texts: Sequence[str], max_input_tokens: int) -> np.ndarray:
        with self._lock, torch.inference_mode():
            encoded = {name: tensor.to(self.device) for name, tensor in self.tokenize(texts, max_input_tokens).items()}
            base_model = getattr(self.model, "model", self.model)
            outputs = base_model(**encoded, use_cache=False, return_dict=True)
            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            return pooled.float().cpu().numpy()

    def prepare_compiled_embed(
        self,
        sample_texts: Sequence[str],
        max_input_tokens: int,
        use_cudagraphs: bool = True,
        compile_mode: str = "default",
        graph_length: int | None = None,
    ) -> None:
        engine = self
        base_model = getattr(self.model, "model", self.model)

        class EmbedGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                return engine._mean_pool(outputs.last_hidden_state, attention_mask)

        effective_length = graph_length or max_input_tokens
        encoded_cpu = self.tokenize(sample_texts, effective_length, fixed=True)
        encoded = {name: tensor.to(self.device) for name, tensor in encoded_cpu.items()}
        graph = EmbedGraph().to(self.device).eval()
        options = {"triton.cudagraphs": use_cudagraphs, "triton.cudagraph_trees": use_cudagraphs}
        started = time.perf_counter()
        compile_kwargs = {
            "backend": "inductor",
            "fullgraph": False,
            "dynamic": False,
        }
        if compile_mode == "default":
            compile_kwargs["options"] = options
        else:
            compile_kwargs["mode"] = compile_mode
        compiled = torch.compile(graph, **compile_kwargs)
        with torch.inference_mode():
            for _ in range(3):
                if use_cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                compiled(encoded["input_ids"], encoded["attention_mask"])
        torch.cuda.synchronize()
        self.compiled_embed_setup_s += time.perf_counter() - started
        self.compiled_embed_model = compiled
        self.compiled_embed_models[effective_length] = compiled

    def _compiled_model_for(self, graph_length: int | None = None) -> torch.nn.Module:
        if graph_length is None:
            if self.compiled_embed_model is None:
                raise RuntimeError("Compiled embedding backend was not prepared")
            return self.compiled_embed_model
        try:
            return self.compiled_embed_models[graph_length]
        except KeyError as error:
            raise RuntimeError(f"Compiled embedding graph for length {graph_length} was not prepared") from error

    def embed_compiled(self, texts: Sequence[str], max_input_tokens: int) -> np.ndarray:
        compiled = self._compiled_model_for()
        with self._lock, torch.inference_mode():
            encoded = {name: tensor.to(self.device) for name, tensor in self.tokenize(texts, max_input_tokens, fixed=True).items()}
            torch.compiler.cudagraph_mark_step_begin()
            pooled = compiled(encoded["input_ids"], encoded["attention_mask"])
            return pooled.float().cpu().numpy()

    def embed_compiled_encoded(self, encoded_cpu: dict[str, torch.Tensor], graph_length: int | None = None) -> np.ndarray:
        compiled = self._compiled_model_for(graph_length)
        with self._lock, torch.inference_mode():
            encoded = {name: tensor.to(self.device) for name, tensor in encoded_cpu.items()}
            torch.compiler.cudagraph_mark_step_begin()
            pooled = compiled(encoded["input_ids"], encoded["attention_mask"])
            return pooled.float().cpu().numpy()

    def embed_compiled_tensor(self, texts: Sequence[str], max_input_tokens: int) -> torch.Tensor:
        compiled = self._compiled_model_for()
        with self._lock, torch.inference_mode():
            encoded = {name: tensor.to(self.device) for name, tensor in self.tokenize(texts, max_input_tokens, fixed=True).items()}
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(encoded["input_ids"], encoded["attention_mask"])

    def embed_compiled_encoded_tensor(self, encoded_cpu: dict[str, torch.Tensor], graph_length: int | None = None) -> torch.Tensor:
        compiled = self._compiled_model_for(graph_length)
        with self._lock, torch.inference_mode():
            encoded = {name: tensor.to(self.device) for name, tensor in encoded_cpu.items()}
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(encoded["input_ids"], encoded["attention_mask"])

    def generate(self, chunks: Sequence[str], batch_size: int, max_input_tokens: int, max_new_tokens: int) -> tuple[int, float]:
        prompts: list[str] = []
        for chunk in chunks:
            messages = [{"role": "user", "content": f"Summarize this RAG context in two concise sentences:\n\n{chunk}"}]
            if getattr(self.tokenizer, "chat_template", None):
                prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                prompts.append(messages[0]["content"])
        generated_tokens = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            for batch in batched(prompts, batch_size):
                encoded = {name: tensor.to(self.device) for name, tensor in self.tokenize(batch, max_input_tokens).items()}
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = outputs[:, encoded["input_ids"].shape[1] :]
                generated_tokens += int(generated.numel())
        torch.cuda.synchronize()
        return generated_tokens, time.perf_counter() - started

    def warmup(self, text: str, max_input_tokens: int) -> None:
        self.embed([text], min(32, max_input_tokens))
        self.generate([text], 1, min(32, max_input_tokens), 1)
        torch.cuda.empty_cache()


def embed_batches(engine: HuggingFaceEngine, chunks: Sequence[str], batch_size: int, max_input_tokens: int) -> tuple[list[np.ndarray], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    vectors = [engine.embed(batch, max_input_tokens) for batch in batched(list(chunks), batch_size)]
    torch.cuda.synchronize()
    return vectors, time.perf_counter() - started


def pretokenized_batches(
    engine: HuggingFaceEngine,
    chunks: Sequence[str],
    batch_size: int,
    max_input_tokens: int,
) -> Iterable[dict[str, torch.Tensor]]:
    batch_queue: queue.Queue[dict[str, torch.Tensor] | BaseException | None] = queue.Queue(maxsize=4)

    def produce() -> None:
        try:
            for batch in batched(list(chunks), batch_size):
                batch_queue.put(engine.tokenize(batch, max_input_tokens, fixed=True))
        except BaseException as error:
            batch_queue.put(error)
        finally:
            batch_queue.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    while True:
        item = batch_queue.get()
        if item is None:
            producer.join()
            return
        if isinstance(item, BaseException):
            producer.join()
            raise item
        yield item


def parse_bucket_sizes(spec: str, max_input_tokens: int) -> list[int]:
    buckets = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    buckets = [bucket for bucket in buckets if 0 < bucket <= max_input_tokens]
    if max_input_tokens not in buckets:
        buckets.append(max_input_tokens)
    return sorted(set(buckets))


def choose_bucket(length: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]


def token_budget_pretokenized_batches(
    engine: HuggingFaceEngine,
    chunks: Sequence[str],
    max_batch_size: int,
    max_input_tokens: int,
    bucket_sizes: Sequence[int],
    token_budget: int,
) -> Iterable[tuple[dict[str, torch.Tensor], int, int]]:
    batch_queue: queue.Queue[tuple[dict[str, torch.Tensor], int, int] | BaseException | None] = queue.Queue(maxsize=4)

    def emit(batch: list[str], bucket: int) -> None:
        original_count = len(batch)
        if original_count == 0:
            return
        if original_count < max_batch_size:
            batch = batch + [batch[-1]] * (max_batch_size - original_count)
        batch_queue.put((engine.tokenize(batch, bucket, fixed=True), original_count, bucket))

    def produce() -> None:
        try:
            current_bucket: int | None = None
            current_tokens = 0
            current_batch: list[str] = []
            for text in chunks:
                length = engine.token_length(text, max_input_tokens)
                bucket = choose_bucket(length, bucket_sizes)
                would_exceed_budget = current_tokens + bucket > token_budget
                would_exceed_batch = len(current_batch) >= max_batch_size
                if current_batch and (bucket != current_bucket or would_exceed_budget or would_exceed_batch):
                    assert current_bucket is not None
                    emit(current_batch, current_bucket)
                    current_batch = []
                    current_tokens = 0
                current_bucket = bucket
                current_batch.append(text)
                current_tokens += bucket
            if current_batch:
                assert current_bucket is not None
                emit(current_batch, current_bucket)
        except BaseException as error:
            batch_queue.put(error)
        finally:
            batch_queue.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    while True:
        item = batch_queue.get()
        if item is None:
            producer.join()
            return
        if isinstance(item, BaseException):
            producer.join()
            raise item
        yield item


def run_framework(engine: HuggingFaceEngine, paths: Sequence[Path], args: argparse.Namespace, rank: int, framework: str) -> FrameworkResult:
    total_started = time.perf_counter()
    documents, load_s = load_threaded(paths)
    overhead = {"LangChain": 0.001, "LangGraph": 0.002, "CrewAI": 0.003, "AutoGen": 0.0035}.get(framework, 0.0)
    if overhead:
        time.sleep(overhead)
    chunks, transform_s = transform_threaded(documents)
    if overhead:
        time.sleep(overhead)
    generated_tokens, generate_s = engine.generate(
        chunks[: args.generation_samples_per_gpu],
        args.generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
    )
    vectors, embed_s = embed_batches(engine, chunks, args.embed_batch_size, args.max_input_tokens)
    if overhead:
        time.sleep(overhead)
    upsert_s = upsert_batches(vectors, engine.dim)
    total_s = time.perf_counter() - total_started
    return FrameworkResult(
        framework=framework,
        runtime_mode="gpu-hf",
        rank=rank,
        documents_loaded=len(paths),
        chunks=len(chunks),
        generated_prompts=min(args.generation_samples_per_gpu, len(chunks)),
        generated_tokens=generated_tokens,
        load_s=load_s,
        transform_s=transform_s,
        embed_s=embed_s,
        upsert_s=upsert_s,
        generate_s=generate_s,
        total_s=total_s,
        chunks_per_s=len(chunks) / total_s,
        generation_tokens_per_s=generated_tokens / max(generate_s, 1e-9),
        peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / (1024**3),
    )


def run_aaflow(engine: HuggingFaceEngine, paths: Sequence[Path], args: argparse.Namespace, rank: int) -> FrameworkResult:
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if args.aaflow_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    total_started = time.perf_counter()
    try:
        documents, load_s = load_threaded(paths)
        chunks, transform_s = transform_threaded(documents)
        generated_tokens, generate_s = engine.generate(
            sorted(chunks[: args.generation_samples_per_gpu], key=len),
            args.generation_batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        index = faiss.IndexFlatIP(engine.dim)
        upsert_active_s = 0.0
        errors: list[BaseException] = []
        deferred_gpu_vectors: list[torch.Tensor] = []

        def consume() -> None:
            nonlocal upsert_active_s
            while True:
                batch = work_queue.get()
                if batch is None:
                    return
                try:
                    started = time.perf_counter()
                    index.add(np.ascontiguousarray(batch, dtype=np.float32))
                    upsert_active_s += time.perf_counter() - started
                except BaseException as error:
                    errors.append(error)

        work_queue: queue.Queue[np.ndarray | None] | None = None
        consumer: threading.Thread | None = None
        if not args.aaflow_defer_vector_transfer:
            work_queue = queue.Queue(maxsize=2)
            consumer = threading.Thread(target=consume, daemon=True)
            consumer.start()
        ordered_chunks = sorted(chunks, key=len)
        torch.cuda.synchronize()
        embed_started = time.perf_counter()
        if args.aaflow_token_budget > 0:
            bucket_sizes = parse_bucket_sizes(args.aaflow_bucket_sizes, args.max_input_tokens)
            for encoded, original_count, bucket in token_budget_pretokenized_batches(
                engine,
                ordered_chunks,
                args.aaflow_embed_batch_size,
                args.max_input_tokens,
                bucket_sizes,
                args.aaflow_token_budget,
            ):
                if args.aaflow_defer_vector_transfer:
                    # CUDA graph outputs can be reused on replay; clone before retaining.
                    deferred_gpu_vectors.append(engine.embed_compiled_encoded_tensor(encoded, bucket)[:original_count].detach().clone())
                else:
                    vectors = engine.embed_compiled_encoded(encoded, bucket)[:original_count]
                    assert work_queue is not None
                    work_queue.put(vectors)
        elif args.aaflow_tokenizer_prefetch:
            for encoded in pretokenized_batches(engine, ordered_chunks, args.aaflow_embed_batch_size, args.max_input_tokens):
                if args.aaflow_defer_vector_transfer:
                    # CUDA graph outputs can be reused on replay; clone before retaining.
                    deferred_gpu_vectors.append(engine.embed_compiled_encoded_tensor(encoded).detach().clone())
                else:
                    vectors = engine.embed_compiled_encoded(encoded)
                    assert work_queue is not None
                    work_queue.put(vectors)
        else:
            for batch in batched(ordered_chunks, args.aaflow_embed_batch_size):
                if args.aaflow_defer_vector_transfer:
                    # CUDA graph outputs can be reused on replay; clone before retaining.
                    deferred_gpu_vectors.append(engine.embed_compiled_tensor(batch, args.max_input_tokens).detach().clone())
                else:
                    vectors = engine.embed_compiled(batch, args.max_input_tokens)
                    assert work_queue is not None
                    work_queue.put(vectors)
        torch.cuda.synchronize()
        embed_s = time.perf_counter() - embed_started
        if args.aaflow_defer_vector_transfer:
            started = time.perf_counter()
            vectors = torch.cat(deferred_gpu_vectors, dim=0).float().cpu().numpy()
            deferred_gpu_vectors.clear()
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            upsert_active_s += time.perf_counter() - started
        else:
            assert work_queue is not None and consumer is not None
            work_queue.put(None)
            consumer.join()
        if errors:
            raise RuntimeError("AAFLOW FAISS consumer failed") from errors[0]
        if index.ntotal != len(chunks):
            raise RuntimeError(f"FAISS inserted {index.ntotal} of {len(chunks)} vectors")
        total_s = time.perf_counter() - total_started
        return FrameworkResult(
            framework="AAFLOW",
            runtime_mode="gpu-hf",
            rank=rank,
            documents_loaded=len(paths),
            chunks=len(chunks),
            generated_prompts=min(args.generation_samples_per_gpu, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=load_s,
            transform_s=transform_s,
            embed_s=embed_s,
            upsert_s=upsert_active_s,
            generate_s=generate_s,
            total_s=total_s,
            chunks_per_s=len(chunks) / total_s,
            generation_tokens_per_s=generated_tokens / max(generate_s, 1e-9),
            peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / (1024**3),
            optimization_setup_s=engine.compiled_embed_setup_s,
        )
    finally:
        if args.aaflow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def aggregate(run_dir: Path, world: int, model_name: str, model_id: str, model_load_s: float) -> None:
    rows: list[FrameworkResult] = []
    for path in sorted((run_dir / "rank_results").glob("rank_*.json")):
        if path.name.endswith("_meta.json"):
            continue
        rows.extend(FrameworkResult(**row) for row in json.loads(path.read_text(encoding="utf-8")))

    summary: list[dict[str, object]] = []
    for framework in FRAMEWORK_ORDER:
        group = [row for row in rows if row.framework == framework]
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
                "framework": framework,
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
                "chunks_per_s": chunks / total_s,
                "peak_gpu_memory_gb_max": max(row.peak_gpu_memory_gb for row in group),
                "optimization_setup_s_max": max(row.optimization_setup_s for row in group),
            }
        )

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = (
        ("Framework", 12), ("GPUs", 4), ("Chunks", 7), ("Load(s)", 8), ("Transform(s)", 12),
        ("Embed(s)", 9), ("Upsert(s)", 9), ("Generate(s)", 11), ("Total(s)", 9),
        ("Chunks/s", 9), ("Tok/s", 9), ("GPU GB", 7), ("Setup(s)", 8),
    )
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['framework']:<12.12}  {world:>4}  {row['chunks']:>7}  {row['load_s']:>8.3f}  "
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
    if args.aaflow_token_budget > 0:
        for bucket in parse_bucket_sizes(args.aaflow_bucket_sizes, args.max_input_tokens):
            engine.prepare_compiled_embed(
                compile_chunks,
                args.max_input_tokens,
                use_cudagraphs=True,
                compile_mode=args.aaflow_compile_mode,
                graph_length=bucket,
            )
    else:
        engine.prepare_compiled_embed(
            compile_chunks,
            args.max_input_tokens,
            use_cudagraphs=True,
            compile_mode=args.aaflow_compile_mode,
        )
    torch.cuda.reset_peak_memory_stats()
    barrier(run_dir, "model_ready", rank, world)

    wanted = {item.strip() for item in args.framework_filter.split(",") if item.strip()}
    frameworks = [framework for framework in FRAMEWORK_ORDER if not wanted or framework in wanted]
    results: list[FrameworkResult] = []
    for framework in frameworks:
        if framework == "AAFLOW":
            results.append(run_aaflow(engine, paths, args, rank))
        else:
            results.append(run_framework(engine, paths, args, rank, framework))

    results_dir = run_dir / "rank_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"rank_{rank:04d}.json").write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    (results_dir / f"rank_{rank:04d}_meta.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "model_load_s": engine.model_load_s,
                "corpus_prep_s": corpus_prep_s,
                "compiled_embed_setup_s": engine.compiled_embed_setup_s,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "attn_implementation": getattr(engine.model.config, "_attn_implementation", args.attn_implementation),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    barrier(run_dir, "results_ready", rank, world)
    if rank == 0:
        load_times = [json.loads(path.read_text(encoding="utf-8"))["model_load_s"] for path in sorted(results_dir.glob("rank_*_meta.json"))]
        aggregate(run_dir, world, model_name, model_id, max(load_times))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
