#!/usr/bin/env python3
"""Prebuild deterministic text shards from a public Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DELIMITER = "\n<AAFLOW_CHUNK>\n"


def chunk_counts(total: int, files: int) -> list[int]:
    counts = [total // files] * files
    for index in range(total % files):
        counts[index] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--subset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--chunks-per-rank", type=int, default=4096)
    parser.add_argument("--files-per-rank", type=int, default=64)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--cache-dir", default="/scratch/djy8hg/huggingface/datasets")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_chunks(dataset, text_column: str, count: int, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for row in dataset:
        text = " ".join(str(row[text_column]).split())
        if not text:
            continue
        buffer = f"{buffer} {text}".strip()
        while len(buffer) >= chunk_chars:
            boundary = buffer.rfind(" ", 0, chunk_chars + 1)
            if boundary < chunk_chars // 2:
                boundary = chunk_chars
            chunks.append(buffer[:boundary].strip())
            buffer = buffer[boundary:].strip()
            if len(chunks) == count:
                return chunks
    raise RuntimeError(f"Dataset produced only {len(chunks)} of {count} requested chunks")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    expected = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_column": args.text_column,
        "ranks": args.ranks,
        "chunks_per_rank": args.chunks_per_rank,
        "files_per_rank": args.files_per_rank,
        "chunk_chars": args.chunk_chars,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(actual.get(key) == value for key, value in expected.items()):
            print(f"Corpus already exists: {output_dir}")
            return 0
        raise RuntimeError(f"Existing corpus manifest does not match request: {manifest_path}")

    dataset = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    total_chunks = args.ranks * args.chunks_per_rank
    chunks = build_chunks(dataset, args.text_column, total_chunks, args.chunk_chars)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in range(args.ranks):
        rank_dir = output_dir / f"rank_{rank:04d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        start = rank * args.chunks_per_rank
        rank_chunks = chunks[start : start + args.chunks_per_rank]
        offset = 0
        for file_index, file_chunks in enumerate(
            chunk_counts(args.chunks_per_rank, args.files_per_rank)
        ):
            records = rank_chunks[offset : offset + file_chunks]
            (rank_dir / f"doc_{file_index:04d}.txt").write_text(
                DELIMITER.join(records),
                encoding="utf-8",
            )
            offset += file_chunks

    manifest = {
        **expected,
        "total_chunks": total_chunks,
        "source_rows": len(dataset),
        "format": "UTF-8 text files containing AAFLOW-delimited dataset chunks",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared corpus: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
