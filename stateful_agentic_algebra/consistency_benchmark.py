"""Consistency benchmark for dense vs KV-cached deterministic decoding.

For each prompt, the benchmark:
  1. runs dense full-context generation,
  2. materializes prefix `past_key_values`,
  3. resumes greedy decode from the KV cache,
  4. compares generated token IDs and text agreement.

The module is designed for real HF models but degrades gracefully: if the model
cannot be loaded, it still writes `consistency.csv` and
`consistency_summary.json` with skipped rows and the failure reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .hf_kv_backend import HFBackendConfig, HFKVBackend


CSV_FIELDS = [
    "run_id",
    "model_id",
    "prompt_index",
    "context_tokens",
    "output_tokens",
    "available",
    "skipped",
    "reason",
    "exact_match",
    "exact_token_match_rate",
    "first_divergence_position",
    "normalized_edit_distance",
    "output_agreement_rate",
    "dense_token_count",
    "cached_token_count",
    "dense_text",
    "cached_text",
    "possible_mismatch_causes",
]


@dataclass
class ConsistencyResult:
    """Comparison result for one prompt."""

    dense_token_ids: list[int]
    cached_token_ids: list[int]
    dense_text: str
    cached_text: str
    exact_match: bool
    exact_token_match_rate: float
    first_divergence_position: Optional[int]
    normalized_edit_distance: float
    output_agreement_rate: float
    possible_mismatch_causes: list[str]


def compare_outputs(
    dense_token_ids: list[int],
    cached_token_ids: list[int],
    dense_text: str = "",
    cached_text: str = "",
    possible_mismatch_causes: Optional[list[str]] = None,
) -> ConsistencyResult:
    """Compare dense and cached outputs using token and text metrics."""

    max_len = max(len(dense_token_ids), len(cached_token_ids), 1)
    shared_len = min(len(dense_token_ids), len(cached_token_ids))
    matching_prefix = sum(1 for idx in range(shared_len) if dense_token_ids[idx] == cached_token_ids[idx])
    exact_match = dense_token_ids == cached_token_ids
    divergence = first_divergence_position(dense_token_ids, cached_token_ids)
    edit_distance = levenshtein_distance(dense_token_ids, cached_token_ids)
    normalized_edit = edit_distance / max_len
    text_agreement = 1.0 if dense_text == cached_text else 0.0
    return ConsistencyResult(
        dense_token_ids=list(dense_token_ids),
        cached_token_ids=list(cached_token_ids),
        dense_text=dense_text,
        cached_text=cached_text,
        exact_match=exact_match,
        exact_token_match_rate=matching_prefix / max_len,
        first_divergence_position=divergence,
        normalized_edit_distance=normalized_edit,
        output_agreement_rate=1.0 if exact_match else text_agreement,
        possible_mismatch_causes=list(possible_mismatch_causes or []),
    )


def first_divergence_position(left: list[int], right: list[int]) -> Optional[int]:
    """Return the first differing token index, or None when exactly equal."""

    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def levenshtein_distance(left: list[int], right: list[int]) -> int:
    """Compute Levenshtein edit distance over token IDs."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_id in enumerate(left, start=1):
        current = [i]
        for j, right_id in enumerate(right, start=1):
            cost = 0 if left_id == right_id else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def run_consistency_benchmark(
    *,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    num_prompts: int,
    output_dir: str | Path,
    device: str = "auto",
    local_files_only: bool = False,
    seed: int = 7,
) -> dict[str, Any]:
    """Run the benchmark and write CSV/summary outputs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    backend = HFKVBackend(
        HFBackendConfig(
            model_id=model_id,
            tokenizer_id=model_id,
            device=device,
            local_files_only=local_files_only,
            seed=seed,
        )
    )
    load_reason = ""
    try:
        backend.load_model()
    except Exception as exc:
        load_reason = str(exc)

    for prompt_index in range(max(1, int(num_prompts))):
        run_id = f"{_safe_name(model_id)}_{prompt_index:04d}"
        if load_reason:
            rows.append(_skip_row(run_id, model_id, prompt_index, context_tokens, output_tokens, load_reason))
            continue
        try:
            prompt = _prompt_for_index(backend, context_tokens, prompt_index)
            dense = backend.run_dense_prefill_decode(prompt, output_tokens=output_tokens)
            prefill = backend.run_prefill(prompt)
            cached = backend.run_decode_with_cache(
                prefill.past_key_values,
                output_tokens=output_tokens,
                next_token_id=prefill.next_token_id,
            )
            causes = infer_possible_mismatch_causes(backend)
            comparison = compare_outputs(
                dense.generated_token_ids,
                cached.generated_token_ids,
                dense.generated_text,
                cached.generated_text,
                possible_mismatch_causes=causes,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "prompt_index": prompt_index,
                    "context_tokens": context_tokens,
                    "output_tokens": output_tokens,
                    "available": True,
                    "skipped": False,
                    "reason": "",
                    "exact_match": comparison.exact_match,
                    "exact_token_match_rate": comparison.exact_token_match_rate,
                    "first_divergence_position": (
                        "" if comparison.first_divergence_position is None else comparison.first_divergence_position
                    ),
                    "normalized_edit_distance": comparison.normalized_edit_distance,
                    "output_agreement_rate": comparison.output_agreement_rate,
                    "dense_token_count": len(comparison.dense_token_ids),
                    "cached_token_count": len(comparison.cached_token_ids),
                    "dense_text": comparison.dense_text,
                    "cached_text": comparison.cached_text,
                    "possible_mismatch_causes": ";".join(comparison.possible_mismatch_causes),
                }
            )
        except Exception as exc:
            rows.append(_skip_row(run_id, model_id, prompt_index, context_tokens, output_tokens, str(exc)))

    _write_csv(out / "consistency.csv", rows)
    summary = summarize_rows(rows)
    (out / "consistency_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def infer_possible_mismatch_causes(backend: HFKVBackend) -> list[str]:
    """Infer likely mismatch causes from model/tokenizer configuration."""

    causes: list[str] = []
    config = getattr(getattr(backend, "model", None), "config", None)
    if config is not None:
        if getattr(config, "sliding_window", None) or getattr(config, "use_sliding_window", False):
            causes.append("sliding window attention")
        if getattr(config, "rope_scaling", None) or getattr(config, "rope_theta", None):
            causes.append("position ids")
        torch_dtype = str(getattr(config, "torch_dtype", "") or "")
        if any(dtype in torch_dtype for dtype in ("float16", "bfloat16")):
            causes.append("dtype/numerical nondeterminism")
    if backend.config.tokenizer_id != backend.config.model_id:
        causes.append("tokenizer mismatch")
    causes.append("attention mask")
    causes.append("position ids")
    causes.append("dtype/numerical nondeterminism")
    return sorted(set(causes))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize consistency rows."""

    available = [row for row in rows if row.get("available") is True]
    skipped = [row for row in rows if row.get("skipped") is True]
    exact_rates = [_float(row.get("exact_token_match_rate")) for row in available]
    edit_rates = [_float(row.get("normalized_edit_distance")) for row in available]
    agreement = [_float(row.get("output_agreement_rate")) for row in available]
    causes = sorted(
        {
            cause
            for row in rows
            for cause in str(row.get("possible_mismatch_causes", "")).split(";")
            if cause
        }
    )
    return {
        "num_prompts": len(rows),
        "available_prompts": len(available),
        "skipped_prompts": len(skipped),
        "exact_match_count": sum(1 for row in available if row.get("exact_match") is True),
        "exact_match_rate": (sum(1 for row in available if row.get("exact_match") is True) / len(available)) if available else 0.0,
        "mean_exact_token_match_rate": statistics.fmean(exact_rates) if exact_rates else 0.0,
        "mean_normalized_edit_distance": statistics.fmean(edit_rates) if edit_rates else 0.0,
        "mean_output_agreement_rate": statistics.fmean(agreement) if agreement else 0.0,
        "possible_mismatch_causes": causes,
        "skip_reasons": sorted({str(row.get("reason", "")) for row in skipped if row.get("reason")}),
    }


def _prompt_for_index(backend: HFKVBackend, context_tokens: int, prompt_index: int) -> str:
    base = backend.build_prompt(context_tokens)
    return f"{base} Deterministic prompt index {prompt_index}."


def _skip_row(run_id: str, model_id: str, prompt_index: int, context_tokens: int, output_tokens: int, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": model_id,
        "prompt_index": prompt_index,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "available": False,
        "skipped": True,
        "reason": reason,
        "exact_match": "",
        "exact_token_match_rate": "",
        "first_divergence_position": "",
        "normalized_edit_distance": "",
        "output_agreement_rate": "",
        "dense_token_count": "",
        "cached_token_count": "",
        "dense_text": "",
        "cached_text": "",
        "possible_mismatch_causes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense vs cached deterministic decoding consistency")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_consistency_benchmark(
        model_id=args.model_id,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        device=args.device,
        local_files_only=args.hf_local_files_only,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": args.output_dir, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
