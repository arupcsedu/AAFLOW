"""KVCOMM baseline adapter and analytic profile model.

This module maps the public FastMAS/KVCOMM implementation into AAFLOW+'s
benchmark schema without vendoring KVCOMM. KVCOMM is a separate HF-based
multi-agent serving stack whose implementation uses:

* anchor matching to select cached examples for a shared placeholder segment,
* offset approximation to transform anchor KV deviations for a new context,
* anchor prediction/update to decide whether newly generated KV should enter
  the online anchor pool.

When a real KVCOMM checkout is supplied through `KVCOMM_REPO`, callers may use
that repository's CLI separately. For the full-paper matrix here, this adapter
provides a KVCOMM-style measured-profile baseline: it consumes the same real
model prefill/decode timings as the other baselines, then applies a partial
cross-context KV reuse model plus explicit anchor-processing overhead.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KVCOMM_BASELINE_NAME = "kvcomm_prefix"
KVCOMM_BASELINE_LABEL = "KVCOMM Anchor Reuse"


@dataclass(frozen=True)
class KVCOMMProfile:
    """Computed KVCOMM-style cost profile for one benchmark row."""

    ttft_sec: float
    total_latency_sec: float
    prefill_sec: float
    decode_sec: float
    transfer_sec: float
    resume_sec: float
    omega_sec: float
    throughput_tokens_per_sec: float
    kv_peak_bytes: int
    kv_transferred_bytes: int
    kv_reuse_ratio: float
    materialize_count: int
    transfer_count: int
    dense_prefill_sec: float
    stateful_prefill_sec: float
    anchor_overhead_sec: float
    reuse_fraction: float


def kvcomm_available(repo_path: str | None = None) -> bool:
    """Return whether a KVCOMM checkout or importable package is available."""

    if importlib.util.find_spec("KVCOMM") is not None:
        return True
    root = Path(repo_path or os.environ.get("KVCOMM_REPO", "")).expanduser()
    return bool(root and (root / "KVCOMM" / "llm" / "kvcomm_engine.py").exists())


def kvcomm_unavailable_reason(repo_path: str | None = None) -> str:
    """Return a concise availability message for CLI listings."""

    root = repo_path or os.environ.get("KVCOMM_REPO", "")
    if root:
        return f"KVCOMM checkout not found or incomplete at {root}"
    return "KVCOMM is not importable; set KVCOMM_REPO to a FastMAS/KVCOMM checkout for real CLI experiments"


def kvcomm_profile(
    *,
    prefill_sec: float,
    decode_sec: float,
    first_token_decode_sec: float,
    kv_bytes: int,
    output_total: int,
    branch_instances: int,
    num_agents: int,
    branch_factor: int,
    num_prompts: int,
    dense_prefill_sec: float,
    omega_state_sec: float,
    omega_text_sec: float,
    reuse_fraction: float | None = None,
    anchor_overhead_fraction: float | None = None,
) -> KVCOMMProfile:
    """Estimate a KVCOMM-style row from measured model timing.

    The default reuse target is 0.70, matching KVCOMM's reported high reuse
    regime without assuming full exact-prefix reuse. Anchor overhead is charged
    as a small fraction of measured prefill per request/branch, representing
    anchor matching, offset approximation, and online anchor prediction.
    """

    reuse = _bounded(
        reuse_fraction if reuse_fraction is not None else float(os.environ.get("KVCOMM_REUSE_FRACTION", "0.70")),
        0.0,
        0.98,
    )
    overhead_fraction = _bounded(
        anchor_overhead_fraction
        if anchor_overhead_fraction is not None
        else float(os.environ.get("KVCOMM_ANCHOR_OVERHEAD_FRACTION", "0.04")),
        0.0,
        1.0,
    )

    instances = max(1, int(branch_instances))
    prompts = max(1, int(num_prompts))
    suffix_fraction = 1.0 - reuse
    anchor_calls = max(0, instances - 1) * prompts
    anchor_overhead_sec = anchor_calls * prefill_sec * overhead_fraction

    # One dense prefix establishes anchors; dependent agents prefill only the
    # non-reused suffix and pay anchor processing. This is local cross-context
    # communication, so there is no distributed transfer cost.
    prefill_total = prompts * prefill_sec + anchor_calls * prefill_sec * suffix_fraction
    decode_total = instances * prompts * decode_sec
    omega_sec = instances * prompts * omega_state_sec + anchor_overhead_sec
    total = prefill_total + decode_total + omega_sec
    ttft = prefill_sec * suffix_fraction + first_token_decode_sec + (prefill_sec * overhead_fraction) + omega_text_sec
    peak_bytes = int(kv_bytes * (1.0 + max(0, instances - 1) * suffix_fraction))

    return KVCOMMProfile(
        ttft_sec=ttft,
        total_latency_sec=total,
        prefill_sec=prefill_total,
        decode_sec=decode_total,
        transfer_sec=0.0,
        resume_sec=0.0,
        omega_sec=omega_sec,
        throughput_tokens_per_sec=output_total / total if total > 0 else 0.0,
        kv_peak_bytes=peak_bytes,
        kv_transferred_bytes=0,
        kv_reuse_ratio=((instances - 1) * reuse / instances) if instances > 0 else 0.0,
        materialize_count=prompts,
        transfer_count=0,
        dense_prefill_sec=dense_prefill_sec,
        stateful_prefill_sec=prefill_total,
        anchor_overhead_sec=anchor_overhead_sec,
        reuse_fraction=reuse,
    )


def kvcomm_metadata(repo_path: str | None = None) -> dict[str, Any]:
    """Return baseline metadata suitable for result rows."""

    available = kvcomm_available(repo_path)
    return {
        "kvcomm_available": available,
        "kvcomm_mode": "external_available" if available else "measured_profile_simulation",
        "kvcomm_repo": str(repo_path or os.environ.get("KVCOMM_REPO", "")),
        "kvcomm_reference": "https://github.com/FastMAS/KVCOMM",
    }


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
