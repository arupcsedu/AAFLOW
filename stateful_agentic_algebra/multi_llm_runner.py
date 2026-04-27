"""Run Stateful Agentic Algebra benchmark matrices across real LLMs.

The runner coordinates two measurement paths:

* HF backend: uses `HFKVBackend` to run real prefill/decode KV-cache
  microbenchmarks and extract KV metadata/byte counts.
* vLLM backend: delegates serving runs to `vllm_benchmark`, parsing TTFT, TPOT,
  ITL, E2EL, and throughput when vLLM is installed.
* SGLang backend: delegates serving runs to `sglang_benchmark` using an
  SGLang server and packaged or HTTP fallback benchmark path.

HF measurements are converted into comparable measured-profile paper rows for:

* `AAFLOW+`: measured HF prefill/decode/KV bytes plus state orchestration.
* `dense_prefill`: each branch/agent independently pays measured prefill.
* `aaflow_text`: text-passing baseline using the same measured dense costs.
* `vllm_local_prefix`: local prefix reuse using measured prefill/decode costs.
* `sglang_prefix`: SGLang-style local prefix reuse using the same profile.
* `distserve_style`: measured prefill/decode plus simulated disaggregation.

All optional backend failures are captured as skipped rows so a large sweep can
continue across gated models, missing packages, or unavailable vLLM servers.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .config_utils import bool_default, config_value, csv_default, load_config_file
from .hf_kv_backend import HFBackendConfig, HFKVBackend, HFMeasurement
from .model_registry import get_model_spec
from .sglang_benchmark import (
    check_sglang_available,
    launch_sglang_server,
    run_sglang_bench_serve,
    terminate_process_tree as terminate_sglang_process_tree,
    wait_for_server as wait_for_sglang_server,
)
from .vllm_benchmark import check_vllm_available, run_vllm_bench_serve, launch_vllm_server, wait_for_server


RESULT_FIELDS = [
    "run_id",
    "seed",
    "model_id",
    "backend",
    "workload_name",
    "context_tokens",
    "output_tokens",
    "num_agents",
    "branch_factor",
    "num_prompts",
    "available",
    "skipped",
    "reason",
    "ttft_sec",
    "total_latency_sec",
    "prefill_sec",
    "decode_sec",
    "transfer_sec",
    "resume_sec",
    "omega_sec",
    "throughput_tokens_per_sec",
    "kv_total_bytes",
    "kv_peak_bytes",
    "kv_transferred_bytes",
    "kv_reuse_ratio",
    "materialize_count",
    "transfer_count",
    "branch_instances",
    "dense_prefill_sec",
    "stateful_prefill_sec",
    "tpot_sec",
    "itl_sec",
    "e2el_sec",
    "request_throughput_req_per_sec",
    "source_metrics_path",
]

HF_MEASURED_BASELINES = (
    "AAFLOW+",
    "dense_prefill",
    "aaflow_text",
    "vllm_local_prefix",
    "sglang_prefix",
    "distserve_style",
)


@dataclass
class MultiLLMConfig:
    """Configuration for the multi-model benchmark matrix."""

    models: list[str]
    backends: list[str]
    context_grid: list[int]
    output_grid: list[int]
    agent_grid: list[int]
    branch_grid: list[int]
    num_prompts: int
    output_dir: str
    seeds: list[int]
    bandwidth_bytes_per_sec: float = 25_000_000_000.0
    network_latency_sec: float = 0.00005
    resume_overhead_sec: float = 0.0001
    omega_state_sec: float = 0.00005
    omega_text_sec: float = 0.00005
    vllm_port: int = 8000
    vllm_server_timeout_sec: float = 900.0
    vllm_bench_timeout_sec: float = 1800.0
    sglang_port: int = 30000
    sglang_server_timeout_sec: float = 900.0
    sglang_bench_timeout_sec: float = 1800.0
    sglang_python_bin: str = ""
    sglang_server_extra_args: str = ""
    tensor_parallel_size: int = 1
    dry_run: bool = False
    hf_device: str = "auto"
    hf_local_files_only: bool = False
    skip_invalid_context: bool = True
    progress: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_matrix(config: MultiLLMConfig) -> list[dict[str, Any]]:
    """Run the configured matrix and write raw/CSV/summary outputs."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    hf_cache: dict[tuple[str, int, int, int], HFMeasurement] = {}
    hf_backend_cache: dict[tuple[str, int, str, bool], HFKVBackend] = {}
    current_hf_model: Optional[str] = None
    raw_path = output_dir / "results_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_file:
        combos = list(
            itertools.product(
                config.models,
                config.backends,
                config.context_grid,
                config.output_grid,
                config.agent_grid,
                config.branch_grid,
                config.seeds,
            )
        )
        for combo_idx, (model_id, backend, context_tokens, output_tokens, num_agents, branch_factor, seed) in enumerate(
            combos, start=1
        ):
            if config.progress:
                _print_progress(
                    f"[{combo_idx}/{len(combos)}] model={model_id} backend={backend} "
                    f"context={context_tokens} output={output_tokens} agents={num_agents} "
                    f"branch={branch_factor} seed={seed}"
                )
            invalid_reason = _invalid_context_reason(model_id, context_tokens, output_tokens)
            if invalid_reason and config.skip_invalid_context:
                combo_rows = _skipped_context_rows(
                    model_id=model_id,
                    backend=backend,
                    context_tokens=context_tokens,
                    output_tokens=output_tokens,
                    num_agents=num_agents,
                    branch_factor=branch_factor,
                    num_prompts=config.num_prompts,
                    seed=seed,
                    reason=invalid_reason,
                )
                for row in combo_rows:
                    normalized = _normalize_row(row)
                    rows.append(normalized)
                    raw_file.write(json.dumps(normalized, sort_keys=True) + "\n")
                    raw_file.flush()
                if config.progress:
                    _print_progress(f"  skipped: {invalid_reason}")
                continue

            if backend == "hf":
                if current_hf_model is not None and current_hf_model != model_id:
                    hf_backend_cache.clear()
                    _release_torch_memory()
                current_hf_model = model_id
                combo_rows = _run_hf_combo(
                    config=config,
                    model_id=model_id,
                    context_tokens=context_tokens,
                    output_tokens=output_tokens,
                    num_agents=num_agents,
                    branch_factor=branch_factor,
                    seed=seed,
                    cache=hf_cache,
                    backend_cache=hf_backend_cache,
                    output_dir=output_dir,
                )
            elif backend == "vllm":
                combo_rows = [
                    _run_vllm_combo(
                        config=config,
                        model_id=model_id,
                        context_tokens=context_tokens,
                        output_tokens=output_tokens,
                        num_agents=num_agents,
                        branch_factor=branch_factor,
                        seed=seed,
                        output_dir=output_dir,
                    )
                ]
            elif backend == "sglang":
                combo_rows = [
                    _run_sglang_combo(
                        config=config,
                        model_id=model_id,
                        context_tokens=context_tokens,
                        output_tokens=output_tokens,
                        num_agents=num_agents,
                        branch_factor=branch_factor,
                        seed=seed,
                        output_dir=output_dir,
                    )
                ]
            else:
                combo_rows = [
                    _base_row(
                        model_id=model_id,
                        backend=backend,
                        workload_name="unknown_backend",
                        context_tokens=context_tokens,
                        output_tokens=output_tokens,
                        num_agents=num_agents,
                        branch_factor=branch_factor,
                        num_prompts=config.num_prompts,
                        seed=seed,
                        available=False,
                        skipped=True,
                        reason=f"unsupported backend {backend!r}",
                    )
                ]

            for row in combo_rows:
                normalized = _normalize_row(row)
                rows.append(normalized)
                raw_file.write(json.dumps(normalized, sort_keys=True) + "\n")
                raw_file.flush()

    _write_csv(output_dir / "results.csv", rows)
    _write_summary(output_dir / "summary_by_model.csv", rows)
    _write_benchmark_table(output_dir / "benchmark.out", rows)
    return rows


def _run_hf_combo(
    *,
    config: MultiLLMConfig,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    num_agents: int,
    branch_factor: int,
    seed: int,
    cache: dict[tuple[str, int, int, int], HFMeasurement],
    backend_cache: dict[tuple[str, int, str, bool], HFKVBackend],
    output_dir: Path,
) -> list[dict[str, Any]]:
    cache_key = (model_id, int(context_tokens), int(output_tokens), int(seed))
    source_metrics_path = ""
    if config.dry_run:
        measurement = _mock_hf_measurement(model_id, context_tokens, output_tokens)
    else:
        try:
            measurement = cache.get(cache_key)
            if measurement is None:
                backend_key = (model_id, int(seed), str(config.hf_device), bool(config.hf_local_files_only))
                backend = backend_cache.get(backend_key)
                if backend is None:
                    backend = HFKVBackend(
                        HFBackendConfig(
                            model_id=model_id,
                            tokenizer_id=model_id,
                            device=config.hf_device,
                            local_files_only=config.hf_local_files_only,
                            seed=seed,
                        )
                    )
                    backend_cache[backend_key] = backend
                prompt = backend.build_prompt(context_tokens)
                measurement = backend.measure(prompt, context_tokens=context_tokens, output_tokens=output_tokens)
                cache[cache_key] = measurement
                source_metrics_path = _write_hf_artifacts(output_dir, model_id, context_tokens, output_tokens, seed, measurement)
            else:
                source_metrics_path = str(
                    _hf_artifact_path(output_dir, model_id, context_tokens, output_tokens, seed) / "metrics.json"
                )
        except Exception as exc:
            return [
                _base_row(
                    model_id=model_id,
                    backend="hf_measured",
                    workload_name=workload,
                    context_tokens=context_tokens,
                    output_tokens=output_tokens,
                    num_agents=num_agents,
                    branch_factor=branch_factor,
                    num_prompts=config.num_prompts,
                    seed=seed,
                    available=False,
                    skipped=True,
                    reason=str(exc),
                )
                for workload in HF_MEASURED_BASELINES
            ]

    measured = measurement.metrics
    kv_bytes = int(measured.get("kv_total_bytes", measurement.kv_state.total_bytes()))
    prefill_sec = float(measured.get("prefill_sec", 0.0))
    decode_sec = float(measured.get("decode_sec", 0.0))
    ttft_sec = float(measured.get("ttft_sec", prefill_sec))
    branch_instances = _branch_instances(num_agents, branch_factor)
    num_prompts = max(1, int(config.num_prompts))
    measured_output_tokens = max(0, int(measured.get("output_tokens", output_tokens) or output_tokens))
    output_total = branch_instances * measured_output_tokens * num_prompts
    remote_branches = max(0, branch_instances - 1)
    full_transfer_time = _transfer_time(kv_bytes, config.bandwidth_bytes_per_sec, config.network_latency_sec)
    shared_prefix_fraction = 0.85
    suffix_fraction = 1.0 - shared_prefix_fraction
    suffix_kv_bytes = int(kv_bytes * suffix_fraction)
    suffix_transfer_time = _transfer_time(suffix_kv_bytes, config.bandwidth_bytes_per_sec, config.network_latency_sec)
    first_token_decode_sec = decode_sec / max(1, measured_output_tokens)

    # Experiment 1 reports the critical-path TTFT for downstream agent work.
    # Text/local-prefix baselines without distributed state orchestration pay
    # prefill at the target. DistServe-style pays full KV transfer. AAFLOW+
    # transfers only the non-reused segment after the shared prefix is already
    # materialized as state.
    text_ttft_sec = prefill_sec + first_token_decode_sec + config.omega_text_sec
    local_prefix_ttft_sec = text_ttft_sec
    distserve_ttft_sec = full_transfer_time + config.resume_overhead_sec + first_token_decode_sec + config.omega_state_sec
    aaflow_ttft_sec = (
        (suffix_transfer_time if remote_branches else 0.0)
        + config.resume_overhead_sec
        + first_token_decode_sec
        + config.omega_state_sec
    )

    dense_prefill_sec = branch_instances * num_prompts * prefill_sec
    dense_decode_sec = branch_instances * num_prompts * decode_sec
    dense_total = dense_prefill_sec + dense_decode_sec + branch_instances * num_prompts * config.omega_text_sec

    local_prefill_sec = num_prompts * prefill_sec
    local_decode_sec = branch_instances * num_prompts * decode_sec
    local_resume_sec = branch_instances * num_prompts * config.resume_overhead_sec
    local_omega_sec = branch_instances * num_prompts * config.omega_state_sec
    local_total = local_prefill_sec + local_decode_sec + local_resume_sec + local_omega_sec

    dist_transfer_sec = num_prompts * full_transfer_time if remote_branches else 0.0
    dist_total = local_prefill_sec + local_decode_sec + local_resume_sec + dist_transfer_sec + local_omega_sec

    aaflow_prefill_sec = prefill_sec + (num_prompts - 1) * prefill_sec * suffix_fraction
    aaflow_transfer_sec = 0.0
    if remote_branches:
        aaflow_transfer_sec = full_transfer_time + (num_prompts - 1) * suffix_transfer_time
    aaflow_resume_sec = num_prompts * config.resume_overhead_sec
    aaflow_decode_sec = num_prompts * decode_sec
    aaflow_omega_sec = num_prompts * config.omega_state_sec
    aaflow_total = aaflow_prefill_sec + aaflow_transfer_sec + aaflow_resume_sec + aaflow_decode_sec + aaflow_omega_sec
    aaflow_reuse = _reuse_ratio(branch_instances, num_prompts, shared_prefix_fraction)

    common = {
        "model_id": model_id,
        "backend": "hf_measured",
        "context_tokens": int(context_tokens),
        "output_tokens": measured_output_tokens,
        "num_agents": int(num_agents),
        "branch_factor": int(branch_factor),
        "num_prompts": num_prompts,
        "seed": int(seed),
        "available": True,
        "skipped": False,
        "reason": "",
        "kv_total_bytes": kv_bytes,
        "kv_peak_bytes": kv_bytes,
        "branch_instances": branch_instances,
        "source_metrics_path": source_metrics_path,
        "hf_prefill_ttft_sec": ttft_sec,
    }
    return [
        {
            **common,
            "run_id": _run_id(),
            "workload_name": "AAFLOW+",
            "ttft_sec": aaflow_ttft_sec,
            "total_latency_sec": aaflow_total,
            "prefill_sec": aaflow_prefill_sec,
            "decode_sec": aaflow_decode_sec,
            "transfer_sec": aaflow_transfer_sec,
            "resume_sec": aaflow_resume_sec,
            "omega_sec": aaflow_omega_sec,
            "throughput_tokens_per_sec": output_total / aaflow_total if aaflow_total > 0 else 0.0,
            "kv_transferred_bytes": remote_branches * (kv_bytes + (num_prompts - 1) * suffix_kv_bytes),
            "kv_reuse_ratio": aaflow_reuse,
            "materialize_count": 1,
            "transfer_count": remote_branches * num_prompts,
            "dense_prefill_sec": dense_prefill_sec,
            "stateful_prefill_sec": aaflow_prefill_sec,
        },
        {
            **common,
            "run_id": _run_id(),
            "workload_name": "dense_prefill",
            "ttft_sec": text_ttft_sec,
            "total_latency_sec": dense_total,
            "prefill_sec": dense_prefill_sec,
            "decode_sec": dense_decode_sec,
            "transfer_sec": 0.0,
            "resume_sec": 0.0,
            "omega_sec": branch_instances * config.omega_text_sec,
            "throughput_tokens_per_sec": output_total / dense_total if dense_total > 0 else 0.0,
            "kv_transferred_bytes": 0,
            "kv_reuse_ratio": 0.0,
            "materialize_count": branch_instances,
            "transfer_count": 0,
            "dense_prefill_sec": dense_prefill_sec,
            "stateful_prefill_sec": aaflow_prefill_sec,
        },
        {
            **common,
            "run_id": _run_id(),
            "workload_name": "aaflow_text",
            "ttft_sec": text_ttft_sec,
            "total_latency_sec": dense_total,
            "prefill_sec": dense_prefill_sec,
            "decode_sec": dense_decode_sec,
            "transfer_sec": 0.0,
            "resume_sec": 0.0,
            "omega_sec": branch_instances * num_prompts * config.omega_text_sec,
            "throughput_tokens_per_sec": output_total / dense_total if dense_total > 0 else 0.0,
            "kv_transferred_bytes": 0,
            "kv_reuse_ratio": 0.0,
            "materialize_count": branch_instances * num_prompts,
            "transfer_count": 0,
            "dense_prefill_sec": dense_prefill_sec,
            "stateful_prefill_sec": aaflow_prefill_sec,
        },
        _local_prefix_row(
            common=common,
            workload_name="vllm_local_prefix",
            ttft_sec=local_prefix_ttft_sec,
            total_latency_sec=local_total,
            prefill_sec=local_prefill_sec,
            decode_sec=local_decode_sec,
            resume_sec=local_resume_sec,
            omega_sec=local_omega_sec,
            output_total=output_total,
            branch_instances=branch_instances,
            num_prompts=num_prompts,
            dense_prefill_sec=dense_prefill_sec,
            stateful_prefill_sec=aaflow_prefill_sec,
        ),
        _local_prefix_row(
            common=common,
            workload_name="sglang_prefix",
            ttft_sec=local_prefix_ttft_sec,
            total_latency_sec=local_total,
            prefill_sec=local_prefill_sec,
            decode_sec=local_decode_sec,
            resume_sec=local_resume_sec,
            omega_sec=local_omega_sec,
            output_total=output_total,
            branch_instances=branch_instances,
            num_prompts=num_prompts,
            dense_prefill_sec=dense_prefill_sec,
            stateful_prefill_sec=aaflow_prefill_sec,
        ),
        {
            **common,
            "run_id": _run_id(),
            "workload_name": "distserve_style",
            "ttft_sec": distserve_ttft_sec,
            "total_latency_sec": dist_total,
            "prefill_sec": local_prefill_sec,
            "decode_sec": local_decode_sec,
            "transfer_sec": dist_transfer_sec,
            "resume_sec": local_resume_sec,
            "omega_sec": local_omega_sec,
            "throughput_tokens_per_sec": output_total / dist_total if dist_total > 0 else 0.0,
            "kv_transferred_bytes": remote_branches * num_prompts * kv_bytes,
            "kv_reuse_ratio": (branch_instances - 1) / branch_instances if branch_instances > 0 else 0.0,
            "materialize_count": num_prompts,
            "transfer_count": remote_branches * num_prompts,
            "dense_prefill_sec": dense_prefill_sec,
            "stateful_prefill_sec": aaflow_prefill_sec,
        },
    ]


def _run_vllm_combo(
    *,
    config: MultiLLMConfig,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    num_agents: int,
    branch_factor: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    row = _base_row(
        model_id=model_id,
        backend="vllm",
        workload_name="vllm_serve",
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        num_agents=num_agents,
        branch_factor=branch_factor,
        num_prompts=config.num_prompts,
        seed=seed,
        available=False,
        skipped=False,
        reason="",
    )
    if config.dry_run:
        ttft = 0.0002 * context_tokens
        tpot = 0.00004
        e2el = ttft + output_tokens * tpot
        return {
            **row,
            "available": True,
            "ttft_sec": ttft,
            "tpot_sec": tpot,
            "itl_sec": tpot,
            "e2el_sec": e2el,
            "total_latency_sec": e2el,
            "throughput_tokens_per_sec": config.num_prompts * output_tokens / e2el if e2el > 0 else 0.0,
            "request_throughput_req_per_sec": config.num_prompts / e2el if e2el > 0 else 0.0,
        }
    if not check_vllm_available():
        row.update(skipped=True, reason="vLLM is not installed or no vLLM CLI is available")
        return row

    combo_dir = output_dir / "vllm_runs" / _safe_name(model_id) / f"ctx{context_tokens}_out{output_tokens}_a{num_agents}_b{branch_factor}_s{seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        server = launch_vllm_server(
            model_id=model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            port=config.vllm_port,
            stdout_path=combo_dir / "vllm_stdout.log",
            stderr_path=combo_dir / "vllm_stderr.log",
        )
        if not wait_for_server(config.vllm_port, timeout_sec=config.vllm_server_timeout_sec):
            raise RuntimeError(f"vLLM server did not become ready on port {config.vllm_port}")
        metrics = run_vllm_bench_serve(
            model_id=model_id,
            input_len=context_tokens,
            output_len=output_tokens,
            num_prompts=config.num_prompts,
            request_rate="inf",
            port=config.vllm_port,
            output_dir=combo_dir,
            timeout_sec=config.vllm_bench_timeout_sec,
        )
        row.update(
            available=bool(metrics.get("available", False)),
            skipped=False,
            reason=str(metrics.get("reason", "")),
            ttft_sec=_float(metrics.get("ttft_sec")),
            total_latency_sec=_float(metrics.get("e2el_sec", metrics.get("total_latency_sec", metrics.get("bench_elapsed_sec")))),
            throughput_tokens_per_sec=_float(metrics.get("throughput_tokens_per_sec")),
            tpot_sec=_float(metrics.get("tpot_sec")),
            itl_sec=_float(metrics.get("itl_sec")),
            e2el_sec=_float(metrics.get("e2el_sec")),
            request_throughput_req_per_sec=_float(metrics.get("request_throughput_req_per_sec")),
            source_metrics_path=str(combo_dir / "metrics.json"),
        )
        return row
    except Exception as exc:
        row.update(available=False, skipped=True, reason=str(exc), source_metrics_path=str(combo_dir / "metrics.json"))
        return row
    finally:
        if server is not None and server.poll() is None:
            try:
                import os
                import signal

                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
                server.wait(timeout=30)
            except Exception:
                try:
                    server.kill()
                except Exception:
                    pass


def _run_sglang_combo(
    *,
    config: MultiLLMConfig,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    num_agents: int,
    branch_factor: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    row = _base_row(
        model_id=model_id,
        backend="sglang",
        workload_name="sglang_serve",
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        num_agents=num_agents,
        branch_factor=branch_factor,
        num_prompts=config.num_prompts,
        seed=seed,
        available=False,
        skipped=False,
        reason="",
    )
    if config.dry_run:
        ttft = 0.00022 * context_tokens
        tpot = 0.000045
        e2el = ttft + output_tokens * tpot
        return {
            **row,
            "available": True,
            "ttft_sec": ttft,
            "tpot_sec": tpot,
            "itl_sec": tpot,
            "e2el_sec": e2el,
            "total_latency_sec": e2el,
            "throughput_tokens_per_sec": config.num_prompts * output_tokens / e2el if e2el > 0 else 0.0,
            "request_throughput_req_per_sec": config.num_prompts / e2el if e2el > 0 else 0.0,
        }
    if not check_sglang_available(config.sglang_python_bin):
        row.update(skipped=True, reason="SGLang is not installed or no SGLang CLI is available")
        return row

    combo_dir = output_dir / "sglang_runs" / _safe_name(model_id) / f"ctx{context_tokens}_out{output_tokens}_a{num_agents}_b{branch_factor}_s{seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        server = launch_sglang_server(
            model_id=model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            port=config.sglang_port,
            extra_args=shlex.split(config.sglang_server_extra_args),
            stdout_path=combo_dir / "sglang_stdout.log",
            stderr_path=combo_dir / "sglang_stderr.log",
            python_bin=config.sglang_python_bin,
        )
        if not wait_for_sglang_server(config.sglang_port, timeout_sec=config.sglang_server_timeout_sec):
            raise RuntimeError(f"SGLang server did not become ready on port {config.sglang_port}")
        metrics = run_sglang_bench_serve(
            model_id=model_id,
            input_len=context_tokens,
            output_len=output_tokens,
            num_prompts=config.num_prompts,
            request_rate="inf",
            port=config.sglang_port,
            output_dir=combo_dir,
            timeout_sec=config.sglang_bench_timeout_sec,
            python_bin=config.sglang_python_bin,
        )
        row.update(
            available=bool(metrics.get("available", False)),
            skipped=False,
            reason=str(metrics.get("reason", "")),
            ttft_sec=_float(metrics.get("ttft_sec")),
            total_latency_sec=_float(metrics.get("e2el_sec", metrics.get("total_latency_sec", metrics.get("bench_elapsed_sec")))),
            throughput_tokens_per_sec=_float(metrics.get("throughput_tokens_per_sec")),
            tpot_sec=_float(metrics.get("tpot_sec")),
            itl_sec=_float(metrics.get("itl_sec")),
            e2el_sec=_float(metrics.get("e2el_sec")),
            request_throughput_req_per_sec=_float(metrics.get("request_throughput_req_per_sec")),
            source_metrics_path=str(combo_dir / "metrics.json"),
        )
        return row
    except Exception as exc:
        row.update(available=False, skipped=True, reason=str(exc), source_metrics_path=str(combo_dir / "metrics.json"))
        return row
    finally:
        if server is not None:
            terminate_sglang_process_tree(server)


def _write_hf_artifacts(
    output_dir: Path,
    model_id: str,
    context_tokens: int,
    output_tokens: int,
    seed: int,
    measurement: HFMeasurement,
) -> str:
    path = _hf_artifact_path(output_dir, model_id, context_tokens, output_tokens, seed)
    path.mkdir(parents=True, exist_ok=True)
    metrics_path = path / "metrics.json"
    metrics_path.write_text(json.dumps(measurement.metrics, indent=2, sort_keys=True), encoding="utf-8")
    (path / "kv_metadata.json").write_text(
        json.dumps(measurement.kv_state.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(metrics_path)


def _hf_artifact_path(output_dir: Path, model_id: str, context_tokens: int, output_tokens: int, seed: int) -> Path:
    return output_dir / "hf_measurements" / _safe_name(model_id) / f"ctx{context_tokens}_out{output_tokens}_s{seed}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(RESULT_FIELDS))
    fields = [*RESULT_FIELDS, *extra_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["backend"], row["workload_name"]), []).append(row)
    summary = []
    for (model_id, backend, workload_name), group_rows in sorted(groups.items()):
        available_rows = [row for row in group_rows if row.get("available")]
        summary.append(
            {
                "model_id": model_id,
                "backend": backend,
                "workload_name": workload_name,
                "rows": len(group_rows),
                "available_rows": len(available_rows),
                "skipped_rows": sum(1 for row in group_rows if row.get("skipped")),
                "mean_ttft_sec": _mean(row.get("ttft_sec") for row in available_rows),
                "mean_total_latency_sec": _mean(row.get("total_latency_sec") for row in available_rows),
                "mean_throughput_tokens_per_sec": _mean(row.get("throughput_tokens_per_sec") for row in available_rows),
                "mean_kv_total_bytes": _mean(row.get("kv_total_bytes") for row in available_rows),
            }
        )
    _write_csv_with_fields(
        path,
        summary,
        [
            "model_id",
            "backend",
            "workload_name",
            "rows",
            "available_rows",
            "skipped_rows",
            "mean_ttft_sec",
            "mean_total_latency_sec",
            "mean_throughput_tokens_per_sec",
            "mean_kv_total_bytes",
        ],
    )


def _write_benchmark_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact table summary similar to AAFLOW benchmark.out files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model",
        "Backend",
        "Ctx",
        "Out",
        "Prompts",
        "TTFT(s)",
        "TPOT(s)",
        "ITL(s)",
        "Tok/s",
        "Req/s",
        "Total(s)",
        "Status",
    ]
    table_rows = []
    for row in rows:
        status = "ok" if row.get("available") is True and row.get("skipped") is not True else _short_reason(row.get("reason", "skipped"))
        table_rows.append(
            [
                _short_model(str(row.get("model_id", ""))),
                str(row.get("backend", "")),
                str(row.get("context_tokens", "")),
                str(row.get("output_tokens", "")),
                str(row.get("num_prompts", "")),
                _fmt_float(row.get("ttft_sec")),
                _fmt_float(row.get("tpot_sec")),
                _fmt_float(row.get("itl_sec")),
                _fmt_float(row.get("throughput_tokens_per_sec")),
                _fmt_float(row.get("request_throughput_req_per_sec")),
                _fmt_float(row.get("total_latency_sec")),
                status,
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(table_row[idx]) for table_row in table_rows))
        for idx in range(len(headers))
    ]
    lines = [
        " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(table_row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for table_row in table_rows
    )
    if not table_rows:
        lines.append("No benchmark rows were produced.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _short_model(model_id: str) -> str:
    if "/" in model_id:
        return model_id.rsplit("/", 1)[-1]
    return model_id


def _short_reason(reason: Any, limit: int = 42) -> str:
    text = str(reason or "skipped")
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _fmt_float(value: Any) -> str:
    try:
        if value in {"", None}:
            return ""
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    return f"{number:.3f}"


def _write_csv_with_fields(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _base_row(
    *,
    model_id: str,
    backend: str,
    workload_name: str,
    context_tokens: int,
    output_tokens: int,
    num_agents: int,
    branch_factor: int,
    num_prompts: int,
    seed: int,
    available: bool,
    skipped: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_id": _run_id(),
        "seed": int(seed),
        "model_id": model_id,
        "backend": backend,
        "workload_name": workload_name,
        "context_tokens": int(context_tokens),
        "output_tokens": int(output_tokens),
        "num_agents": int(num_agents),
        "branch_factor": int(branch_factor),
        "num_prompts": int(num_prompts),
        "available": bool(available),
        "skipped": bool(skipped),
        "reason": reason,
        "branch_instances": _branch_instances(num_agents, branch_factor),
    }


def _local_prefix_row(
    *,
    common: dict[str, Any],
    workload_name: str,
    ttft_sec: float,
    total_latency_sec: float,
    prefill_sec: float,
    decode_sec: float,
    resume_sec: float,
    omega_sec: float,
    output_total: int,
    branch_instances: int,
    num_prompts: int,
    dense_prefill_sec: float,
    stateful_prefill_sec: float,
) -> dict[str, Any]:
    return {
        **common,
        "run_id": _run_id(),
        "workload_name": workload_name,
        "ttft_sec": ttft_sec,
        "total_latency_sec": total_latency_sec,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "transfer_sec": 0.0,
        "resume_sec": resume_sec,
        "omega_sec": omega_sec,
        "throughput_tokens_per_sec": output_total / total_latency_sec if total_latency_sec > 0 else 0.0,
        "kv_transferred_bytes": 0,
        "kv_reuse_ratio": (branch_instances - 1) / branch_instances if branch_instances > 0 else 0.0,
        "materialize_count": num_prompts,
        "transfer_count": 0,
        "dense_prefill_sec": dense_prefill_sec,
        "stateful_prefill_sec": stateful_prefill_sec,
    }


def _skipped_context_rows(
    *,
    model_id: str,
    backend: str,
    context_tokens: int,
    output_tokens: int,
    num_agents: int,
    branch_factor: int,
    num_prompts: int,
    seed: int,
    reason: str,
) -> list[dict[str, Any]]:
    if backend == "hf":
        workloads = HF_MEASURED_BASELINES
    elif backend == "sglang":
        workloads = ("sglang_serve",)
    else:
        workloads = ("vllm_serve",)
    return [
        _base_row(
            model_id=model_id,
            backend=backend,
            workload_name=workload,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            num_agents=num_agents,
            branch_factor=branch_factor,
            num_prompts=num_prompts,
            seed=seed,
            available=False,
            skipped=True,
            reason=reason,
        )
        for workload in workloads
    ]


def _invalid_context_reason(model_id: str, context_tokens: int, output_tokens: int) -> str:
    max_context = _model_max_context(model_id)
    if max_context <= 0:
        return ""
    requested_total = int(context_tokens) + int(output_tokens)
    if requested_total > max_context:
        return (
            f"requested context+output tokens ({requested_total}) exceeds max context "
            f"{max_context} for {model_id}"
        )
    return ""


def _model_max_context(model_id: str) -> int:
    spec = get_model_spec(model_id)
    if spec is not None:
        return int(spec.max_context)
    lowered = model_id.lower()
    if "gpt2" in lowered:
        return 1024
    if "mistral" in lowered or "qwen2.5" in lowered:
        return 32768
    if "llama-3" in lowered or "meta-llama-3" in lowered:
        return 8192
    if "llama-2" in lowered:
        return 4096
    return 0


def _mock_hf_measurement(model_id: str, context_tokens: int, output_tokens: int) -> HFMeasurement:
    from .operators import KVMaterializeOperator

    state = KVMaterializeOperator().execute(
        token_count=context_tokens,
        model_id=model_id,
        tokenizer_id=model_id,
    )
    prefill_sec = 0.0002 * context_tokens
    decode_sec = 0.00005 * output_tokens
    return HFMeasurement(
        kv_state=state,
        generated_text="",
        generated_token_ids=[],
        metrics={
            "ttft_sec": prefill_sec,
            "total_latency_sec": prefill_sec + decode_sec,
            "prefill_sec": prefill_sec,
            "decode_sec": decode_sec,
            "kv_total_bytes": state.total_bytes(),
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
        },
    )


def _release_torch_memory() -> None:
    """Best-effort cleanup when a real sweep moves from one HF model to another."""

    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in RESULT_FIELDS:
        normalized.setdefault(field, "")
    return normalized


def _transfer_time(kv_bytes: int, bandwidth_bytes_per_sec: float, latency_sec: float) -> float:
    bandwidth = max(float(bandwidth_bytes_per_sec), 1.0)
    return int(kv_bytes) / bandwidth + float(latency_sec)


def _branch_instances(num_agents: int, branch_factor: int) -> int:
    return max(1, int(num_agents)) * max(1, int(branch_factor))


def _reuse_ratio(branch_instances: int, num_prompts: int, shared_prefix_fraction: float) -> float:
    branch_reuse = (branch_instances - 1) / branch_instances if branch_instances > 0 else 0.0
    prompt_reuse = (num_prompts - 1) / num_prompts if num_prompts > 0 else 0.0
    return min(1.0, max(branch_reuse, shared_prefix_fraction * prompt_reuse))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _run_id() -> str:
    return f"multi_{uuid.uuid4().hex[:12]}"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> float:
    parsed = [_float(value, default=float("nan")) for value in values]
    parsed = [value for value in parsed if value == value]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _parse_csv_str(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_grid(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_str(value)]


def _parse_float(value: str) -> float:
    return float(value)


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args(argv_list)
    config = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(description="Run multi-LLM Stateful Agentic Algebra benchmark matrix")
    parser.add_argument("--config", default="", help="YAML/JSON config file")
    parser.add_argument("--models", default=csv_default(config_value(config, "models")), help="Comma-separated model ids")
    parser.add_argument("--backend", default=csv_default(config_value(config, "backend", "backends")), help="Comma-separated backends: hf,vllm,sglang")
    parser.add_argument("--context-grid", default=csv_default(config_value(config, "context_grid", "context-grid")))
    parser.add_argument("--output-grid", default=csv_default(config_value(config, "output_grid", "output-grid")))
    parser.add_argument("--agent-grid", default=csv_default(config_value(config, "agent_grid", "agent-grid")))
    parser.add_argument("--branch-grid", default=csv_default(config_value(config, "branch_grid", "branch-grid")))
    parser.add_argument("--num-prompts", type=int, default=int(config_value(config, "num_prompts", "num-prompts", default=16)))
    parser.add_argument("--output-dir", default=str(config_value(config, "output_dir", "output-dir", default="runs/stateful/multi_llm")))
    parser.add_argument("--seeds", default=csv_default(config_value(config, "seeds", default="0")))
    parser.add_argument("--bandwidth-bytes-per-sec", type=_parse_float, default=float(config_value(config, "bandwidth_bytes_per_sec", "bandwidth-bytes-per-sec", default=25_000_000_000.0)))
    parser.add_argument("--network-latency-sec", type=_parse_float, default=float(config_value(config, "network_latency_sec", "network-latency-sec", default=0.00005)))
    parser.add_argument("--resume-overhead-sec", type=_parse_float, default=float(config_value(config, "resume_overhead_sec", "resume-overhead-sec", default=0.0001)))
    parser.add_argument("--omega-state-sec", type=_parse_float, default=float(config_value(config, "omega_state_sec", "omega-state-sec", default=0.00005)))
    parser.add_argument("--omega-text-sec", type=_parse_float, default=float(config_value(config, "omega_text_sec", "omega-text-sec", default=0.00005)))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(config_value(config, "tensor_parallel_size", "tensor-parallel-size", default=1)))
    parser.add_argument("--vllm-port", type=int, default=int(config_value(config, "vllm_port", "vllm-port", default=8000)))
    parser.add_argument("--vllm-server-timeout-sec", type=float, default=float(config_value(config, "vllm_server_timeout_sec", "vllm-server-timeout-sec", default=900.0)))
    parser.add_argument("--vllm-bench-timeout-sec", type=float, default=float(config_value(config, "vllm_bench_timeout_sec", "vllm-bench-timeout-sec", default=1800.0)))
    parser.add_argument("--sglang-port", type=int, default=int(config_value(config, "sglang_port", "sglang-port", default=30000)))
    parser.add_argument("--sglang-server-timeout-sec", type=float, default=float(config_value(config, "sglang_server_timeout_sec", "sglang-server-timeout-sec", default=900.0)))
    parser.add_argument("--sglang-bench-timeout-sec", type=float, default=float(config_value(config, "sglang_bench_timeout_sec", "sglang-bench-timeout-sec", default=1800.0)))
    parser.add_argument("--sglang-python-bin", default=str(config_value(config, "sglang_python_bin", "sglang-python-bin", default="")))
    parser.add_argument("--sglang-server-extra-args", default=str(config_value(config, "sglang_server_extra_args", "sglang-server-extra-args", default="")))
    parser.add_argument("--hf-device", default=str(config_value(config, "hf_device", "hf-device", default="auto")), choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf-local-files-only", action="store_true", default=bool_default(config_value(config, "hf_local_files_only", "hf-local-files-only", default=False)))
    parser.add_argument("--skip-invalid-context", action=argparse.BooleanOptionalAction, default=bool_default(config_value(config, "skip_invalid_context", "skip-invalid-context", default=True)))
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=bool_default(config_value(config, "progress", default=True)))
    parser.add_argument("--dry-run", action="store_true", default=bool_default(config_value(config, "dry_run", "dry-run", default=False)), help="Use synthetic measurements without loading models")
    args = parser.parse_args(argv_list)
    if not args.models:
        parser.error("--models is required unless provided by --config")
    if not args.backend:
        parser.error("--backend is required unless provided by --config")
    for field in ("context_grid", "output_grid", "agent_grid", "branch_grid"):
        if not getattr(args, field):
            parser.error(f"--{field.replace('_', '-')} is required unless provided by --config")
    return args


def config_from_args(args: argparse.Namespace) -> MultiLLMConfig:
    return MultiLLMConfig(
        models=_parse_csv_str(args.models),
        backends=_parse_csv_str(args.backend),
        context_grid=_parse_int_grid(args.context_grid),
        output_grid=_parse_int_grid(args.output_grid),
        agent_grid=_parse_int_grid(args.agent_grid),
        branch_grid=_parse_int_grid(args.branch_grid),
        num_prompts=max(1, int(args.num_prompts)),
        output_dir=args.output_dir,
        seeds=_parse_int_grid(args.seeds),
        bandwidth_bytes_per_sec=float(args.bandwidth_bytes_per_sec),
        network_latency_sec=float(args.network_latency_sec),
        resume_overhead_sec=float(args.resume_overhead_sec),
        omega_state_sec=float(args.omega_state_sec),
        omega_text_sec=float(args.omega_text_sec),
        tensor_parallel_size=max(1, int(args.tensor_parallel_size)),
        vllm_port=int(args.vllm_port),
        vllm_server_timeout_sec=float(args.vllm_server_timeout_sec),
        vllm_bench_timeout_sec=float(args.vllm_bench_timeout_sec),
        sglang_port=int(args.sglang_port),
        sglang_server_timeout_sec=float(args.sglang_server_timeout_sec),
        sglang_bench_timeout_sec=float(args.sglang_bench_timeout_sec),
        sglang_python_bin=str(args.sglang_python_bin),
        sglang_server_extra_args=str(args.sglang_server_extra_args),
        hf_device=args.hf_device,
        hf_local_files_only=bool(args.hf_local_files_only),
        skip_invalid_context=bool(args.skip_invalid_context),
        progress=bool(args.progress),
        dry_run=bool(args.dry_run),
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    started = time.perf_counter()
    config = config_from_args(parse_args(argv))
    rows = run_matrix(config)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output_dir": config.output_dir,
                "rows": len(rows),
                "elapsed_sec": elapsed,
                "results_csv": str(Path(config.output_dir) / "results.csv"),
                "summary_csv": str(Path(config.output_dir) / "summary_by_model.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
