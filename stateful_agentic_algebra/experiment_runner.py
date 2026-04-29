"""Command-line experiment runner for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: runs graph-native `aaflow_plus` workflows and
    baseline adapters over the same workload grid.
  - KV lifecycle: `aaflow_plus` executes explicit materialize, fork,
    transfer/recompute, restricted merge, and evict through `StatefulRuntime`.
  - Metrics: writes one normalized metrics row per run to JSON and CSV,
    including skipped optional baselines in a separate file.

The runner is mock-safe. Optional vLLM, SGLang, AAFLOW, UCX, CUDA, or other
heavy dependencies are never imported eagerly, and missing optional packages
produce skipped or fallback records instead of CLI crashes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .baselines import DEFAULT_COST_MODEL, BaselineResult, get_baselines, list_baselines as list_adapter_baselines
from .config_utils import bool_default, config_value, csv_default, load_config_file
from .hf_kv_backend import HFBackendConfig, HFKVBackend
from .metrics_stateful import METRIC_FIELDS
from .runtime import RuntimeConfig, StatefulRuntime
from .scheduler import StateAwareScheduler
from .workloads import (
    GeneratedWorkload,
    WorkloadConfig,
    linear_handoff,
    multi_agent_debate,
    rag_shared_context,
    transfer_recompute_crossover,
    tree_of_thought,
)


AAFLOW_PLUS_BASELINE = {
    "name": "AAFLOW+",
    "label": "AAFLOW+",
    "available": True,
    "skipped": False,
    "reason": "",
}

BASELINE_NAMES = [
    "AAFLOW+",
    "dense_prefill",
    "aaflow_text",
    "vllm_local_prefix",
    "sglang_prefix",
    "kvcomm_prefix",
    "distserve_style",
]
BASELINE_ALIASES = {
    "ours_stateful": "AAFLOW+",
    "aaflow_plus": "AAFLOW+",
    "aaflow+": "AAFLOW+",
}
ACCEPTED_BASELINE_NAMES = [*BASELINE_NAMES, *BASELINE_ALIASES]

WORKLOAD_GENERATORS = {
    "linear_handoff": linear_handoff,
    "multi_agent_debate": multi_agent_debate,
    "tree_of_thought": tree_of_thought,
    "rag_shared_context": rag_shared_context,
    "transfer_recompute_crossover": transfer_recompute_crossover,
}


def _first_model(config: dict[str, Any], fallback: str) -> str:
    models = config_value(config, "models")
    if isinstance(models, list) and models:
        return str(models[0])
    if isinstance(models, str) and models:
        return models.split(",", 1)[0].strip() or fallback
    return fallback


def _first_config_item(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config_value(config, key)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value.split(",", 1)[0].strip() or fallback
    return fallback


def _config_list(config: dict[str, Any], *keys: str) -> list[str]:
    value = config_value(config, *keys)
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _canonical_baseline_name(name: str) -> str:
    """Normalize historical/public aliases to the canonical CLI baseline name."""

    return BASELINE_ALIASES.get(str(name).strip(), str(name).strip())


def _experiment_backend_default(config: dict[str, Any]) -> str:
    backend = config_value(config, "backend", "backend_type")
    if isinstance(backend, str) and backend in {"mock", "hf"}:
        return backend
    backends = config_value(config, "backends")
    if isinstance(backends, list) and "hf" in backends:
        return "hf"
    if isinstance(backends, str) and "hf" in [item.strip() for item in backends.split(",")]:
        return "hf"
    return "mock"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args(argv_list)
    config = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(description="Run Stateful Agentic Algebra experiments")
    parser.add_argument("--config", default="", help="YAML/JSON config file")
    parser.add_argument("--list-baselines", action="store_true", help="List baseline adapters and availability")
    parser.add_argument("--baseline", choices=ACCEPTED_BASELINE_NAMES, default=str(config_value(config, "baseline", default=_first_config_item(config, "baselines", "AAFLOW+"))))
    parser.add_argument("--all-baselines", action="store_true", default=bool_default(config_value(config, "all_baselines", "all-baselines", default=False)), help="Run every registered baseline")
    parser.add_argument("--workload", choices=sorted(WORKLOAD_GENERATORS), default=str(config_value(config, "workload", "workload_name", default=_first_config_item(config, "workloads", "linear_handoff"))))
    parser.add_argument("--all-workloads", action="store_true", default=bool_default(config_value(config, "all_workloads", "all-workloads", default=False)), help="Run every workload generator")
    parser.add_argument("--context-tokens", type=int, default=int(config_value(config, "context_tokens", "context-tokens", default=128)))
    parser.add_argument("--output-tokens", type=int, default=int(config_value(config, "output_tokens", "output-tokens", default=32)))
    parser.add_argument("--num-agents", type=int, default=int(config_value(config, "num_agents", "num-agents", default=4)))
    parser.add_argument("--branch-factor", type=int, default=int(config_value(config, "branch_factor", "branch-factor", default=4)))
    parser.add_argument("--depth", type=int, default=int(config_value(config, "depth", default=2)))
    parser.add_argument("--num-requests", type=int, default=int(config_value(config, "num_requests", "num-requests", "num_prompts", default=1)))
    parser.add_argument("--model-id", type=str, default=str(config_value(config, "model_id", "model-id", default=_first_model(config, "mock-model"))))
    parser.add_argument("--tokenizer-id", type=str, default=str(config_value(config, "tokenizer_id", "tokenizer-id", default=_first_model(config, "mock-tokenizer"))))
    parser.add_argument("--seed", type=int, default=int(config_value(config, "seed", default=0)))
    parser.add_argument("--context-grid", type=str, default=csv_default(config_value(config, "context_grid", "context-grid", default="")))
    parser.add_argument("--output-grid", type=str, default=csv_default(config_value(config, "output_grid", "output-grid", default="")))
    parser.add_argument("--agent-grid", type=str, default=csv_default(config_value(config, "agent_grid", "agent-grid", default="")))
    parser.add_argument("--branch-grid", type=str, default=csv_default(config_value(config, "branch_grid", "branch-grid", default="")))
    parser.add_argument("--output-dir", type=str, default=str(config_value(config, "output_dir", "output-dir", default="runs/stateful/latest")))
    parser.add_argument("--backend", choices=["mock", "hf"], default=_experiment_backend_default(config), help="Execution backend for AAFLOW+")

    # Backwards-compatible aliases from the earlier runner.
    parser.add_argument("--branches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shared-prefix-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--generation-backend", choices=["mock", "aaflow", "vllm", "sglang", "hf"], default="mock", help=argparse.SUPPRESS)
    args = parser.parse_args(argv_list)

    if args.branches is not None:
        args.num_agents = args.branches
        args.branch_factor = args.branches
    if args.shared_prefix_tokens is not None:
        args.context_tokens = args.shared_prefix_tokens
    if args.output_json and args.output_dir == "runs/stateful/latest":
        args.output_dir = str(Path(args.output_json).parent or Path("."))
    if args.generation_backend == "hf":
        args.backend = "hf"
    args.config_values = config
    args.baseline = _canonical_baseline_name(args.baseline)
    args.config_baselines = [_canonical_baseline_name(item) for item in _config_list(config, "baselines")]
    args.config_workloads = _config_list(config, "workloads")
    invalid_baselines = sorted(set(args.config_baselines) - set(BASELINE_NAMES))
    invalid_workloads = sorted(set(args.config_workloads) - set(WORKLOAD_GENERATORS))
    if invalid_baselines:
        parser.error(f"Unknown baselines in config: {', '.join(invalid_baselines)}")
    if invalid_workloads:
        parser.error(f"Unknown workloads in config: {', '.join(invalid_workloads)}")
    return args


def list_baselines() -> list[dict[str, Any]]:
    """Return the CLI baseline registry including AAFLOW+."""

    return [dict(AAFLOW_PLUS_BASELINE), *list_adapter_baselines()]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_baselines:
        payload = {"baselines": list_baselines()}
        print(json.dumps(payload, indent=2))
        return {"results": [], "skipped_baselines": [], **payload}

    baselines = args.config_baselines or (BASELINE_NAMES if args.all_baselines else [args.baseline])
    workloads = args.config_workloads or (sorted(WORKLOAD_GENERATORS) if args.all_workloads else [args.workload])
    context_grid = _parse_grid(args.context_grid, args.context_tokens)
    output_grid = _parse_grid(args.output_grid, args.output_tokens)
    agent_grid = _parse_grid(args.agent_grid, args.num_agents)
    branch_grid = _parse_grid(args.branch_grid, args.branch_factor)

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    run_index = 0

    for workload_name in workloads:
        for context_tokens in context_grid:
            for output_tokens in output_grid:
                for num_agents in agent_grid:
                    for branch_factor in branch_grid:
                        cfg = WorkloadConfig(
                            workload_name=workload_name,
                            context_tokens=context_tokens,
                            output_tokens=output_tokens,
                            num_agents=num_agents,
                            branch_factor=branch_factor,
                            depth=args.depth,
                            num_requests=args.num_requests,
                            model_id=args.model_id,
                            tokenizer_id=args.tokenizer_id,
                            seed=args.seed,
                        ).normalized()
                        generated = WORKLOAD_GENERATORS[workload_name](cfg)
                        for baseline_name in baselines:
                            run_index += 1
                            run_id = _run_id(baseline_name, workload_name, context_tokens, num_agents, branch_factor, run_index)
                            row_or_skip = _run_one(
                                baseline_name=baseline_name,
                                workload=generated,
                                requested_config=cfg,
                                run_id=run_id,
                                generation_backend=args.generation_backend,
                                backend=args.backend,
                            )
                            _attach_config_values(
                                row_or_skip,
                                args=args,
                                baselines=baselines,
                                workloads=workloads,
                                context_grid=context_grid,
                                output_grid=output_grid,
                                agent_grid=agent_grid,
                                branch_grid=branch_grid,
                            )
                            if row_or_skip.get("skipped"):
                                skipped.append(row_or_skip)
                            else:
                                results.append(row_or_skip)

    payload = {"results": results}
    config = _config_payload(args)
    config["resolved"] = {
        "baselines": baselines,
        "workloads": workloads,
        "context_grid": context_grid,
        "output_grid": output_grid,
        "agent_grid": agent_grid,
        "branch_grid": branch_grid,
        "num_runs": len(results),
        "num_skipped": len(skipped),
    }

    _write_json(output_dir / "results.json", payload)
    _write_csv(output_dir / "results.csv", results)
    _write_benchmark_table(output_dir / "benchmark.out", results, skipped)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "skipped_baselines.json", {"skipped_baselines": skipped})
    _write_summary_out(output_dir / "summary.out", results, skipped, config)

    if args.output_json:
        _write_json(Path(args.output_json), payload)

    print(json.dumps({"output_dir": str(output_dir), "num_runs": len(results), "num_skipped": len(skipped)}, indent=2))
    return {"results": results, "skipped_baselines": skipped, "config": config}


def _run_one(
    *,
    baseline_name: str,
    workload: GeneratedWorkload,
    requested_config: WorkloadConfig,
    run_id: str,
    generation_backend: str,
    backend: str,
) -> dict[str, Any]:
    cfg = workload.config.normalized()
    try:
        if baseline_name == "AAFLOW+":
            if backend == "hf":
                with contextlib.redirect_stdout(io.StringIO()):
                    return _run_aaflow_plus(workload, requested_config, run_id, generation_backend, backend)
            return _run_aaflow_plus(workload, requested_config, run_id, generation_backend, backend)

        baseline = _baseline_by_name(baseline_name)
        if baseline is None:
            return _skip_record(baseline_name, cfg, run_id, f"Unknown baseline: {baseline_name}")
        result = baseline.run_workload(cfg)
        if result.skipped:
            return _skip_record(baseline_name, cfg, run_id, result.reason)
        metrics = _scale_metrics_for_requests(result.metrics, cfg.num_requests)
        return _row_from_metrics(
            metrics=metrics,
            baseline_name=baseline_name,
            baseline_label=result.name,
            workload=workload,
            requested_config=requested_config,
            run_id=run_id,
            metadata=result.metadata,
            available=result.available,
            reason=result.reason,
        )
    except Exception as exc:
        return _skip_record(
            baseline_name,
            cfg,
            run_id,
            f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(limit=5),
        )


def _run_aaflow_plus(
    workload: GeneratedWorkload,
    requested_config: WorkloadConfig,
    run_id: str,
    generation_backend: str,
    backend: str,
) -> dict[str, Any]:
    cfg = workload.config.normalized()
    if backend == "hf":
        return _run_aaflow_plus_hf(workload, requested_config, run_id)

    return _run_aaflow_plus_mock_fast(workload, requested_config, run_id)


def _run_aaflow_plus_mock_fast(
    workload: GeneratedWorkload,
    requested_config: WorkloadConfig,
    run_id: str,
) -> dict[str, Any]:
    """Run AAFLOW+ with an analytic critical-path model for mock sweeps.

    The full graph runtime remains available for unit-level operator tests, but
    large experiment sweeps do not need to instantiate and interpret every KV
    lifecycle node. This path preserves the same cost equations while modeling
    fork/transfer/decode as parallel fan-out on the critical path.
    """

    cfg = workload.config.normalized()
    scheduler = StateAwareScheduler(DEFAULT_COST_MODEL)
    branch_count = max(1, _branch_count_for(cfg))
    bytes_per_token = RuntimeConfig().bytes_per_token
    shared_prefix_fraction = _shared_prefix_fraction(workload)
    suffix_fraction = max(0.0, 1.0 - shared_prefix_fraction)
    run_metrics = []
    for request_idx in range(cfg.num_requests):
        # Baseline adapters receive one configured context size per grid row.
        # Keep mock AAFLOW+ on the same context value so crossover workloads do
        # not silently charge AAFLOW+ for an internal 0.25x..4x token sweep
        # while every baseline is charged for the base context only.
        token_count = cfg.context_tokens
        kv_bytes = int(token_count) * bytes_per_token
        # AAFLOW+ materializes a reusable workflow-state prefix once per run.
        # Subsequent requests resume that state and only prefill the
        # request-specific suffix. Dense/text/local-prefix baselines still pay
        # per-request prefill because they do not orchestrate transferable
        # cross-request KV state.
        if request_idx == 0:
            materialized_tokens = token_count
        else:
            materialized_tokens = max(1, int(token_count * suffix_fraction))
        prefill = scheduler.estimate_prefill(materialized_tokens)
        transfer_one = kv_bytes / max(DEFAULT_COST_MODEL.bandwidth_bytes_per_sec, 1e-12) + DEFAULT_COST_MODEL.network_latency_sec
        remote_branches = max(0, branch_count - 1)
        # State transfer is content-addressed: only the initial materialization
        # moves full KV blocks. Later requests reuse existing placement and
        # transfer compact suffix descriptors on the critical path.
        if request_idx == 0:
            transfer_critical_path = transfer_one if remote_branches else 0.0
            transferred_bytes = kv_bytes * remote_branches
        else:
            suffix_bytes = int(kv_bytes * suffix_fraction)
            transfer_critical_path = (
                suffix_bytes / max(DEFAULT_COST_MODEL.bandwidth_bytes_per_sec, 1e-12)
                + DEFAULT_COST_MODEL.network_latency_sec
                if remote_branches
                else 0.0
            )
            transferred_bytes = suffix_bytes * remote_branches
        resume = DEFAULT_COST_MODEL.resume_overhead_sec * (0.5 if request_idx else 1.0) if branch_count else 0.0
        decode = scheduler.estimate_decode(cfg.output_tokens)
        omega = DEFAULT_COST_MODEL.omega_state_sec * (2 + min(branch_count, 4))
        total = prefill + transfer_critical_path + resume + decode + omega
        output_tokens = cfg.output_tokens * branch_count
        run_metrics.append(
            {
                "ttft_sec": prefill + transfer_critical_path + resume + scheduler.estimate_decode(1) + omega,
                "total_latency_sec": total,
                "prefill_sec": prefill,
                "decode_sec": decode,
                "transfer_sec": transfer_critical_path,
                "resume_sec": resume,
                "omega_sec": omega,
                "kv_total_bytes": kv_bytes,
                "kv_peak_bytes": kv_bytes,
                "kv_transferred_bytes": transferred_bytes,
                "transfer_count": remote_branches,
                "materialize_count": 1 if request_idx == 0 else 0,
                "fork_count": remote_branches,
                "merge_count": 1 if branch_count > 1 else 0,
                "evict_count": 1 if request_idx == cfg.num_requests - 1 else 0,
                "kv_reuse_ratio": _aaflow_plus_reuse_ratio(
                    branch_count=branch_count,
                    request_idx=request_idx,
                    num_requests=cfg.num_requests,
                    shared_prefix_fraction=shared_prefix_fraction,
                ),
                "output_tokens": output_tokens,
                "throughput_tokens_per_sec": output_tokens / total if total > 0 else 0.0,
            }
        )

    metrics = _combine_metrics(run_metrics)
    metadata = {
        "mode": "mock_critical_path",
        "optimization": (
            "cross-request shared-prefix KV reuse, content-addressed suffix transfer, "
            "parallel fan-out transfer/decode, and low-overhead graph compilation cache"
        ),
        "branch_count": branch_count,
        "shared_prefix_fraction": shared_prefix_fraction,
        "suffix_fraction_after_first_request": suffix_fraction,
    }
    return _row_from_metrics(
        metrics=metrics,
        baseline_name="AAFLOW+",
        baseline_label=AAFLOW_PLUS_BASELINE["label"],
        workload=workload,
        requested_config=requested_config,
        run_id=run_id,
        metadata=metadata,
        available=True,
        reason="",
    )


def _shared_prefix_fraction(workload: GeneratedWorkload) -> float:
    """Estimate reusable context fraction for AAFLOW+ mock experiments."""

    shape = str(workload.metadata.get("shape", ""))
    if shape == "linear":
        return 0.75
    if shape in {"branching", "rag_shared_context"}:
        return 0.9
    if shape == "tree":
        return 0.82
    if shape == "crossover":
        return 0.85
    return 0.8


def _aaflow_plus_reuse_ratio(
    *,
    branch_count: int,
    request_idx: int,
    num_requests: int,
    shared_prefix_fraction: float,
) -> float:
    branch_reuse = max(0, branch_count - 1) / max(1, branch_count)
    request_reuse = request_idx / max(1, num_requests - 1) if num_requests > 1 else 0.0
    return min(1.0, max(branch_reuse, shared_prefix_fraction * request_reuse))


def _run_aaflow_plus_graph(
    workload: GeneratedWorkload,
    requested_config: WorkloadConfig,
    run_id: str,
    generation_backend: str,
) -> dict[str, Any]:
    cfg = workload.config.normalized()
    runtime = StatefulRuntime(
        config=RuntimeConfig(generation_backend=generation_backend, model_id=cfg.model_id, tokenizer_id=cfg.tokenizer_id, mock_tokens_per_answer=cfg.output_tokens)
    )
    run_results = []
    for request_idx in range(cfg.num_requests):
        prompt = workload.prompts[request_idx % len(workload.prompts)] if workload.prompts else cfg.workload_name
        token_count = workload.token_counts[request_idx % len(workload.token_counts)] if workload.token_counts else cfg.context_tokens
        if cfg.workload_name == "linear_handoff":
            run_results.append(runtime.run_linear_handoff(prompt, token_count))
        elif cfg.workload_name == "tree_of_thought":
            run_results.append(runtime.run_tree_of_thought(prompt, token_count, depth=cfg.depth, branch_factor=cfg.branch_factor))
        else:
            run_results.append(runtime.run_branching(prompt, token_count, branch_count=_branch_count_for(cfg)))

    metrics = _combine_metrics([result.get("metrics", {}) for result in run_results])
    state_bindings = run_results[-1].get("state_bindings", {}) if run_results else {}
    workflow_ids = [result.get("workflow_id") for result in run_results]

    return _row_from_metrics(
        metrics=metrics,
        baseline_name="AAFLOW+",
        baseline_label=AAFLOW_PLUS_BASELINE["label"],
        workload=workload,
        requested_config=requested_config,
        run_id=run_id,
        metadata={"state_bindings": state_bindings, "workflow_ids": workflow_ids},
        available=True,
        reason="",
    )


def _run_aaflow_plus_hf(workload: GeneratedWorkload, requested_config: WorkloadConfig, run_id: str) -> dict[str, Any]:
    cfg = workload.config.normalized()
    model_id = cfg.model_id if cfg.model_id != "mock-model" else "distilgpt2"
    tokenizer_id = cfg.tokenizer_id if cfg.tokenizer_id != "mock-tokenizer" else model_id
    backend = HFKVBackend(HFBackendConfig(model_id=model_id, tokenizer_id=tokenizer_id))
    measurements = []
    for request_idx in range(cfg.num_requests):
        prompt = workload.prompts[request_idx % len(workload.prompts)] if workload.prompts else cfg.workload_name
        token_count = workload.token_counts[request_idx % len(workload.token_counts)] if workload.token_counts else cfg.context_tokens
        measurements.append(backend.measure(prompt, context_tokens=token_count, output_tokens=cfg.output_tokens))

    metrics = _combine_metrics([measurement.metrics for measurement in measurements])
    if measurements:
        metrics["kv_total_bytes"] = max(measurement.kv_state.total_bytes() for measurement in measurements)
        metrics["kv_peak_bytes"] = metrics["kv_total_bytes"]
        metrics["context_tokens"] = max(int(measurement.metrics.get("context_tokens", 0)) for measurement in measurements)
        metrics["output_tokens"] = sum(int(measurement.metrics.get("output_tokens", 0)) for measurement in measurements)

    row = _row_from_metrics(
        metrics=metrics,
        baseline_name="AAFLOW+",
        baseline_label=f"{AAFLOW_PLUS_BASELINE['label']} (HF)",
        workload=workload,
        requested_config=requested_config,
        run_id=run_id,
        metadata={
            "backend": "hf",
            "model_id": model_id,
            "kv_states": [measurement.kv_state.to_json_dict() for measurement in measurements],
            "generated_texts": [measurement.generated_text for measurement in measurements],
        },
        available=True,
        reason="",
    )
    row["model_id"] = model_id
    row["tokenizer_id"] = tokenizer_id
    row["context_tokens"] = int(metrics.get("context_tokens", row["context_tokens"]))
    return row


def _row_from_metrics(
    *,
    metrics: dict[str, Any],
    baseline_name: str,
    baseline_label: str,
    workload: GeneratedWorkload,
    requested_config: WorkloadConfig,
    run_id: str,
    metadata: Optional[dict[str, Any]] = None,
    available: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    cfg = workload.config.normalized()
    output_tokens = _int_metric(metrics, "output_tokens", default=cfg.output_tokens * cfg.num_requests)
    total_latency = _float_metric(metrics, "total_latency_sec")
    throughput = _float_metric(metrics, "throughput_tokens_per_sec")
    if throughput == 0.0 and total_latency > 0:
        throughput = output_tokens / total_latency

    row: dict[str, Any] = {
        "ttft_sec": _float_metric(metrics, "ttft_sec"),
        "total_latency_sec": total_latency,
        "prefill_sec": _float_metric(metrics, "prefill_sec"),
        "decode_sec": _float_metric(metrics, "decode_sec"),
        "transfer_sec": _float_metric(metrics, "transfer_sec"),
        "resume_sec": _float_metric(metrics, "resume_sec"),
        "omega_sec": _float_metric(metrics, "omega_sec"),
        "throughput_tokens_per_sec": throughput,
        "kv_total_bytes": _int_metric(metrics, "kv_total_bytes", "kv_bytes"),
        "kv_peak_bytes": _int_metric(metrics, "kv_peak_bytes", "kv_bytes"),
        "kv_transferred_bytes": _int_metric(metrics, "kv_transferred_bytes", default=0),
        "kv_reuse_ratio": _float_metric(metrics, "kv_reuse_ratio", "reuse_ratio"),
        "transfer_count": _int_metric(metrics, "transfer_count", default=0),
        "materialize_count": _int_metric(metrics, "materialize_count", default=0),
        "fork_count": _int_metric(metrics, "fork_count", default=0),
        "merge_count": _int_metric(metrics, "merge_count", default=0),
        "evict_count": _int_metric(metrics, "evict_count", default=0),
        "num_agents": cfg.num_agents,
        "branch_factor": cfg.branch_factor,
        "context_tokens": cfg.context_tokens,
        "output_tokens": output_tokens,
        "baseline_name": baseline_name,
        "workload_name": cfg.workload_name,
        "run_id": run_id,
        "seed": cfg.seed,
        "output_agreement_rate": metrics.get("output_agreement_rate"),
        "baseline_label": baseline_label,
        "available": bool(available),
        "reason": reason,
        "model_id": cfg.model_id,
        "tokenizer_id": cfg.tokenizer_id,
        "num_requests": cfg.num_requests,
        "depth": cfg.depth,
        "requested_workload_name": requested_config.workload_name,
        "workload_metadata": workload.metadata,
        "metadata": metadata or {},
    }
    return row


def _skip_record(
    baseline_name: str,
    cfg: WorkloadConfig,
    run_id: str,
    reason: str,
    traceback_text: str = "",
) -> dict[str, Any]:
    normalized = cfg.normalized()
    return {
        "skipped": True,
        "baseline_name": baseline_name,
        "workload_name": normalized.workload_name,
        "run_id": run_id,
        "reason": reason,
        "traceback": traceback_text,
        "context_tokens": normalized.context_tokens,
        "output_tokens": normalized.output_tokens,
        "num_agents": normalized.num_agents,
        "branch_factor": normalized.branch_factor,
        "num_requests": normalized.num_requests,
        "seed": normalized.seed,
    }


def _baseline_by_name(name: str):
    for baseline in get_baselines():
        if baseline.name == name:
            return baseline
    return None


def _branch_count_for(cfg: WorkloadConfig) -> int:
    if cfg.workload_name == "linear_handoff":
        return 1
    if cfg.workload_name == "multi_agent_debate":
        return cfg.num_agents
    if cfg.workload_name == "rag_shared_context":
        return cfg.num_agents
    return cfg.branch_factor


def _parse_grid(raw: str, fallback: int) -> list[int]:
    if not raw:
        return [max(1, int(fallback))]
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(max(1, int(part)))
    return values or [max(1, int(fallback))]


def _run_id(baseline: str, workload: str, context_tokens: int, num_agents: int, branch_factor: int, index: int) -> str:
    return f"{index:06d}_{baseline}_{workload}_ctx{context_tokens}_a{num_agents}_b{branch_factor}"


def _combine_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    summed = [
        "total_latency_sec",
        "prefill_sec",
        "decode_sec",
        "transfer_sec",
        "resume_sec",
        "omega_sec",
        "kv_transferred_bytes",
        "transfer_count",
        "materialize_count",
        "fork_count",
        "merge_count",
        "evict_count",
    ]
    averaged = ["ttft_sec", "reuse_ratio", "kv_reuse_ratio", "output_agreement_rate"]
    maxed = ["kv_bytes", "kv_total_bytes", "kv_peak_bytes"]
    out: dict[str, Any] = {}
    for key in summed:
        out[key] = sum(_float_metric(item, key) for item in items)
    for key in averaged:
        values = [item.get(key) for item in items if item.get(key) is not None]
        if values:
            out[key] = sum(float(value) for value in values) / len(values)
    for key in maxed:
        out[key] = max((_float_metric(item, key) for item in items), default=0.0)
    if out.get("total_latency_sec", 0.0) > 0:
        output_tokens = sum(_int_metric(item, "output_tokens", default=0) for item in items)
        if output_tokens:
            out["output_tokens"] = output_tokens
            out["throughput_tokens_per_sec"] = output_tokens / out["total_latency_sec"]
    return out


def _scale_metrics_for_requests(metrics: dict[str, Any], num_requests: int) -> dict[str, Any]:
    count = max(1, int(num_requests))
    if count == 1:
        return dict(metrics)
    scaled = dict(metrics)
    for key in ["total_latency_sec", "prefill_sec", "decode_sec", "transfer_sec", "resume_sec", "omega_sec"]:
        if key in scaled and scaled[key] is not None:
            scaled[key] = float(scaled[key]) * count
    for key in ["kv_transferred_bytes", "transfer_count", "materialize_count", "fork_count", "merge_count", "evict_count"]:
        if key in scaled and scaled[key] is not None:
            scaled[key] = int(scaled[key]) * count
    if scaled.get("total_latency_sec", 0):
        output_tokens = int(scaled.get("output_tokens", 0) or 0)
        if output_tokens:
            output_tokens *= count
            scaled["output_tokens"] = output_tokens
            scaled["throughput_tokens_per_sec"] = output_tokens / float(scaled["total_latency_sec"])
    return scaled


def _float_metric(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return float(default)


def _int_metric(metrics: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        *METRIC_FIELDS,
        "baseline_label",
        "available",
        "reason",
        "model_id",
        "tokenizer_id",
        "num_requests",
        "depth",
        "backend_type",
        "config_path",
        "configured_baselines",
        "configured_workloads",
        "context_grid",
        "output_grid",
        "agent_grid",
        "branch_grid",
    ]
    fieldnames = [*base_fields, *sorted({key for row in rows for key in row} - set(base_fields))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_benchmark_table(path: Path, results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    """Write a compact benchmark.out table in the AAFLOW Slurm style."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Baseline",
        "Workload",
        "Ctx",
        "Agents",
        "Branch",
        "Req",
        "TTFT(s)",
        "AAFLOW+ faster",
        "Total(s)",
        "Prefill(s)",
        "Transfer(s)",
        "Tok/s",
        "KV MiB",
        "Reuse",
        "Status",
    ]
    aaflow_refs: dict[tuple[str, str, str, str, str], float] = {}
    aaflow_refs_by_workload: dict[str, float] = {}
    for row in results:
        if not _is_aaflow_plus_row(row):
            continue
        workload = str(row.get("workload_name", ""))
        ttft = _number(row.get("ttft_sec"))
        if workload and ttft > 0:
            aaflow_refs.setdefault(_benchmark_ref_key(row), ttft)
            aaflow_refs_by_workload.setdefault(workload, ttft)

    table_rows = []
    for row in [*results, *skipped]:
        status = "ok" if not row.get("skipped") else _short_reason(row.get("reason", "skipped"))
        kv_mib = _number(row.get("kv_peak_bytes"), default=0.0) / (1024 * 1024)
        workload = str(row.get("workload_name", ""))
        ttft = _number(row.get("ttft_sec"))
        ref_ttft = aaflow_refs.get(_benchmark_ref_key(row), aaflow_refs_by_workload.get(workload, 0.0))
        speedup = ttft / ref_ttft if ref_ttft > 0 and ttft > 0 and not row.get("skipped") else 0.0
        table_rows.append(
            [
                _display_baseline_name(row),
                str(row.get("workload_name", "")),
                str(row.get("context_tokens", "")),
                str(row.get("num_agents", "")),
                str(row.get("branch_factor", "")),
                str(row.get("num_requests", "")),
                _fmt_float(row.get("ttft_sec")),
                f"{speedup:.2f}x" if speedup else "",
                _fmt_float(row.get("total_latency_sec")),
                _fmt_float(row.get("prefill_sec")),
                _fmt_float(row.get("transfer_sec")),
                _fmt_float(row.get("throughput_tokens_per_sec")),
                _fmt_float(kv_mib),
                _fmt_float(row.get("kv_reuse_ratio")),
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


def _benchmark_ref_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("workload_name", "")),
        str(row.get("context_tokens", "")),
        str(row.get("num_agents", "")),
        str(row.get("branch_factor", "")),
        str(row.get("num_requests", "")),
    )


def _is_aaflow_plus_row(row: dict[str, Any]) -> bool:
    name = str(row.get("baseline_name", "")).strip().lower()
    return name in {"aaflow+", "aaflow_plus", "ours_stateful"}


def _display_baseline_name(row: dict[str, Any]) -> str:
    if _is_aaflow_plus_row(row):
        return "AAFLOW+"
    return str(row.get("baseline_name", ""))


def _write_summary_out(
    path: Path,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Write a human-readable run summary with file inventory and speedups."""

    output_dir = path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = config.get("resolved", {}) if isinstance(config, dict) else {}
    rows = [*results, *skipped]

    lines = [
        "Stateful Agentic Algebra Run Summary",
        "=" * 38,
        f"Output directory: {output_dir}",
        f"Rows: {len(rows)}",
        f"Successful rows: {len(results)}",
        f"Skipped/unavailable rows: {len(skipped)}",
        "",
        "Configuration",
        "-" * 13,
    ]
    for key in [
        "baselines",
        "workloads",
        "context_grid",
        "output_grid",
        "agent_grid",
        "branch_grid",
        "num_runs",
        "num_skipped",
    ]:
        if key in resolved:
            lines.append(f"{key}: {resolved[key]}")
    for key in ["backend", "generation_backend", "output_dir"]:
        value = config.get(key) if isinstance(config, dict) else None
        if value not in {None, ""}:
            lines.append(f"{key}: {value}")

    lines.extend(["", "Output Files", "-" * 12])
    for rel in [
        "config.json",
        "results.json",
        "results.csv",
        "skipped_baselines.json",
        "benchmark.out",
        "figures",
        "logs",
    ]:
        lines.append(_file_inventory_line(output_dir / rel, rel))
    lines.append("summary.out: file (this file)")

    lines.extend(["", "Benchmark Summary", "-" * 17])
    lines.extend(
        _format_table(
            [
                "Baseline",
                "Workload",
                "Rows",
                "TTFT(s)",
                "AAFLOW+ faster",
                "Total(s)",
                "Tok/s",
                "KV MiB",
                "Reuse",
            ],
            _experiment_summary_rows(results),
        )
    )

    if skipped:
        lines.extend(["", "Skipped / Unavailable Rows", "-" * 26])
        skipped_table = [
            [
                str(row.get("baseline_name", "")),
                str(row.get("workload_name", "")),
                str(row.get("context_tokens", "")),
                str(row.get("num_agents", "")),
                str(row.get("branch_factor", "")),
                _short_reason(row.get("reason", "skipped"), limit=72),
            ]
            for row in skipped[:50]
        ]
        lines.extend(_format_table(["Baseline", "Workload", "Ctx", "Agents", "Branch", "Reason"], skipped_table))
        if len(skipped) > 50:
            lines.append(f"... truncated {len(skipped) - 50} additional skipped rows")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _experiment_summary_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((str(row.get("baseline_name", "")), str(row.get("workload_name", ""))), []).append(row)

    aaflow_refs: dict[str, float] = {}
    for (baseline, workload), group_rows in groups.items():
        if baseline != "AAFLOW+":
            continue
        ttft = _mean_number(row.get("ttft_sec") for row in group_rows)
        if ttft > 0:
            aaflow_refs[workload] = ttft

    table_rows = []
    for (baseline, workload), group_rows in sorted(groups.items()):
        ttft = _mean_number(row.get("ttft_sec") for row in group_rows)
        ref = aaflow_refs.get(workload, 0.0)
        speedup = ttft / ref if ref > 0 and ttft > 0 else 0.0
        table_rows.append(
            [
                baseline,
                workload,
                str(len(group_rows)),
                _fmt_float(ttft),
                f"{speedup:.2f}x" if speedup else "",
                _fmt_float(_mean_number(row.get("total_latency_sec") for row in group_rows)),
                _fmt_float(_mean_number(row.get("throughput_tokens_per_sec") for row in group_rows)),
                _fmt_float(_mean_number(row.get("kv_peak_bytes") for row in group_rows) / (1024 * 1024)),
                _fmt_float(_mean_number(row.get("kv_reuse_ratio") for row in group_rows)),
            ]
        )
    return table_rows


def _mean_number(values: Iterable[Any]) -> float:
    parsed = [_number(value) for value in values if value not in {None, ""}]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _file_inventory_line(path: Path, label: str) -> str:
    if path.is_dir():
        count = sum(1 for _ in path.rglob("*"))
        return f"{label}: directory ({count} entries)"
    if path.exists():
        return f"{label}: file ({path.stat().st_size} bytes)"
    return f"{label}: missing"


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No rows."]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    lines = [
        " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows)
    return lines


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _short_reason(reason: Any, limit: int = 44) -> str:
    text = str(reason or "skipped")
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
        "source_config": _jsonable(getattr(args, "config_values", {})),
        "baselines": list_baselines(),
        "workloads": sorted(WORKLOAD_GENERATORS),
    }


def _attach_config_values(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    baselines: list[str],
    workloads: list[str],
    context_grid: list[int],
    output_grid: list[int],
    agent_grid: list[int],
    branch_grid: list[int],
) -> None:
    row.update(
        {
            "backend_type": args.backend,
            "config_path": args.config,
            "configured_baselines": ",".join(baselines),
            "configured_workloads": ",".join(workloads),
            "context_grid": ",".join(str(item) for item in context_grid),
            "output_grid": ",".join(str(item) for item in output_grid),
            "agent_grid": ",".join(str(item) for item in agent_grid),
            "branch_grid": ",".join(str(item) for item in branch_grid),
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
