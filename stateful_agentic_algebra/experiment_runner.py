"""Command-line experiment runner for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: runs graph-native `ours_stateful` workflows and
    baseline adapters over the same workload grid.
  - KV lifecycle: `ours_stateful` executes explicit materialize, fork,
    transfer/recompute, restricted merge, and evict through `StatefulRuntime`.
  - Metrics: writes one normalized metrics row per run to JSON and CSV,
    including skipped optional baselines in a separate file.

The runner is mock-safe. Optional vLLM, SGLang, AAFLOW, UCX, CUDA, or other
heavy dependencies are never imported eagerly, and missing optional packages
produce skipped or fallback records instead of CLI crashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .baselines import BaselineResult, get_baselines, list_baselines as list_adapter_baselines
from .metrics_stateful import METRIC_FIELDS
from .runtime import RuntimeConfig, StatefulRuntime
from .workloads import (
    GeneratedWorkload,
    WorkloadConfig,
    linear_handoff,
    multi_agent_debate,
    rag_shared_context,
    transfer_recompute_crossover,
    tree_of_thought,
)


OURS_BASELINE = {
    "name": "ours_stateful",
    "label": "Ours Stateful Agentic Algebra",
    "available": True,
    "skipped": False,
    "reason": "",
}

BASELINE_NAMES = [
    "ours_stateful",
    "dense_prefill",
    "aaflow_text",
    "vllm_local_prefix",
    "sglang_prefix",
    "distserve_style",
]

WORKLOAD_GENERATORS = {
    "linear_handoff": linear_handoff,
    "multi_agent_debate": multi_agent_debate,
    "tree_of_thought": tree_of_thought,
    "rag_shared_context": rag_shared_context,
    "transfer_recompute_crossover": transfer_recompute_crossover,
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stateful Agentic Algebra experiments")
    parser.add_argument("--list-baselines", action="store_true", help="List baseline adapters and availability")
    parser.add_argument("--baseline", choices=BASELINE_NAMES, default="ours_stateful")
    parser.add_argument("--all-baselines", action="store_true", help="Run every registered baseline")
    parser.add_argument("--workload", choices=sorted(WORKLOAD_GENERATORS), default="linear_handoff")
    parser.add_argument("--all-workloads", action="store_true", help="Run every workload generator")
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="mock-model")
    parser.add_argument("--tokenizer-id", type=str, default="mock-tokenizer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-grid", type=str, default="")
    parser.add_argument("--agent-grid", type=str, default="")
    parser.add_argument("--branch-grid", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="runs/stateful/latest")

    # Backwards-compatible aliases from the earlier runner.
    parser.add_argument("--branches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shared-prefix-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--generation-backend", choices=["mock", "aaflow", "vllm", "sglang"], default="mock", help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.branches is not None:
        args.num_agents = args.branches
        args.branch_factor = args.branches
    if args.shared_prefix_tokens is not None:
        args.context_tokens = args.shared_prefix_tokens
    if args.output_json and args.output_dir == "runs/stateful/latest":
        args.output_dir = str(Path(args.output_json).parent or Path("."))
    return args


def list_baselines() -> list[dict[str, Any]]:
    """Return the CLI baseline registry including `ours_stateful`."""

    return [dict(OURS_BASELINE), *list_adapter_baselines()]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_baselines:
        payload = {"baselines": list_baselines()}
        print(json.dumps(payload, indent=2))
        return {"results": [], "skipped_baselines": [], **payload}

    baselines = BASELINE_NAMES if args.all_baselines else [args.baseline]
    workloads = sorted(WORKLOAD_GENERATORS) if args.all_workloads else [args.workload]
    context_grid = _parse_grid(args.context_grid, args.context_tokens)
    agent_grid = _parse_grid(args.agent_grid, args.num_agents)
    branch_grid = _parse_grid(args.branch_grid, args.branch_factor)

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    run_index = 0

    for workload_name in workloads:
        for context_tokens in context_grid:
            for num_agents in agent_grid:
                for branch_factor in branch_grid:
                    cfg = WorkloadConfig(
                        workload_name=workload_name,
                        context_tokens=context_tokens,
                        output_tokens=args.output_tokens,
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
        "agent_grid": agent_grid,
        "branch_grid": branch_grid,
        "num_runs": len(results),
        "num_skipped": len(skipped),
    }

    _write_json(output_dir / "results.json", payload)
    _write_csv(output_dir / "results.csv", results)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "skipped_baselines.json", {"skipped_baselines": skipped})

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
) -> dict[str, Any]:
    cfg = workload.config.normalized()
    try:
        if baseline_name == "ours_stateful":
            return _run_ours_stateful(workload, requested_config, run_id, generation_backend)

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


def _run_ours_stateful(
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
        baseline_name="ours_stateful",
        baseline_label=OURS_BASELINE["label"],
        workload=workload,
        requested_config=requested_config,
        run_id=run_id,
        metadata={"state_bindings": state_bindings, "workflow_ids": workflow_ids},
        available=True,
        reason="",
    )


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
    fieldnames = [*METRIC_FIELDS, "baseline_label", "available", "reason", "model_id", "tokenizer_id", "num_requests", "depth"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
        "baselines": list_baselines(),
        "workloads": sorted(WORKLOAD_GENERATORS),
    }


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
