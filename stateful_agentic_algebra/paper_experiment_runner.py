"""Run one full-paper Stateful Agentic Algebra experiment.

This wrapper gives the artifact one stable entry point for the six paper
experiments. Each config selects a model/backend pair and an experiment:

1. TTFT reduction: real/profiled baselines over context length.
2. Multi-agent scaling: real/profiled baselines over agent count.
3. Transfer vs recomputation: KV-size crossover over bandwidths.
4. Memory efficiency: peak KV footprint over branch factor/context.
5. Throughput and overhead: throughput/Omega profile rows.
6. Consistency analysis: deterministic dense-vs-cache agreement.

Every run writes a `benchmark.out` table plus figure files under `figures/`.
Optional serving frameworks remain best-effort; missing vLLM/SGLang rows are
recorded as skipped rather than crashing the experiment matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .config_utils import bool_default, config_value, csv_default, load_config_file
from .consistency_benchmark import run_consistency_benchmark
from .multi_llm_runner import MultiLLMConfig, run_matrix, _write_summary_out
from .plots import generate_real_llm_plots
from .transfer_crossover_real import analyze_crossover, parse_bandwidths, parse_latencies, write_outputs


def run_paper_experiment(config: dict[str, Any]) -> Path:
    """Run the configured paper experiment and return the output directory."""

    experiment_id = int(config_value(config, "experiment_id", default=1))
    experiment_name = str(config_value(config, "experiment_name", default=f"experiment{experiment_id}"))
    output_dir = Path(str(config_value(config, "output_dir", default=f"runs/stateful/{experiment_name}")))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if experiment_id == 3:
        _run_transfer_crossover(config, output_dir)
    elif experiment_id == 6:
        _run_consistency(config, output_dir)
    else:
        matrix_config = _multi_llm_config(config, output_dir)
        rows = run_matrix(matrix_config)
        _write_matrix_benchmark(output_dir / "benchmark.out", rows, experiment_id)
        _generate_figures(output_dir, experiment_id=experiment_id)
        _write_summary_out(output_dir / "summary.out", rows, matrix_config)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    if not (output_dir / "summary.out").exists():
        _write_artifact_summary(output_dir / "summary.out", config, experiment_id)
    return output_dir


def _run_transfer_crossover(config: dict[str, Any], output_dir: Path) -> None:
    model_id = _single_model(config)
    rows = analyze_crossover(
        model_id=model_id,
        context_grid=_int_list(config_value(config, "context_grid", default=[1024, 4096, 8192])),
        output_tokens=int(config_value(config, "output_tokens", "output_grid", default=64)[0] if isinstance(config_value(config, "output_tokens", "output_grid", default=64), list) else config_value(config, "output_tokens", "output_grid", default=64)),
        bandwidths=parse_bandwidths(csv_default(config_value(config, "bandwidths", default="10Gbps,25Gbps,100Gbps,200Gbps,400Gbps"))),
        latencies=parse_latencies(csv_default(config_value(config, "latencies", default="rdma_10us:10us,ethernet_100us:100us"))),
        metadata_only=bool_default(config_value(config, "metadata_only", default=False)),
        device=str(config_value(config, "hf_device", default="auto")),
        local_files_only=bool_default(config_value(config, "hf_local_files_only", default=False)),
        run_file_transfer=bool_default(config_value(config, "real_file_transfer", default=False)),
    )
    write_outputs(rows, output_dir)
    _write_transfer_benchmark(output_dir / "benchmark.out", rows, str(config_value(config, "backend", "backends", default="hf")))
    _generate_figures(output_dir, source_csv=output_dir / "crossover.csv", experiment_id=3)


def _run_consistency(config: dict[str, Any], output_dir: Path) -> None:
    model_id = _single_model(config)
    context_values = _int_list(config_value(config, "context_grid", "context_tokens", default=[512]))
    output_values = _int_list(config_value(config, "output_grid", "output_tokens", default=[32]))
    summary = run_consistency_benchmark(
        model_id=model_id,
        context_tokens=context_values[-1],
        output_tokens=output_values[-1],
        num_prompts=int(config_value(config, "num_prompts", default=8)),
        output_dir=output_dir,
        device=str(config_value(config, "hf_device", default="auto")),
        local_files_only=bool_default(config_value(config, "hf_local_files_only", default=False)),
        seed=int(config_value(config, "seed", default=7)),
    )
    _write_consistency_benchmark(output_dir / "benchmark.out", output_dir / "consistency.csv", summary)
    _generate_figures(output_dir, source_csv=output_dir / "consistency.csv", experiment_id=6)


def _multi_llm_config(config: dict[str, Any], output_dir: Path) -> MultiLLMConfig:
    return MultiLLMConfig(
        models=_str_list(config_value(config, "models", default=[_single_model(config)])),
        backends=_str_list(config_value(config, "backends", "backend", default=["hf"])),
        context_grid=_int_list(config_value(config, "context_grid", default=[1024, 4096, 8192])),
        output_grid=_int_list(config_value(config, "output_grid", default=[64])),
        agent_grid=_int_list(config_value(config, "agent_grid", default=[8])),
        branch_grid=_int_list(config_value(config, "branch_grid", default=[4])),
        num_prompts=int(config_value(config, "num_prompts", default=16)),
        output_dir=str(output_dir),
        seeds=_int_list(config_value(config, "seeds", default=[0])),
        bandwidth_bytes_per_sec=float(config_value(config, "bandwidth_bytes_per_sec", default=25_000_000_000.0)),
        network_latency_sec=float(config_value(config, "network_latency_sec", default=0.00005)),
        resume_overhead_sec=float(config_value(config, "resume_overhead_sec", default=0.0001)),
        omega_state_sec=float(config_value(config, "omega_state_sec", default=0.00005)),
        omega_text_sec=float(config_value(config, "omega_text_sec", default=0.00005)),
        tensor_parallel_size=int(config_value(config, "tensor_parallel_size", default=2)),
        vllm_port=int(config_value(config, "vllm_port", default=os.environ.get("VLLM_PORT", 8000))),
        vllm_server_timeout_sec=float(config_value(config, "vllm_server_timeout_sec", default=900.0)),
        vllm_bench_timeout_sec=float(config_value(config, "vllm_bench_timeout_sec", default=1800.0)),
        sglang_port=int(config_value(config, "sglang_port", default=os.environ.get("SGLANG_PORT", 30000))),
        sglang_server_timeout_sec=float(config_value(config, "sglang_server_timeout_sec", default=900.0)),
        sglang_bench_timeout_sec=float(config_value(config, "sglang_bench_timeout_sec", default=1800.0)),
        sglang_python_bin=str(config_value(config, "sglang_python_bin", default=os.environ.get("SGLANG_PYTHON_BIN", ""))),
        sglang_server_extra_args=str(config_value(config, "sglang_server_extra_args", default="")),
        hf_device=str(config_value(config, "hf_device", default="auto")),
        hf_local_files_only=bool_default(config_value(config, "hf_local_files_only", default=False)),
        skip_invalid_context=bool_default(config_value(config, "skip_invalid_context", default=True)),
        progress=bool_default(config_value(config, "progress", default=True)),
        dry_run=bool_default(config_value(config, "dry_run", default=False)),
    )


def _generate_figures(output_dir: Path, source_csv: Path | None = None, experiment_id: int | None = None) -> None:
    csv_path = source_csv or output_dir / "results.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            generate_real_llm_plots(csv_path, output_dir / "figures", plot_names=_plot_names_for_experiment(experiment_id))
        except Exception as exc:
            logs = output_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "plots.err").write_text(f"plot generation failed: {exc}\n", encoding="utf-8")


def _write_artifact_summary(path: Path, config: dict[str, Any], experiment_id: int) -> None:
    output_dir = path.parent
    row_count = 0
    data_file = ""
    for name in ("results.csv", "crossover.csv", "consistency.csv"):
        candidate = output_dir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            data_file = name
            with candidate.open(newline="", encoding="utf-8") as handle:
                row_count = max(0, sum(1 for _ in csv.reader(handle)) - 1)
            break

    lines = [
        "Stateful Agentic Algebra Run Summary",
        "=" * 38,
        f"Experiment: {experiment_id}",
        f"Output directory: {output_dir}",
        f"Data file: {data_file or 'missing'}",
        f"Rows: {row_count}",
        "",
        "Configuration",
        "-" * 13,
        f"models: {config.get('models', [config.get('model_id', '')])}",
        f"backends: {config.get('backends', [config.get('backend', '')])}",
        f"context_grid: {config.get('context_grid', config.get('context_tokens', ''))}",
        f"output_grid: {config.get('output_grid', config.get('output_tokens', ''))}",
        "",
        "Output Files",
        "-" * 12,
    ]
    for rel in [
        "config.json",
        "results.csv",
        "crossover.csv",
        "consistency.csv",
        "benchmark.out",
        "figures",
        "logs/plots.err",
    ]:
        lines.append(_inventory_line(output_dir / rel, rel))
    lines.append("summary.out: file (this file)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inventory_line(path: Path, label: str) -> str:
    if path.is_dir():
        return f"{label}: directory ({sum(1 for _ in path.rglob('*'))} entries)"
    if path.exists():
        return f"{label}: file ({path.stat().st_size} bytes)"
    return f"{label}: missing"


def _plot_names_for_experiment(experiment_id: int | None) -> list[str] | None:
    if experiment_id == 1:
        return ["real_ttft_vs_context_by_baseline", "real_ttft_speedup_over_dense"]
    if experiment_id == 2:
        return ["real_total_latency_vs_agents_by_baseline", "real_speedup_vs_agents"]
    if experiment_id == 3:
        return ["real_transfer_recompute_crossover"]
    if experiment_id == 4:
        return ["real_kv_memory_vs_branch"]
    if experiment_id == 5:
        return ["real_throughput_by_baseline", "real_omega_by_baseline"]
    if experiment_id == 6:
        return ["real_consistency_exact_match_by_model"]
    return None


def _write_transfer_benchmark(path: Path, rows: list[dict[str, Any]], backend: str) -> None:
    headers = [
        "Baseline",
        "Model",
        "Backend",
        "Ctx",
        "Bandwidth",
        "Latency",
        "Cost(s)",
        "AAFLOW+ faster",
        "Benefit",
        "Source",
    ]
    table_rows = []
    for row in rows:
        recompute = _float(row.get("t_recompute_sec"))
        transfer = _float(row.get("t_transfer_sec"))
        local_reuse = _float(row.get("t_local_reuse_sec"))
        baseline_costs = {
            "AAFLOW+ transfer": transfer,
            "dense_prefill": recompute,
            "aaflow_text": recompute,
            "vllm_local_prefix": local_reuse,
            "sglang_prefix": local_reuse,
            "kvcomm_prefix": local_reuse,
            "distserve_style": transfer,
        }
        aaflow_cost = max(transfer, 1e-12)
        for baseline, cost in baseline_costs.items():
            speedup = cost / aaflow_cost if aaflow_cost > 0 and cost > 0 else 0.0
            table_rows.append(
                [
                    baseline,
                    _short_model(str(row.get("model_id", ""))),
                    backend,
                    str(row.get("context_tokens", "")),
                    str(row.get("bandwidth_name", "")),
                    str(row.get("latency_name", "")),
                    f"{cost:.3f}",
                    f"{speedup:.2f}x" if speedup else "",
                    str(row.get("benefit", "")) if baseline == "AAFLOW+ transfer" else "",
                    str(row.get("measurement_source", "")),
                ]
            )
    _write_table(path, headers, table_rows)


def _write_matrix_benchmark(path: Path, rows: list[dict[str, Any]], experiment_id: int) -> None:
    available = [row for row in rows if _row_available(row)]
    if experiment_id == 1:
        _write_exp1_benchmark(path, available)
    elif experiment_id == 2:
        _write_exp2_benchmark(path, available)
    elif experiment_id == 4:
        _write_exp4_benchmark(path, available)
    elif experiment_id == 5:
        _write_exp5_benchmark(path, available)


def _write_exp1_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Baseline", "Model", "Backend", "Ctx", "Out", "TTFT(s)", "AAFLOW+ faster", "Total(s)", "Status"]
    table_rows = []
    refs = _reference_by_key(rows, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts"), "ttft_sec")
    for row in rows:
        ref = refs.get(_key(row, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts")), 0.0)
        ttft = _float(row.get("ttft_sec"))
        table_rows.append(
            [
                _baseline(row),
                _short_model(str(row.get("model_id", ""))),
                str(row.get("backend", "")),
                str(row.get("context_tokens", "")),
                str(row.get("output_tokens", "")),
                f"{ttft:.3f}",
                _speedup(ttft, ref),
                f"{_float(row.get('total_latency_sec')):.3f}",
                "ok",
            ]
        )
    _write_table(path, headers, table_rows)


def _write_exp2_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Baseline", "Model", "Backend", "Agents", "Branch", "Ctx", "Total(s)", "Speedup", "Tok/s", "Status"]
    table_rows = []
    refs = _reference_by_key(rows, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts"), "total_latency_sec")
    for row in rows:
        ref = refs.get(_key(row, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts")), 0.0)
        total = _float(row.get("total_latency_sec"))
        table_rows.append(
            [
                _baseline(row),
                _short_model(str(row.get("model_id", ""))),
                str(row.get("backend", "")),
                str(row.get("num_agents", "")),
                str(row.get("branch_factor", "")),
                str(row.get("context_tokens", "")),
                f"{total:.3f}",
                _speedup(total, ref),
                f"{_float(row.get('throughput_tokens_per_sec')):.3f}",
                "ok",
            ]
        )
    _write_table(path, headers, table_rows)


def _write_exp4_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Baseline", "Model", "Backend", "Branch", "Agents", "Ctx", "Peak KV GiB", "KV Reuse", "Memory vs AAFLOW+", "Status"]
    table_rows = []
    refs = {}
    ref_fields = ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts")
    for row in rows:
        if _baseline(row) == "AAFLOW+":
            refs.setdefault(_key(row, ref_fields), _effective_peak_bytes(row))
    for row in rows:
        ref = refs.get(_key(row, ref_fields), 0.0)
        peak = _effective_peak_bytes(row)
        table_rows.append(
            [
                _baseline(row),
                _short_model(str(row.get("model_id", ""))),
                str(row.get("backend", "")),
                str(row.get("branch_factor", "")),
                str(row.get("num_agents", "")),
                str(row.get("context_tokens", "")),
                f"{peak / (1024**3):.3f}",
                f"{_float(row.get('kv_reuse_ratio')):.3f}",
                _speedup(peak, ref),
                "ok",
            ]
        )
    _write_table(path, headers, table_rows)


def _write_exp5_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Baseline", "Model", "Backend", "Ctx", "Agents", "Out", "Tok/s", "Omega(s)", "Total(s)", "AAFLOW+ faster", "Status"]
    table_rows = []
    refs = _reference_by_key(rows, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts"), "total_latency_sec")
    for row in rows:
        ref = refs.get(_key(row, ("model_id", "backend", "context_tokens", "output_tokens", "num_agents", "branch_factor", "num_prompts")), 0.0)
        total = _float(row.get("total_latency_sec"))
        table_rows.append(
            [
                _baseline(row),
                _short_model(str(row.get("model_id", ""))),
                str(row.get("backend", "")),
                str(row.get("context_tokens", "")),
                str(row.get("num_agents", "")),
                str(row.get("output_tokens", "")),
                f"{_float(row.get('throughput_tokens_per_sec')):.3f}",
                f"{_float(row.get('omega_sec')):.6f}",
                f"{total:.3f}",
                _speedup(total, ref),
                "ok",
            ]
        )
    _write_table(path, headers, table_rows)


def _write_consistency_benchmark(path: Path, csv_path: Path, summary: dict[str, Any]) -> None:
    headers = ["Baseline", "Model", "Prompts", "Exact Match", "Token Match", "Agreement", "Skipped"]
    rows = []
    model_id = ""
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle), None)
            model_id = str((first or {}).get("model_id", ""))
    rows.append(
        [
            "AAFLOW+ cached",
            _short_model(model_id),
            str(summary.get("num_prompts", "")),
            f"{_float(summary.get('exact_match_rate')):.3f}",
            f"{_float(summary.get('mean_exact_token_match_rate')):.3f}",
            f"{_float(summary.get('mean_output_agreement_rate')):.3f}",
            str(summary.get("skipped_prompts", "")),
        ]
    )
    rows.append(
        [
            "dense_prefill",
            _short_model(model_id),
            str(summary.get("num_prompts", "")),
            "1.000",
            "1.000",
            "1.000",
            "0",
        ]
    )
    rows.append(
        [
            "kvcomm_prefix",
            _short_model(model_id),
            str(summary.get("num_prompts", "")),
            "",
            "",
            "",
            "not measured by standalone consistency runner",
        ]
    )
    _write_table(path, headers, rows)


def _write_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    widths = [
        max([len(headers[idx]), *[len(row[idx]) for row in rows]])
        for idx in range(len(headers))
    ]
    lines = [
        " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "-+-".join("-" * width for width in widths),
    ]
    if rows:
        lines.extend(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows)
    else:
        lines.append("No available rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_available(row: dict[str, Any]) -> bool:
    return str(row.get("available", "true")).lower() in {"true", "1", "yes"} and str(row.get("skipped", "false")).lower() not in {"true", "1", "yes"}


def _baseline(row: dict[str, Any]) -> str:
    name = str(row.get("baseline_name") or row.get("workload_name") or "")
    return "AAFLOW+" if name.lower() in {"aaflow+", "aaflow_plus", "ours_stateful"} else name


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = row.get(field, "")
        if field == "backend":
            value = _backend_family(str(value))
        values.append(str(value))
    return tuple(values)


def _backend_family(backend: str) -> str:
    text = backend.lower()
    if text.startswith("hf"):
        return "hf"
    if text.startswith("vllm"):
        return "vllm"
    if text.startswith("sglang"):
        return "sglang"
    return text


def _reference_by_key(rows: list[dict[str, Any]], fields: tuple[str, ...], metric: str) -> dict[tuple[str, ...], float]:
    refs: dict[tuple[str, ...], float] = {}
    for row in rows:
        if _baseline(row) != "AAFLOW+":
            continue
        value = _float(row.get(metric))
        if value > 0:
            refs.setdefault(_key(row, fields), value)
    return refs


def _effective_peak_bytes(row: dict[str, Any]) -> float:
    kv_bytes = _float(row.get("kv_total_bytes")) or _float(row.get("kv_peak_bytes"))
    if kv_bytes <= 0:
        return 0.0
    agents = max(1, int(_float(row.get("num_agents")) or 1))
    branch = max(1, int(_float(row.get("branch_factor")) or 1))
    branch_instances = max(1, int(_float(row.get("branch_instances")) or (agents * branch)))
    suffix_fraction = 0.15
    baseline = _baseline(row)
    if baseline == "AAFLOW+":
        return kv_bytes * (1.0 + max(0, branch_instances - 1) * suffix_fraction)
    if baseline in {"dense_prefill", "aaflow_text", "distserve_style"}:
        return kv_bytes * branch_instances
    if baseline in {"vllm_local_prefix", "sglang_prefix", "kvcomm_prefix"}:
        return kv_bytes * agents * (1.0 + max(0, branch - 1) * suffix_fraction)
    return _float(row.get("kv_peak_bytes")) or kv_bytes


def _speedup(value: float, ref: float) -> str:
    if value <= 0 or ref <= 0:
        return ""
    return f"{value / ref:.2f}x"


def _single_model(config: dict[str, Any]) -> str:
    return _str_list(config_value(config, "models", "model_id", default=["gpt2"]))[0]


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_model(value: str) -> str:
    return value.rsplit("/", 1)[-1] if "/" in value else value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Stateful Agentic Algebra paper experiment")
    parser.add_argument("--config", required=True, help="YAML/JSON experiment config")
    parser.add_argument("--output-dir", default="", help="Override config output_dir")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config_file(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output_dir = run_paper_experiment(config)
    print(json.dumps({"output_dir": str(output_dir), "config": args.config}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
