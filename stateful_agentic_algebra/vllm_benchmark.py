"""vLLM serving benchmark wrapper.

This module runs real vLLM serving benchmarks through the vLLM command-line
interface when vLLM is installed. It intentionally does not import vLLM at
module import time, so the rest of Stateful Agentic Algebra remains usable on
CPU-only or dependency-light environments.

The wrapper starts `vllm serve`, waits for the OpenAI-compatible server, runs
`vllm bench serve` against synthetic random prompts, parses reported latency
metrics such as TTFT, TPOT, ITL, and E2EL, then kills the server process group.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import URLError
from urllib.request import urlopen


def check_vllm_available() -> bool:
    """Return True when the vLLM CLI or Python package appears available."""

    return shutil.which("vllm") is not None or importlib.util.find_spec("vllm") is not None


def launch_vllm_server(
    model_id: str,
    tensor_parallel_size: int,
    port: int,
    extra_args: Optional[list[str]] = None,
    stdout_path: Optional[str | Path] = None,
    stderr_path: Optional[str | Path] = None,
) -> subprocess.Popen[Any]:
    """Launch `vllm serve` and return the server process.

    The process is started in its own process group on POSIX systems so cleanup
    can kill child workers reliably.
    """

    command = [
        *_vllm_cli_base(),
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--tensor-parallel-size",
        str(max(1, int(tensor_parallel_size))),
    ]
    command.extend(extra_args or [])
    stdout_file = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.DEVNULL
    stderr_file = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
    try:
        return subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception:
        if hasattr(stdout_file, "close"):
            stdout_file.close()
        if hasattr(stderr_file, "close"):
            stderr_file.close()
        raise


def wait_for_server(port: int, timeout_sec: float = 600.0, poll_interval_sec: float = 2.0) -> bool:
    """Wait until the vLLM OpenAI-compatible server responds."""

    deadline = time.time() + float(timeout_sec)
    urls = [
        f"http://127.0.0.1:{int(port)}/v1/models",
        f"http://localhost:{int(port)}/v1/models",
    ]
    while time.time() < deadline:
        for url in urls:
            try:
                with urlopen(url, timeout=2.0) as response:
                    if 200 <= int(response.status) < 500:
                        return True
            except (URLError, TimeoutError, OSError, socket.timeout):
                pass
        time.sleep(float(poll_interval_sec))
    return False


def run_vllm_bench_serve(
    model_id: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    request_rate: float | str,
    port: int,
    output_dir: str | Path,
    timeout_sec: float = 1800.0,
    extra_args: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run `vllm bench serve` and write raw output plus parsed metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command = [
        *_vllm_cli_base(),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--model",
        model_id,
        "--tokenizer",
        model_id,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(max(1, int(input_len))),
        "--random-output-len",
        str(max(1, int(output_len))),
        "--num-prompts",
        str(max(1, int(num_prompts))),
        "--request-rate",
        str(request_rate),
        "--ignore-eos",
    ]
    command.extend(extra_args or [])
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=float(timeout_sec), check=False)
    elapsed = time.perf_counter() - started
    raw_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (output_path / "vllm_bench_raw.txt").write_text(raw_output, encoding="utf-8")
    metrics = parse_vllm_results(raw_output)
    metrics.update(
        {
            "backend": "vllm",
            "model_id": model_id,
            "input_len": int(input_len),
            "output_len": int(output_len),
            "num_prompts": int(num_prompts),
            "request_rate": request_rate,
            "port": int(port),
            "bench_elapsed_sec": elapsed,
            "bench_returncode": proc.returncode,
            "bench_command": command,
            "available": proc.returncode == 0,
        }
    )
    if proc.returncode != 0:
        metrics["reason"] = f"vllm bench serve exited with code {proc.returncode}"
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def parse_vllm_results(output: str) -> dict[str, Any]:
    """Parse common `vllm bench serve` result labels into metric fields."""

    metrics: dict[str, Any] = {}
    for raw_label, raw_value in _LABEL_RE.findall(output or ""):
        label = _normalize_label(raw_label)
        value = _parse_number(raw_value)
        if value is None:
            continue
        metrics[label] = value
        if label.endswith("_ms"):
            metrics[label[:-3] + "_sec"] = value / 1000.0

    alias_pairs = {
        "ttft_sec": ("mean_ttft_sec", "median_ttft_sec", "p50_ttft_sec"),
        "tpot_sec": ("mean_tpot_sec", "median_tpot_sec", "p50_tpot_sec"),
        "itl_sec": ("mean_itl_sec", "median_itl_sec", "p50_itl_sec"),
        "e2el_sec": ("mean_e2el_sec", "median_e2el_sec", "p50_e2el_sec", "mean_request_latency_sec"),
        "throughput_tokens_per_sec": (
            "output_token_throughput_tok_s",
            "total_token_throughput_tok_s",
            "token_throughput_tok_s",
        ),
        "request_throughput_req_per_sec": ("request_throughput_req_s",),
    }
    for target, candidates in alias_pairs.items():
        for candidate in candidates:
            if candidate in metrics:
                metrics[target] = metrics[candidate]
                break
    return metrics


def run_cli(args: argparse.Namespace) -> int:
    """Run the CLI workflow and return a process exit code."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": args.model_id,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "tensor_parallel_size": args.tensor_parallel_size,
        "port": args.port,
        "server_timeout_sec": args.server_timeout_sec,
        "bench_timeout_sec": args.bench_timeout_sec,
        "extra_args": args.extra_args or [],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if not check_vllm_available():
        message = "vLLM is not installed or no vLLM CLI is available; skipping vLLM benchmark"
        metrics = {"available": False, "skipped": True, "reason": message, **config}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "vllm_stdout.log").write_text("", encoding="utf-8")
        (output_dir / "vllm_stderr.log").write_text(message + "\n", encoding="utf-8")
        (output_dir / "vllm_bench_raw.txt").write_text("", encoding="utf-8")
        print(message)
        return 2 if args.require_vllm else 0

    server: Optional[subprocess.Popen[Any]] = None
    try:
        server = launch_vllm_server(
            model_id=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.port,
            extra_args=args.extra_args,
            stdout_path=output_dir / "vllm_stdout.log",
            stderr_path=output_dir / "vllm_stderr.log",
        )
        ready = wait_for_server(args.port, timeout_sec=args.server_timeout_sec)
        if not ready:
            raise RuntimeError(f"vLLM server did not become ready on port {args.port}")
        metrics = run_vllm_bench_serve(
            model_id=args.model_id,
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            port=args.port,
            output_dir=output_dir,
            timeout_sec=args.bench_timeout_sec,
        )
        print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2, sort_keys=True))
        return 0 if metrics.get("available", False) else 1
    except Exception as exc:
        metrics = {"available": False, "skipped": False, "reason": str(exc), **config}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        if not (output_dir / "vllm_bench_raw.txt").exists():
            (output_dir / "vllm_bench_raw.txt").write_text("", encoding="utf-8")
        print(f"vLLM benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            _terminate_process_tree(server)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM serving benchmark and parse latency metrics")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-timeout-sec", type=float, default=900.0)
    parser.add_argument("--bench-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    raise SystemExit(run_cli(parse_args(argv)))


def _vllm_cli_base() -> list[str]:
    executable = shutil.which("vllm")
    if executable:
        return [executable]
    if importlib.util.find_spec("vllm") is not None:
        return [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    raise RuntimeError("vLLM is not installed or no vLLM CLI is available")


def _terminate_process_tree(process: subprocess.Popen[Any], timeout_sec: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=timeout_sec)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass


def _normalize_label(label: str) -> str:
    text = label.strip().lower()
    text = text.replace("%", "pct")
    text = text.replace("#", "num")
    replacements = {
        "time to first token": "ttft",
        "time per output token": "tpot",
        "inter-token latency": "itl",
        "end-to-end latency": "e2el",
        "end to end latency": "e2el",
        "request throughput": "request_throughput",
        "output token throughput": "output_token_throughput",
        "total token throughput": "total_token_throughput",
        "successful requests": "successful_requests",
        "benchmark duration": "benchmark_duration",
        "total input tokens": "total_input_tokens",
        "total generated tokens": "total_generated_tokens",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("(ms)", " ms")
    text = text.replace("(s)", " sec")
    text = text.replace("(req/s)", " req_s")
    text = text.replace("(tok/s)", " tok_s")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = text.replace("p99", "p99").replace("p95", "p95").replace("p50", "p50")
    return text


def _parse_number(value: str) -> Optional[float | int]:
    text = value.strip().split()[0]
    if text.lower() in {"inf", "infinity"}:
        return float("inf")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


_LABEL_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9 /_().%#-]*?)\s*:\s*([-+0-9.eEinfINF]+)", re.MULTILINE)


if __name__ == "__main__":
    main()
