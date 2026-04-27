"""SGLang serving benchmark wrapper.

This module provides the real SGLang serving path for Stateful Agentic
Algebra.  It intentionally avoids importing SGLang at module import time so the
rest of the package continues to work on machines where SGLang is not
installed.

The preferred path is:

1. launch `sglang serve` or `python -m sglang.launch_server`,
2. wait for the OpenAI-compatible HTTP endpoint,
3. run SGLang's packaged serving benchmark when available,
4. fall back to a small OpenAI-compatible HTTP benchmark when the benchmark CLI
   is missing or incompatible with the installed SGLang version.

The collected fields match the vLLM wrapper where possible: TTFT, TPOT, ITL,
E2EL, token throughput, request throughput, raw logs, and a metrics JSON file.
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check_sglang_available(python_bin: str | None = None) -> bool:
    """Return True when SGLang appears importable or its CLI is available."""

    if _sglang_executable(python_bin) is not None:
        return True
    if python_bin:
        command = [
            python_bin,
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('sglang') else 1)",
        ]
        try:
            proc = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
                env=_subprocess_env(python_bin),
            )
            return proc.returncode == 0
        except Exception:
            return False
    return importlib.util.find_spec("sglang") is not None


def launch_sglang_server(
    model_id: str,
    tensor_parallel_size: int,
    port: int,
    extra_args: Optional[list[str]] = None,
    stdout_path: Optional[str | Path] = None,
    stderr_path: Optional[str | Path] = None,
    python_bin: str | None = None,
) -> subprocess.Popen[Any]:
    """Launch an SGLang server and return the server process."""

    command = [
        *_sglang_server_base(python_bin),
        "--model-path",
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
            env=_subprocess_env(python_bin),
        )
    except Exception:
        if hasattr(stdout_file, "close"):
            stdout_file.close()
        if hasattr(stderr_file, "close"):
            stderr_file.close()
        raise


def wait_for_server(port: int, timeout_sec: float = 600.0, poll_interval_sec: float = 2.0) -> bool:
    """Wait until the SGLang HTTP server responds."""

    deadline = time.time() + float(timeout_sec)
    health_urls = [
        f"http://127.0.0.1:{int(port)}/health",
        f"http://localhost:{int(port)}/health",
    ]
    while time.time() < deadline:
        for url in health_urls:
            try:
                with urlopen(url, timeout=2.0) as response:
                    if 200 <= int(response.status) < 300:
                        return True
            except (HTTPError, URLError, TimeoutError, OSError, socket.timeout):
                pass
        time.sleep(float(poll_interval_sec))
    return False


def run_sglang_bench_serve(
    model_id: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    request_rate: float | str,
    port: int,
    output_dir: str | Path,
    timeout_sec: float = 1800.0,
    extra_args: Optional[list[str]] = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    """Run a real SGLang serving benchmark and write raw output/metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bench_raw = ""
    fallback_reason = ""
    metrics: dict[str, Any] = {}

    try:
        command = _sglang_bench_command(
            model_id=model_id,
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            request_rate=request_rate,
            port=port,
            extra_args=extra_args,
            python_bin=python_bin,
        )
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(timeout_sec),
            check=False,
            env=_subprocess_env(python_bin),
        )
        bench_raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        metrics = parse_sglang_results(bench_raw)
        metrics["bench_returncode"] = proc.returncode
        metrics["bench_command"] = command
        if proc.returncode != 0:
            fallback_reason = f"sglang bench command exited with code {proc.returncode}"
    except Exception as exc:
        fallback_reason = f"sglang bench command unavailable: {exc}"

    if fallback_reason:
        fallback = _run_http_fallback(
            model_id=model_id,
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            port=port,
            timeout_sec=timeout_sec,
        )
        metrics.update(fallback)
        metrics["reason"] = fallback_reason + "; used OpenAI-compatible HTTP fallback"
        bench_raw = bench_raw + ("\n" if bench_raw else "") + json.dumps(fallback, indent=2, sort_keys=True)

    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "backend": "sglang",
            "model_id": model_id,
            "input_len": int(input_len),
            "output_len": int(output_len),
            "num_prompts": int(num_prompts),
            "request_rate": request_rate,
            "port": int(port),
            "bench_elapsed_sec": elapsed,
            "available": bool(metrics.get("available", True)),
        }
    )
    if not metrics.get("total_latency_sec") and metrics.get("e2el_sec"):
        metrics["total_latency_sec"] = metrics["e2el_sec"]
    if not metrics.get("throughput_tokens_per_sec") and elapsed > 0:
        metrics["throughput_tokens_per_sec"] = int(output_len) * int(num_prompts) / elapsed
    if not metrics.get("request_throughput_req_per_sec") and elapsed > 0:
        metrics["request_throughput_req_per_sec"] = int(num_prompts) / elapsed

    (output_path / "sglang_bench_raw.txt").write_text(bench_raw, encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def parse_sglang_results(output: str) -> dict[str, Any]:
    """Parse common SGLang/vLLM-style serving benchmark labels."""

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
        "python_bin": args.python_bin,
        "extra_args": args.extra_args or [],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if not check_sglang_available(args.python_bin):
        message = "SGLang is not installed or no SGLang CLI is available; skipping SGLang benchmark"
        metrics = {"available": False, "skipped": True, "reason": message, **config}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "sglang_stdout.log").write_text("", encoding="utf-8")
        (output_dir / "sglang_stderr.log").write_text(message + "\n", encoding="utf-8")
        (output_dir / "sglang_bench_raw.txt").write_text("", encoding="utf-8")
        print(message)
        return 2 if args.require_sglang else 0

    server: Optional[subprocess.Popen[Any]] = None
    try:
        server = launch_sglang_server(
            model_id=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.port,
            extra_args=args.extra_args,
            stdout_path=output_dir / "sglang_stdout.log",
            stderr_path=output_dir / "sglang_stderr.log",
            python_bin=args.python_bin,
        )
        ready = wait_for_server(args.port, timeout_sec=args.server_timeout_sec)
        if not ready:
            raise RuntimeError(f"SGLang server did not become ready on port {args.port}")
        metrics = run_sglang_bench_serve(
            model_id=args.model_id,
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            port=args.port,
            output_dir=output_dir,
            timeout_sec=args.bench_timeout_sec,
            python_bin=args.python_bin,
        )
        print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2, sort_keys=True))
        return 0 if metrics.get("available", False) else 1
    except Exception as exc:
        metrics = {"available": False, "skipped": False, "reason": str(exc), **config}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        if not (output_dir / "sglang_bench_raw.txt").exists():
            (output_dir / "sglang_bench_raw.txt").write_text("", encoding="utf-8")
        print(f"SGLang benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            terminate_process_tree(server)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGLang serving benchmark and parse latency metrics")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-timeout-sec", type=float, default=900.0)
    parser.add_argument("--bench-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--python-bin", default=os.environ.get("SGLANG_PYTHON_BIN", ""))
    parser.add_argument("--require-sglang", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    raise SystemExit(run_cli(parse_args(argv)))


def terminate_process_tree(process: subprocess.Popen[Any], timeout_sec: float = 30.0) -> None:
    """Terminate a server process group."""

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


def _sglang_server_base(python_bin: str | None) -> list[str]:
    executable = _sglang_executable(python_bin)
    if executable:
        return [executable, "serve"]
    python = python_bin or sys.executable
    return [python, "-m", "sglang.launch_server"]


def _sglang_bench_command(
    *,
    model_id: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    request_rate: float | str,
    port: int,
    extra_args: Optional[list[str]],
    python_bin: str | None,
) -> list[str]:
    python = python_bin or sys.executable
    command = [
        python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
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
    ]
    command.extend(extra_args or [])
    return command


def _sglang_executable(python_bin: str | None = None) -> str | None:
    if python_bin:
        candidate = Path(python_bin).resolve().parent / "sglang"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("sglang")


def _subprocess_env(python_bin: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if python_bin:
        bin_dir = str(Path(python_bin).resolve().parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        lib_paths = _nvidia_lib_paths(python_bin)
        if lib_paths:
            env["LD_LIBRARY_PATH"] = lib_paths + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _nvidia_lib_paths(python_bin: str) -> str:
    command = [
        python_bin,
        "-c",
        (
            "import site; "
            "from pathlib import Path; "
            "roots=[Path(p) for p in site.getsitepackages()]; "
            "print(':'.join(str(p) for root in roots for p in root.glob('nvidia/*/lib') if p.is_dir()))"
        ),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _run_http_fallback(
    *,
    model_id: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    port: int,
    timeout_sec: float,
) -> dict[str, Any]:
    prompt = _synthetic_prompt(input_len)
    latencies: list[float] = []
    generated_tokens = 0
    started = time.perf_counter()
    for _ in range(max(1, int(num_prompts))):
        request_started = time.perf_counter()
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max(1, int(output_len)),
            "temperature": 0,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{int(port)}/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=min(float(timeout_sec), 300.0)) as response:
            body = response.read().decode("utf-8")
        latency = time.perf_counter() - request_started
        latencies.append(latency)
        try:
            parsed = json.loads(body)
            usage = parsed.get("usage", {})
            generated_tokens += int(usage.get("completion_tokens", output_len))
        except Exception:
            generated_tokens += int(output_len)
    elapsed = time.perf_counter() - started
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "available": True,
        "bench_path": "openai_http_fallback",
        "ttft_sec": mean_latency,
        "e2el_sec": mean_latency,
        "total_latency_sec": mean_latency,
        "throughput_tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else 0.0,
        "request_throughput_req_per_sec": len(latencies) / elapsed if elapsed > 0 else 0.0,
        "successful_requests": len(latencies),
    }


def _synthetic_prompt(input_len: int) -> str:
    words = ["stateful", "agentic", "algebra", "benchmark", "prefix", "cache", "serving", "latency"]
    return " ".join(words[idx % len(words)] for idx in range(max(1, int(input_len))))


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
