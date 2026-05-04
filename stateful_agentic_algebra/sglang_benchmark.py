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


def find_available_port(start_port: int, attempts: int = 200, excluded: Optional[set[int]] = None) -> int:
    """Return the first locally bindable TCP port at or above ``start_port``.

    SGLang starts slowly enough that stale or concurrently-starting servers can
    race with a simple preflight check.  The check therefore avoids
    ``SO_REUSEADDR`` and verifies both bindability and connectability.
    """

    base = max(1, int(start_port))
    blocked = excluded or set()
    for offset in range(max(1, int(attempts))):
        port = base + offset
        if port > 65535:
            break
        if port in blocked:
            continue
        if _port_is_available(port):
            return port
    raise RuntimeError(f"no free localhost port found in range {base}-{min(65535, base + attempts - 1)}")


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


def wait_for_launched_server(
    port: int,
    process: subprocess.Popen[Any],
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 2.0,
    stderr_path: Optional[str | Path] = None,
) -> bool:
    """Wait for a launched SGLang server and fail fast on process/port errors."""

    deadline = time.time() + float(timeout_sec)
    health_urls = [
        f"http://127.0.0.1:{int(port)}/health",
        f"http://localhost:{int(port)}/health",
    ]
    last_stderr_size = 0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited before readiness on port {port} "
                f"with code {process.returncode}: {_tail_file(stderr_path)}"
            )
        stderr_text = _tail_file(stderr_path, max_bytes=4096, start_at=last_stderr_size)
        if stderr_path:
            try:
                last_stderr_size = max(last_stderr_size, Path(stderr_path).stat().st_size)
            except OSError:
                pass
        lowered = stderr_text.lower()
        if "address already in use" in lowered or "error while attempting to bind" in lowered:
            raise RuntimeError(f"SGLang server failed to bind port {port}: {_tail_file(stderr_path)}")
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

    _purge_stale_jit_caches(_subprocess_env(args.python_bin))

    server: Optional[subprocess.Popen[Any]] = None
    try:
        actual_port = int(args.port)
        attempted_ports: set[int] = set()
        last_error: Exception | None = None
        for launch_attempt in range(4):
            if server is not None:
                previous_port = actual_port
                terminate_process_tree(server)
                wait_for_port_release(previous_port)
                server = None
            actual_port = find_available_port(int(args.port) + launch_attempt * 251, excluded=attempted_ports)
            attempted_ports.add(actual_port)
            if actual_port != int(args.port):
                print(f"SGLang port {args.port} is busy; using free port {actual_port}", file=sys.stderr)
            config["requested_port"] = args.port
            config["port"] = actual_port
            (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
            try:
                server = launch_sglang_server(
                    model_id=args.model_id,
                    tensor_parallel_size=args.tensor_parallel_size,
                    port=actual_port,
                    extra_args=args.extra_args,
                    stdout_path=output_dir / "sglang_stdout.log",
                    stderr_path=output_dir / "sglang_stderr.log",
                    python_bin=args.python_bin,
                )
                ready = wait_for_launched_server(
                    actual_port,
                    server,
                    timeout_sec=args.server_timeout_sec,
                    stderr_path=output_dir / "sglang_stderr.log",
                )
                if not ready:
                    raise RuntimeError(f"SGLang server did not become ready on port {actual_port}")
                break
            except Exception as exc:
                last_error = exc
                if "bind" not in str(exc).lower() and "address already in use" not in str(exc).lower():
                    raise
        else:
            raise RuntimeError(f"SGLang server failed after port retries: {last_error}")
        metrics = run_sglang_bench_serve(
            model_id=args.model_id,
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            port=actual_port,
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
            wait_for_port_release(actual_port)


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

    # SGLang can leave worker children alive after the launch parent has
    # already exited, especially after late uvicorn bind failures. Always try
    # to signal the process group instead of returning early on a polled parent.
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


def wait_for_port_release(port: int, timeout_sec: float = 20.0, poll_interval_sec: float = 0.5) -> bool:
    """Wait until localhost ``port`` can be bound by a new server."""

    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if _port_is_available(int(port)):
            return True
        time.sleep(float(poll_interval_sec))
    return _port_is_available(int(port))


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", int(port))) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
            sock.listen(1)
        except OSError:
            return False
    return True


def _tail_file(path: Optional[str | Path], max_bytes: int = 2048, start_at: int = 0) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        with p.open("rb") as handle:
            if start_at > 0:
                handle.seek(start_at)
                data = handle.read(max_bytes)
            else:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                data = handle.read(max_bytes)
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _sglang_server_base(python_bin: str | None) -> list[str]:
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
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cuda_home = _resolve_cuda_home(env)
    path_parts: list[str] = []
    lib_parts: list[str] = []
    if cuda_home:
        env["CUDA_HOME"] = str(cuda_home)
        env["CUDA_PATH"] = str(cuda_home)
        path_parts.append(str(cuda_home / "bin"))
        lib64 = cuda_home / "lib64"
        if lib64.is_dir():
            lib_parts.append(str(lib64))
    if python_bin:
        bin_dir = str(Path(python_bin).resolve().parent)
        path_parts.append(bin_dir)
        lib_paths = _nvidia_lib_paths(python_bin)
        if lib_paths:
            lib_parts.append(lib_paths)
    if path_parts:
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    if lib_parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_parts + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _resolve_cuda_home(env: dict[str, str]) -> Optional[Path]:
    """Find a CUDA toolkit with a real ``nvcc`` for SGLang JIT subprocesses."""

    candidates: list[Path] = []
    for key in ("SAA_CUDA_HOME",):
        value = env.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    user = env.get("USER")
    if user:
        candidates.append(Path("/raid") / user / "cuda-12.8")
    candidates.append(Path("/raid/arup/cuda-12.8"))
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = env.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(Path("/usr/local/cuda"))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        nvcc = candidate / "bin" / "nvcc"
        if nvcc.exists() and os.access(nvcc, os.X_OK):
            return candidate
    return None


def _purge_stale_jit_caches(env: dict[str, str]) -> None:
    """Remove JIT cache directories generated with a stale CUDA toolkit path.

    FlashInfer and tvm-ffi write absolute CUDA paths into ``build.ninja``. If a
    cache was created while ``/usr/local/cuda`` pointed at CUDA 11, later runs
    can keep invoking that stale compiler even after ``CUDA_HOME`` is fixed.
    """

    cuda_home = env.get("CUDA_HOME") or ""
    if not cuda_home:
        return
    expected = str(Path(cuda_home).resolve())
    cache_roots = [
        Path.home() / ".cache" / "flashinfer",
        Path.home() / ".cache" / "tvm-ffi",
    ]
    for root in cache_roots:
        if not root.exists():
            continue
        try:
            ninja_files = list(root.rglob("build.ninja"))
        except OSError:
            continue
        for ninja_file in ninja_files:
            if not ninja_file.exists():
                continue
            try:
                text = ninja_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "/usr/local/cuda" not in text and expected in text:
                continue
            cache_dir = ninja_file.parent
            try:
                shutil.rmtree(cache_dir)
            except OSError:
                pass


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
    prompt = _synthetic_prompt(input_len, model_id=model_id)
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

def _synthetic_prompt(input_len: int, model_id: str | None = None) -> str:
    """Build a synthetic prompt that does not exceed ``input_len`` tokens.

    SGLang validates requests after tokenization.  The old fallback generated
    ``input_len`` whitespace-delimited words, which can tokenize to more than
    ``input_len`` model tokens for SentencePiece/BPE tokenizers and caused
    HTTP 400 errors near the context limit.  When Transformers is available,
    generate through the model tokenizer and trim until re-tokenization fits.
    """

    target_tokens = max(1, int(input_len))
    if model_id:
        tokenized = _token_bounded_prompt(model_id, target_tokens)
        if tokenized:
            return tokenized
    # Conservative fallback when tokenizer loading is unavailable.
    return _synthetic_prompt_text(max(1, target_tokens // 2))


def _token_bounded_prompt(model_id: str, target_tokens: int) -> str:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token())
    except Exception:
        return ""

    seed_text = " stateful agentic algebra benchmark prefix cache serving latency"
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        return ""

    ids = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    for _ in range(128):
        if not ids:
            break
        prompt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if 0 < len(encoded) <= target_tokens:
            return prompt
        trim_by = max(1, len(encoded) - target_tokens)
        if trim_by >= len(ids):
            ids = ids[:-1]
        else:
            ids = ids[:-trim_by]
    return ""


def _synthetic_prompt_text(word_count: int) -> str:
    words = ["stateful", "agentic", "algebra", "benchmark", "prefix", "cache", "serving", "latency"]
    return " ".join(words[idx % len(words)] for idx in range(max(1, int(word_count))))


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_file = Path.home() / ".hf_token"
    if not token_file.exists():
        return None
    try:
        text = token_file.read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(r"HF_TOKEN=([^\n\r\s;]+)", text)
    if match:
        return match.group(1).strip().strip("\"'")
    stripped = text.strip()
    return stripped if stripped.startswith("hf_") else None


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
