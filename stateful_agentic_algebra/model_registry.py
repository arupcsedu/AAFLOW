"""Static model registry for Stateful Agentic Algebra benchmarks.

Listing this registry never downloads models and never calls the Hugging Face
Hub. Availability is a conservative local/environment check intended for
experiment planning:
  - smoke models are available when a compatible backend package is importable;
  - gated/auth-required models are marked unavailable unless HF auth appears to
    be configured;
  - large tensor-parallel models are marked unavailable unless their requested
    tensor parallel size is greater than one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class ModelSpec:
    """Metadata for one benchmarkable LLM."""

    model_id: str
    family: str
    size_label: str
    backend_options: list[str] = field(default_factory=list)
    default_dtype: str = "auto"
    max_context: int = 2048
    requires_auth: bool = False
    tensor_parallel_size: int = 1
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_model_registry() -> list[ModelSpec]:
    """Return the default static model registry."""

    return [
        ModelSpec(
            model_id="distilgpt2",
            family="gpt2",
            size_label="82M",
            backend_options=["hf"],
            default_dtype="float32",
            max_context=1024,
            requires_auth=False,
            tensor_parallel_size=1,
            notes="Small CPU smoke-test model; no auth required.",
        ),
        ModelSpec(
            model_id="gpt2",
            family="gpt2",
            size_label="124M",
            backend_options=["hf"],
            default_dtype="float32",
            max_context=1024,
            requires_auth=False,
            tensor_parallel_size=1,
            notes="Small CPU-compatible smoke-test model.",
        ),
        ModelSpec(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            family="llama",
            size_label="8B",
            backend_options=["hf", "vllm"],
            default_dtype="bfloat16",
            max_context=8192,
            requires_auth=True,
            tensor_parallel_size=1,
            notes="Gated Meta model; requires Hugging Face access approval and token.",
        ),
        ModelSpec(
            model_id="meta-llama/Llama-2-7b-hf",
            family="llama",
            size_label="7B",
            backend_options=["hf", "vllm"],
            default_dtype="float16",
            max_context=4096,
            requires_auth=True,
            tensor_parallel_size=1,
            notes="Gated Meta model; listed as an alternative Llama-2 baseline.",
        ),
        ModelSpec(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            family="mistral",
            size_label="7B",
            backend_options=["hf", "vllm"],
            default_dtype="bfloat16",
            max_context=32768,
            requires_auth=False,
            tensor_parallel_size=1,
            notes="Instruction model suitable for HF or vLLM benchmarking.",
        ),
        ModelSpec(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            family="qwen",
            size_label="7B",
            backend_options=["hf", "vllm"],
            default_dtype="bfloat16",
            max_context=32768,
            requires_auth=False,
            tensor_parallel_size=1,
            notes="Instruction model with long-context support.",
        ),
        ModelSpec(
            model_id="Qwen/Qwen2.5-32B-Instruct",
            family="qwen",
            size_label="32B",
            backend_options=["vllm"],
            default_dtype="bfloat16",
            max_context=32768,
            requires_auth=False,
            tensor_parallel_size=2,
            notes="Larger optional model; intended for tensor-parallel vLLM runs.",
        ),
    ]


def list_models(registry: Optional[Iterable[ModelSpec]] = None) -> list[dict[str, Any]]:
    """Return model specs with availability annotations."""

    return [
        {**spec.to_json_dict(), **model_availability(spec)}
        for spec in (list(registry) if registry is not None else default_model_registry())
    ]


def model_availability(spec: ModelSpec) -> dict[str, Any]:
    """Conservative local/environment availability check."""

    reasons = []
    backend_status = [_backend_status(backend) for backend in spec.backend_options]
    backend_available = any(status["available"] for status in backend_status)

    if not backend_available:
        backend_reasons = [f"{status['backend']}: {status['reason']}" for status in backend_status if status["reason"]]
        reasons.append("none of the requested backends are usable without missing local packages")
        reasons.extend(backend_reasons)
    if spec.requires_auth and not _has_hf_auth():
        reasons.append("model is gated/auth-required and no Hugging Face token was detected")
    if _is_large_tensor_parallel_model(spec) and spec.tensor_parallel_size <= 1:
        reasons.append("large model requires tensor_parallel_size > 1")

    return {
        "available": not reasons,
        "reason": "; ".join(reasons),
    }


def _backend_status(backend: str) -> dict[str, Any]:
    if backend == "mock":
        return {"backend": backend, "available": True, "reason": ""}
    if backend == "hf":
        missing = [name for name in ("torch", "transformers", "huggingface_hub", "filelock") if not _module_available(name)]
        return {
            "backend": backend,
            "available": not missing,
            "reason": "" if not missing else f"missing Python packages: {', '.join(missing)}",
        }
    if backend == "vllm":
        missing = [] if _module_available("vllm") else ["vllm"]
        return {
            "backend": backend,
            "available": not missing,
            "reason": "" if not missing else "missing Python packages: vllm",
        }
    return {"backend": backend, "available": False, "reason": f"unknown backend {backend}"}


def get_model_spec(model_id: str, registry: Optional[Iterable[ModelSpec]] = None) -> Optional[ModelSpec]:
    """Find a model spec by id."""

    for spec in registry or default_model_registry():
        if spec.model_id == model_id:
            return spec
    return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_hf_auth() -> bool:
    return bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _is_large_tensor_parallel_model(spec: ModelSpec) -> bool:
    lowered = spec.size_label.lower()
    return any(label in lowered for label in ("32b", "70b", "72b"))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Stateful Agentic Algebra model specs")
    parser.add_argument("--list", action="store_true", help="Print available and unavailable model specs")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.list:
        print(json.dumps({"models": list_models()}, indent=2, sort_keys=True))
        return
    print(json.dumps({"models": list_models()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
