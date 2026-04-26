"""Optional vLLM backend adapter.

This module is intentionally thin and lazy. vLLM is imported only inside
`VLLMBackend`, so ordinary stateful tests and mock/HF experiments do not fail
when vLLM is not installed.

The current adapter measures request latency, best-effort TTFT if vLLM exposes
it in public metrics, output token counts, and throughput. KV export/import is
left as explicit placeholders because vLLM does not expose a stable public API
for portable KV cache materialization across releases.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VLLMBackendConfig:
    """Configuration for the optional vLLM adapter."""

    model_id: str = "gpt2"
    tokenizer_id: Optional[str] = None
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    dtype: str = "auto"
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLLMRunResult:
    """Result from a vLLM prompt batch."""

    texts: list[str]
    metrics: dict[str, Any]
    raw_outputs: Any = None


class VLLMBackend:
    """Minimal vLLM generation and measurement adapter."""

    def __init__(self, config: Optional[VLLMBackendConfig] = None) -> None:
        self.config = config or VLLMBackendConfig()
        self.llm = None
        self.sampling_params = None
        self._vllm = None

    @staticmethod
    def available() -> bool:
        """Return False when vLLM is not installed or cannot be imported."""

        if importlib.util.find_spec("vllm") is None:
            return False
        try:
            import vllm  # noqa: F401  # type: ignore
        except Exception:
            return False
        return True

    def load(self) -> None:
        """Instantiate the vLLM engine lazily."""

        if self.llm is not None:
            return
        if not self.available():
            raise RuntimeError("vLLM is not installed")
        try:
            import vllm  # type: ignore
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Could not import vLLM: {exc}") from exc

        self._vllm = vllm
        engine_kwargs = {
            "model": self.config.model_id,
            "tokenizer": self.config.tokenizer_id or self.config.model_id,
            "trust_remote_code": self.config.trust_remote_code,
            "dtype": self.config.dtype,
            **self.config.extra_engine_kwargs,
        }
        self.llm = LLM(**engine_kwargs)
        self.sampling_params = SamplingParams(
            max_tokens=max(0, int(self.config.max_tokens)),
            temperature=float(self.config.temperature),
            top_p=float(self.config.top_p),
        )

    def run_prompts(self, prompts: list[str] | str, max_tokens: Optional[int] = None) -> VLLMRunResult:
        """Run prompts and return normalized latency/token metrics."""

        self.load()
        assert self.llm is not None
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        sampling_params = self.sampling_params
        if max_tokens is not None:
            from vllm import SamplingParams  # type: ignore

            sampling_params = SamplingParams(
                max_tokens=max(0, int(max_tokens)),
                temperature=float(self.config.temperature),
                top_p=float(self.config.top_p),
            )

        start = time.perf_counter()
        outputs = self.llm.generate(prompt_list, sampling_params)
        latency_sec = time.perf_counter() - start
        texts = [_output_text(output) for output in outputs]
        output_tokens = sum(_output_token_count(output) for output in outputs)
        ttft_sec = _extract_ttft(outputs)
        throughput = output_tokens / latency_sec if latency_sec > 0 else 0.0

        return VLLMRunResult(
            texts=texts,
            raw_outputs=outputs,
            metrics={
                "request_latency_sec": latency_sec,
                "total_latency_sec": latency_sec,
                "ttft_sec": ttft_sec,
                "total_output_tokens": output_tokens,
                "output_tokens": output_tokens,
                "throughput_tokens_per_sec": throughput,
                "num_prompts": len(prompt_list),
                "backend": "vllm",
            },
        )

    def export_kv_state(self, *_: Any, **__: Any) -> Any:
        """Placeholder for future stable vLLM KV export API."""

        raise NotImplementedError(
            "vLLM KV export is not implemented because vLLM does not expose a stable public KV cache export API."
        )

    def import_kv_state(self, *_: Any, **__: Any) -> Any:
        """Placeholder for future stable vLLM KV import API."""

        raise NotImplementedError(
            "vLLM KV import is not implemented because vLLM does not expose a stable public KV cache import API."
        )


def _output_text(output: Any) -> str:
    candidates = getattr(output, "outputs", None) or []
    if not candidates:
        return ""
    return str(getattr(candidates[0], "text", ""))


def _output_token_count(output: Any) -> int:
    total = 0
    for item in getattr(output, "outputs", None) or []:
        token_ids = getattr(item, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
        else:
            text = getattr(item, "text", "")
            total += len(str(text).split())
    return total


def _extract_ttft(outputs: Any) -> Optional[float]:
    values = []
    for output in outputs or []:
        metrics = getattr(output, "metrics", None)
        for attr in ("time_to_first_token", "ttft", "first_token_time"):
            value = getattr(metrics, attr, None) if metrics is not None else None
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
    if not values:
        return None
    return sum(values) / len(values)
