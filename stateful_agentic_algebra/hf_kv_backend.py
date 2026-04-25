"""HuggingFace KV-cache measurement backend.

This optional backend uses a HuggingFace causal LM with `use_cache=True` to
measure real prefill and decode timing, extract `past_key_values`, and convert
KV tensor metadata into the package `KVState` representation.

Heavy dependencies are imported only when the backend is instantiated or run.
The default model is `distilgpt2`, runs on CPU, and automatically uses CUDA
when `torch.cuda.is_available()` unless a device is specified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .state_objects import KVBlock, KVState, KVStateStatus


@dataclass
class HFBackendConfig:
    """Configuration for HuggingFace KV-cache measurement."""

    model_id: str = "distilgpt2"
    tokenizer_id: Optional[str] = None
    device: str = "auto"
    device_map: str = "auto"
    local_files_only: bool = False
    torch_dtype: Optional[str] = None
    keep_tensors: bool = False
    owner_node: str = "local"
    seed: int = 7


@dataclass
class HFMeasurement:
    """Result from one HF KV-cache measurement run."""

    kv_state: KVState
    metrics: dict[str, Any]
    generated_text: str
    generated_token_ids: list[int] = field(default_factory=list)
    dense_generated_text: str = ""
    dense_generated_token_ids: list[int] = field(default_factory=list)
    outputs_match: Optional[bool] = None


@dataclass
class HFPrefillResult:
    """Output of one full-prefix prefill pass."""

    prompt: str
    input_ids: Any
    attention_mask: Any
    past_key_values: Any
    next_token_id: int
    prefill_sec: float
    ttft_sec: float
    context_tokens: int


@dataclass
class HFDecodeResult:
    """Output of deterministic greedy decode from a KV cache."""

    generated_token_ids: list[int]
    generated_text: str
    decode_sec: float
    total_latency_sec: float
    tokens_per_sec: float
    past_key_values: Any


class HFKVBackend:
    """Measure KV-cache metadata and timings using HuggingFace Transformers."""

    def __init__(self, config: Optional[HFBackendConfig] = None) -> None:
        self.config = config or HFBackendConfig()
        if self.config.tokenizer_id is None:
            self.config.tokenizer_id = self.config.model_id
        self.torch = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.input_device = None
        self._last_prefill_next_token_id: Optional[int] = None

    @staticmethod
    def optional_available() -> bool:
        """Return whether Torch and Transformers can be imported."""

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    def load_model(self, model_id: Optional[str] = None) -> None:
        """Load tokenizer/model lazily.

        CUDA runs use `device_map="auto"` when requested and select bfloat16 or
        float16 by default. CPU fallback is intentionally limited to small GPT-2
        style models so accidental 7B CPU loads fail fast.
        """

        if self.model is not None and self.tokenizer is not None:
            return
        if model_id is not None:
            self.config.model_id = model_id
            if self.config.tokenizer_id is None:
                self.config.tokenizer_id = model_id
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"HuggingFace backend requires torch and transformers: {exc}") from exc

        self.torch = torch
        self._set_seed()
        self.device = self._resolve_device(torch)
        if self.device == "cpu" and not self._is_cpu_safe_model(self.config.model_id):
            raise RuntimeError(
                f"CPU fallback is limited to small smoke-test models; refusing to load {self.config.model_id!r}"
            )
        dtype = self._resolve_dtype(torch)
        model_kwargs: dict[str, Any] = {"local_files_only": self.config.local_files_only}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        use_auto_map = self.device == "cuda" and self.config.device_map == "auto"
        if use_auto_map:
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_id or self.config.model_id,
            local_files_only=self.config.local_files_only,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
        except Exception:
            if not use_auto_map:
                raise
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **fallback_kwargs)
            self.model.to(self.device)
        else:
            if not use_auto_map:
                self.model.to(self.device)
        self.model.eval()
        self.input_device = self._model_input_device()

    def load(self) -> None:
        """Backward-compatible alias for `load_model()`."""

        self.load_model()

    def build_prompt(self, context_tokens: int) -> str:
        """Build a deterministic synthetic prompt with approximately N tokens."""

        self.load_model()
        assert self.tokenizer is not None
        base = (
            "Stateful agentic algebra measures reusable key value cache state "
            "across multi agent workflows. "
        )
        text = base
        while len(self.tokenizer.encode(text, add_special_tokens=False)) < max(1, int(context_tokens)):
            text += base
        ids = self.tokenizer.encode(text, add_special_tokens=False)[: max(1, int(context_tokens))]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def run_prefill(self, prompt: str) -> HFPrefillResult:
        """Run a full-prefix prefill pass with `use_cache=True`."""

        self.load_model()
        assert self.torch is not None
        assert self.tokenizer is not None
        assert self.model is not None
        torch = self.torch

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        self._synchronize()
        with torch.no_grad():
            start = time.perf_counter()
            outputs = self.model(**encoded, use_cache=True)
            self._synchronize()
            prefill_sec = time.perf_counter() - start
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        next_token_id = int(next_token[0, 0].detach().cpu().item())
        self._last_prefill_next_token_id = next_token_id
        return HFPrefillResult(
            prompt=prompt,
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            past_key_values=self._legacy_past(outputs.past_key_values),
            next_token_id=next_token_id,
            prefill_sec=prefill_sec,
            ttft_sec=prefill_sec,
            context_tokens=int(encoded["input_ids"].shape[1]),
        )

    def run_decode_with_cache(
        self,
        past_key_values: Any,
        output_tokens: int,
        next_token_id: Optional[int] = None,
    ) -> HFDecodeResult:
        """Greedily decode a continuation from an existing KV cache."""

        self.load_model()
        assert self.torch is not None
        assert self.tokenizer is not None
        assert self.model is not None
        torch = self.torch

        output_tokens = max(0, int(output_tokens))
        if output_tokens == 0:
            return HFDecodeResult([], "", 0.0, 0.0, 0.0, self._legacy_past(past_key_values))
        first_token_id = next_token_id if next_token_id is not None else self._last_prefill_next_token_id
        if first_token_id is None:
            raise RuntimeError("run_decode_with_cache requires a next_token_id or a prior run_prefill() call")

        past = self._legacy_past(past_key_values)
        next_token = torch.tensor([[int(first_token_id)]], dtype=torch.long, device=self.input_device)
        generated_ids: list[int] = []
        self._synchronize()
        with torch.no_grad():
            start = time.perf_counter()
            for idx in range(output_tokens):
                generated_ids.append(int(next_token[0, 0].detach().cpu().item()))
                if idx == output_tokens - 1:
                    break
                outputs = self.model(input_ids=next_token, past_key_values=past, use_cache=True)
                past = self._legacy_past(outputs.past_key_values)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            self._synchronize()
            decode_sec = time.perf_counter() - start
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return HFDecodeResult(
            generated_token_ids=generated_ids,
            generated_text=text,
            decode_sec=decode_sec,
            total_latency_sec=decode_sec,
            tokens_per_sec=(len(generated_ids) / decode_sec) if decode_sec > 0 else 0.0,
            past_key_values=past,
        )

    def run_dense_prefill_decode(self, prompt: str, output_tokens: int) -> HFDecodeResult:
        """Run deterministic full-prompt prefill plus greedy decode."""

        prefill = self.run_prefill(prompt)
        decode = self.run_decode_with_cache(
            prefill.past_key_values,
            output_tokens=output_tokens,
            next_token_id=prefill.next_token_id,
        )
        total = prefill.prefill_sec + decode.decode_sec
        decode.total_latency_sec = total
        decode.tokens_per_sec = (len(decode.generated_token_ids) / total) if total > 0 else 0.0
        return decode

    def extract_kv_metadata(self, past_key_values: Any) -> KVState:
        """Convert real `past_key_values` into serializable `KVState` metadata."""

        self.load_model()
        return self.past_key_values_to_kv_state(
            past_key_values=past_key_values,
            model_id=self.config.model_id,
            tokenizer_id=self.config.tokenizer_id or self.config.model_id,
            model_config_hash=self._model_config_hash(),
            position_encoding=str(getattr(getattr(self.model, "config", None), "model_type", "unknown")),
            owner_node=self.config.owner_node,
            owner_device=str(self.device),
            keep_tensors=self.config.keep_tensors,
            metadata={"backend": "hf", "status": KVStateStatus.MATERIALIZED.value},
        )

    @staticmethod
    def measure_kv_bytes(past_key_values: Any) -> int:
        """Return total bytes occupied by key/value tensors in a HF cache."""

        total = 0
        for layer in HFKVBackend._legacy_past_static(past_key_values):
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                total += _tensor_nbytes(layer[0]) + _tensor_nbytes(layer[1])
        return int(total)

    def measure(self, prompt: str, context_tokens: int = 128, output_tokens: int = 32) -> HFMeasurement:
        """Run cached and dense deterministic paths and return KV metadata."""

        prompt = self._expand_prompt(prompt, context_tokens)
        prefill = self.run_prefill(prompt)
        cached = self.run_decode_with_cache(
            prefill.past_key_values,
            output_tokens=output_tokens,
            next_token_id=prefill.next_token_id,
        )
        dense = self.run_dense_prefill_decode(prompt, output_tokens=output_tokens)
        kv_state = self.extract_kv_metadata(cached.past_key_values)
        kv_state.metadata.update(
            {
                "actual_context_tokens": prefill.context_tokens,
                "requested_context_tokens": int(context_tokens),
                "generated_tokens": len(cached.generated_token_ids),
                "outputs_match": cached.generated_token_ids == dense.generated_token_ids,
            }
        )
        ttft_sec = prefill.ttft_sec
        total_latency_sec = prefill.prefill_sec + cached.decode_sec
        metrics = {
            "ttft_sec": ttft_sec,
            "total_latency_sec": total_latency_sec,
            "prefill_sec": prefill.prefill_sec,
            "decode_sec": cached.decode_sec,
            "transfer_sec": 0.0,
            "resume_sec": 0.0,
            "omega_sec": 0.0,
            "throughput_tokens_per_sec": (len(cached.generated_token_ids) / total_latency_sec) if total_latency_sec > 0 else 0.0,
            "tokens_per_sec": (len(cached.generated_token_ids) / total_latency_sec) if total_latency_sec > 0 else 0.0,
            "kv_total_bytes": kv_state.total_bytes(),
            "kv_peak_bytes": kv_state.total_bytes(),
            "kv_transferred_bytes": 0,
            "kv_reuse_ratio": 0.0,
            "transfer_count": 0,
            "materialize_count": 1,
            "fork_count": 0,
            "merge_count": 0,
            "evict_count": 0,
            "context_tokens": prefill.context_tokens,
            "output_tokens": len(cached.generated_token_ids),
            "generated_tokens": len(cached.generated_token_ids),
            "dense_total_latency_sec": dense.total_latency_sec,
            "outputs_match": cached.generated_token_ids == dense.generated_token_ids,
        }
        return HFMeasurement(
            kv_state=kv_state,
            metrics=metrics,
            generated_text=cached.generated_text,
            generated_token_ids=cached.generated_token_ids,
            dense_generated_text=dense.generated_text,
            dense_generated_token_ids=dense.generated_token_ids,
            outputs_match=cached.generated_token_ids == dense.generated_token_ids,
        )

    @staticmethod
    def past_key_values_to_kv_state(
        past_key_values: Any,
        model_id: str,
        tokenizer_id: str,
        model_config_hash: str,
        position_encoding: str,
        owner_node: str = "local",
        owner_device: str = "cpu",
        keep_tensors: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KVState:
        """Convert HF `past_key_values` into metadata-only `KVState`."""

        legacy = HFKVBackend._legacy_past_static(past_key_values)
        blocks: list[KVBlock] = []
        for layer_id, layer in enumerate(legacy):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                continue
            key, value = layer[0], layer[1]
            key_shape = tuple(int(x) for x in getattr(key, "shape", ()))
            value_shape = tuple(int(x) for x in getattr(value, "shape", ()))
            token_end = _infer_token_length(key_shape, value_shape)
            nbytes = _tensor_nbytes(key) + _tensor_nbytes(value)
            blocks.append(
                KVBlock(
                    block_id=f"hf_block_{layer_id}_{uuid.uuid4().hex[:8]}",
                    layer_id=layer_id,
                    token_start=0,
                    token_end=token_end,
                    key_shape=key_shape,
                    value_shape=value_shape,
                    dtype=str(getattr(key, "dtype", "unknown")),
                    device=str(getattr(key, "device", owner_device)),
                    nbytes=nbytes,
                    key_ref=key if keep_tensors else None,
                    value_ref=value if keep_tensors else None,
                )
            )

        return KVState(
            state_id=f"hf_kv_{uuid.uuid4().hex[:12]}",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            model_config_hash=model_config_hash,
            position_encoding=position_encoding,
            blocks=blocks,
            lineage=[],
            owner_node=owner_node,
            owner_device=owner_device,
            metadata=dict(metadata or {}),
        )

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.config.torch_dtype is not None:
            return getattr(torch, self.config.torch_dtype)
        if self.device != "cuda":
            return None
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _resolve_device(self, torch: Any) -> str:
        requested = (self.config.device or "auto").lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if requested.startswith("cuda"):
            return "cuda"
        return "cpu"

    def _model_input_device(self) -> Any:
        assert self.torch is not None
        if self.model is None:
            return self.torch.device(self.device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device(self.device)

    def _synchronize(self) -> None:
        if self.torch is not None and self.device == "cuda":
            self.torch.cuda.synchronize()

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        if self.torch is not None:
            self.torch.manual_seed(self.config.seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(self.config.seed)

    @staticmethod
    def _is_cpu_safe_model(model_id: str) -> bool:
        lowered = model_id.lower()
        return any(name in lowered for name in ("gpt2", "tiny", "distil"))

    def _model_config_hash(self) -> str:
        config = getattr(self.model, "config", None)
        if config is None:
            return "unknown"
        try:
            raw = config.to_json_string()
        except Exception:
            raw = repr(config)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _expand_prompt(self, prompt: str, context_tokens: int) -> str:
        if self.tokenizer is not None:
            target = max(1, int(context_tokens))
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) >= target:
                return self.tokenizer.decode(ids[:target], skip_special_tokens=True)
            seed = ids or self.tokenizer.encode("stateful agentic algebra kv cache", add_special_tokens=False)
            while len(ids) < target:
                ids.extend(seed)
            return self.tokenizer.decode(ids[:target], skip_special_tokens=True)
        words = prompt.split()
        if len(words) >= context_tokens:
            return prompt
        seed = words or ["stateful", "agentic", "algebra", "kv", "cache"]
        repeated = []
        while len(words) + len(repeated) < context_tokens:
            repeated.extend(seed)
        return " ".join([*words, *repeated[: max(0, context_tokens - len(words))]])

    def _legacy_past(self, past_key_values: Any) -> Any:
        return self._legacy_past_static(past_key_values)

    @staticmethod
    def _legacy_past_static(past_key_values: Any) -> Any:
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return past_key_values


def _infer_token_length(key_shape: tuple[int, ...], value_shape: tuple[int, ...]) -> int:
    shape = key_shape or value_shape
    if not shape:
        return 0
    if len(shape) >= 3:
        return int(shape[-2])
    return int(shape[-1])


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if callable(nbytes):
        try:
            return int(nbytes())
        except Exception:
            pass
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure HuggingFace KV-cache metadata and timing")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default=None, help="Optional dtype name, e.g. bfloat16 or float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/stateful/hf_real_gpt2")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = HFKVBackend(
        HFBackendConfig(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id or args.model_id,
            device=args.device,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            seed=args.seed,
        )
    )
    prompt = backend.build_prompt(args.context_tokens)
    measurement = backend.measure(prompt, context_tokens=args.context_tokens, output_tokens=args.output_tokens)
    metrics = dict(measurement.metrics)
    metrics.update(
        {
            "model_id": args.model_id,
            "tokenizer_id": args.tokenizer_id or args.model_id,
            "device": backend.device,
            "context_tokens_requested": args.context_tokens,
            "output_tokens_requested": args.output_tokens,
            "seed": args.seed,
            "outputs_match": measurement.outputs_match,
        }
    )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "kv_metadata.json").write_text(
        json.dumps(measurement.kv_state.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "generated_dense.txt").write_text(measurement.dense_generated_text, encoding="utf-8")
    (output_dir / "generated_cached.txt").write_text(measurement.generated_text, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
