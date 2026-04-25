"""Stateful operators for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: `StatefulOperator` defines the common operator
    contract: name, input schema, output schema, state policy, communication
    pattern, and `execute()`.
  - KV state object: KV operators consume and produce `KVState` objects.
  - KV materialize / transfer / fork / restricted merge / evict: implemented
    as `KVMaterializeOperator`, `KVTransferOperator`, `KVForkOperator`,
    `KVMergeOperator`, and `KVEvictOperator`.
  - Metrics: transfer and eviction records carry latency/byte accounting;
    runtime adapters can additionally record TTFT, recompute cost, throughput,
    memory, reuse ratio, and framework overhead Omega.

The implementation is mock-first. It does not require real KV tensors. Block
references are preserved across fork/merge initially so future HuggingFace,
vLLM, or SGLang integrations can attach actual buffers without changing the
metadata API.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .state_objects import (
    KVBlock,
    KVState,
    KVStateStatus,
    OperatorSpec,
    OperatorType,
    StateCompatibilityError,
    WorkflowState,
)
from .transport import Transport


DTYPE_SIZES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "int8": 1,
    "uint8": 1,
}

RESTRICTED_MERGE_POLICIES = {"prefix_compatible", "segment_concat", "summary_reduce"}


@dataclass
class OperatorResult:
    """Output from an operator execution."""

    name: str
    value: Any = None
    kv_state_id: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvictionRecord:
    """Record returned by `KVEvictOperator`."""

    state_id: str
    bytes_evicted: int
    owner_node: str
    owner_device: str
    evicted_at: float
    reason: str = "manual"


@dataclass
class MergeSummary:
    """Summary object returned by `summary_reduce` merge policy."""

    state_ids: list[str]
    merge_policy: str
    total_bytes: int
    token_span: tuple[int, int]
    summary: str


class StatefulOperator:
    """Base class for stateful algebra operators."""

    def __init__(
        self,
        name: str,
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        state_policy: str = "stateless",
        communication_pattern: str = "local",
    ) -> None:
        self.name = name
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.state_policy = state_policy
        self.communication_pattern = communication_pattern

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the operator."""

        raise NotImplementedError


class KVMaterializeOperator(StatefulOperator):
    """Materialize a synthetic or future real KV state."""

    def __init__(
        self,
        name: str = "kv_materialize",
        layer_count: int = 1,
        head_count: int = 1,
        hidden_size: int = 128,
        dtype: str = "float16",
        device: str = "cpu",
        position_encoding: str = "rope",
        owner_node: str = "local",
        owner_device: Optional[str] = None,
        backend: str = "mock",
    ) -> None:
        super().__init__(
            name=name,
            input_schema={"prompt_text": "str?", "token_count": "int?", "model_id": "str", "tokenizer_id": "str"},
            output_schema={"kv_state": "KVState"},
            state_policy="materialize",
            communication_pattern="local",
        )
        self.layer_count = layer_count
        self.head_count = head_count
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.position_encoding = position_encoding
        self.owner_node = owner_node
        self.owner_device = owner_device or device
        self.backend = backend

    def execute(
        self,
        prompt_text: Optional[str] = None,
        token_count: Optional[int] = None,
        model_id: str = "mock-model",
        tokenizer_id: str = "mock-tokenizer",
        model_config_hash: str = "mock-config",
        real_backend: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KVState:
        """Create a KV state.

        In mock mode, token count controls synthetic block sizes. In real mode,
        callers may pass a future backend object; this first implementation
        still returns metadata-only state when no backend protocol is provided.
        """

        tokens = self._resolve_token_count(prompt_text, token_count)
        if real_backend is not None and hasattr(real_backend, "materialize_kv"):
            candidate = real_backend.materialize_kv(
                prompt_text=prompt_text,
                token_count=tokens,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
            )
            if isinstance(candidate, KVState):
                return candidate

        blocks = self._mock_blocks(tokens)
        return KVState(
            state_id=f"kv_{uuid.uuid4().hex[:12]}",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            model_config_hash=model_config_hash,
            position_encoding=self.position_encoding,
            blocks=blocks,
            lineage=[],
            owner_node=self.owner_node,
            owner_device=self.owner_device,
            metadata={
                **(metadata or {}),
                "backend": self.backend,
                "status": KVStateStatus.MATERIALIZED.value,
                "token_count": tokens,
                "layer_count": self.layer_count,
                "head_count": self.head_count,
                "hidden_size": self.hidden_size,
                "dtype": self.dtype,
            },
        )

    def _resolve_token_count(self, prompt_text: Optional[str], token_count: Optional[int]) -> int:
        if token_count is not None:
            return max(0, int(token_count))
        if not prompt_text:
            return 0
        return len(prompt_text.split())

    def _mock_blocks(self, token_count: int) -> list[KVBlock]:
        dtype_size = DTYPE_SIZES.get(self.dtype.lower(), 2)
        head_dim = max(1, self.hidden_size // max(1, self.head_count))
        key_shape = (self.head_count, token_count, head_dim)
        value_shape = (self.head_count, token_count, head_dim)
        # Per layer: key tensor + value tensor.
        bytes_per_block = token_count * self.hidden_size * dtype_size * 2
        return [
            KVBlock(
                block_id=f"block_{uuid.uuid4().hex[:12]}",
                layer_id=layer_id,
                token_start=0,
                token_end=token_count,
                key_shape=key_shape,
                value_shape=value_shape,
                dtype=self.dtype,
                device=self.device,
                nbytes=bytes_per_block,
                key_ref=None,
                value_ref=None,
            )
            for layer_id in range(max(0, self.layer_count))
        ]


class KVTransferOperator(StatefulOperator):
    """Transfer KV placement metadata through the transport interface."""

    def __init__(self, name: str = "kv_transfer", transport: Optional[Any] = None) -> None:
        super().__init__(
            name=name,
            input_schema={"state": "KVState", "source_node": "str", "target_node": "str"},
            output_schema={"kv_state": "KVState"},
            state_policy="transfer",
            communication_pattern="point_to_point",
        )
        self.transport = transport or Transport()

    def execute(self, state: KVState, source_node: str, target_node: str) -> KVState:
        start = time.perf_counter()
        transfer = self.transport.transfer(source_node, target_node, state.total_bytes())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        transfer_ms = getattr(transfer, "transfer_cost_ms", elapsed_ms)
        moved = state.fork(f"kv_{uuid.uuid4().hex[:12]}")
        moved.owner_node = target_node
        moved.metadata.update(
            {
                "status": KVStateStatus.TRANSFERRED.value,
                "source_node": source_node,
                "target_node": target_node,
                "transfer_latency_ms": transfer_ms,
                "transfer_bytes": state.total_bytes(),
                "transport_backend": getattr(transfer, "backend", getattr(self.transport, "name", "unknown")),
            }
        )
        return moved


class KVForkOperator(StatefulOperator):
    """Fork KV state into multiple branch states."""

    def __init__(self, name: str = "kv_fork") -> None:
        super().__init__(
            name=name,
            input_schema={"state": "KVState", "branch_count": "int"},
            output_schema={"states": "list[KVState]"},
            state_policy="fork",
            communication_pattern="local",
        )

    def execute(self, state: KVState, branch_count: int) -> list[KVState]:
        branches = []
        for branch_idx in range(max(0, int(branch_count))):
            child = state.fork(f"{state.state_id}_branch_{branch_idx}_{uuid.uuid4().hex[:8]}")
            child.metadata.update({"branch_index": branch_idx, "status": KVStateStatus.MATERIALIZED.value})
            branches.append(child)
        return branches


class KVMergeOperator(StatefulOperator):
    """Merge KV states with restricted merge policies only."""

    def __init__(self, name: str = "kv_merge") -> None:
        super().__init__(
            name=name,
            input_schema={"states": "list[KVState]", "merge_policy": "str"},
            output_schema={"kv_state": "KVState | MergeSummary"},
            state_policy="restricted_merge",
            communication_pattern="local",
        )

    def execute(self, states: list[KVState], merge_policy: str = "prefix_compatible") -> KVState | MergeSummary:
        if merge_policy not in RESTRICTED_MERGE_POLICIES:
            raise StateCompatibilityError(f"Unsupported restricted merge policy: {merge_policy}")
        if not states:
            raise StateCompatibilityError("Cannot merge an empty state list")
        self._check_compatibility(states)

        if merge_policy == "summary_reduce":
            return self._summary_reduce(states, merge_policy)
        if merge_policy == "prefix_compatible":
            self._check_prefix_compatible(states)
            blocks = list(states[0].blocks)
        else:
            blocks = []
            for state in states:
                blocks.extend(state.blocks)

        first = states[0]
        merged = KVState(
            state_id=f"kv_{uuid.uuid4().hex[:12]}",
            model_id=first.model_id,
            tokenizer_id=first.tokenizer_id,
            model_config_hash=first.model_config_hash,
            position_encoding=first.position_encoding,
            blocks=blocks,
            lineage=[state.state_id for state in states],
            owner_node=first.owner_node,
            owner_device=first.owner_device,
            metadata={"status": KVStateStatus.MERGED.value, "merge_policy": merge_policy},
        )
        return merged

    def _check_compatibility(self, states: list[KVState]) -> None:
        first = states[0]
        for other in states[1:]:
            if not first.is_compatible(other):
                raise StateCompatibilityError(
                    f"Incompatible KV states: {first.state_id} and {other.state_id}"
                )

    def _check_prefix_compatible(self, states: list[KVState]) -> None:
        base_span = states[0].token_span()
        for state in states[1:]:
            if state.token_span() != base_span:
                raise StateCompatibilityError(
                    f"Prefix-incompatible token spans: {base_span} and {state.token_span()}"
                )

    def _summary_reduce(self, states: list[KVState], merge_policy: str) -> MergeSummary:
        starts = [state.token_span()[0] for state in states]
        ends = [state.token_span()[1] for state in states]
        return MergeSummary(
            state_ids=[state.state_id for state in states],
            merge_policy=merge_policy,
            total_bytes=sum(state.total_bytes() for state in states),
            token_span=(min(starts), max(ends)),
            summary=f"summary_reduce({len(states)} states)",
        )


class KVEvictOperator(StatefulOperator):
    """Evict a KV state and return a durable eviction record."""

    def __init__(self, name: str = "kv_evict") -> None:
        super().__init__(
            name=name,
            input_schema={"state": "KVState"},
            output_schema={"eviction_record": "EvictionRecord"},
            state_policy="evict",
            communication_pattern="local",
        )

    def execute(self, state: KVState, reason: str = "manual") -> EvictionRecord:
        state.metadata["status"] = KVStateStatus.EVICTED.value
        record = EvictionRecord(
            state_id=state.state_id,
            bytes_evicted=state.total_bytes(),
            owner_node=state.owner_node,
            owner_device=state.owner_device,
            evicted_at=time.time(),
            reason=reason,
        )
        state.metadata["eviction_record"] = {
            "bytes_evicted": record.bytes_evicted,
            "evicted_at": record.evicted_at,
            "reason": reason,
        }
        return record


def _resolve_kv_id(state: WorkflowState, token: str) -> str:
    """Resolve either a raw KV id or an operator output name to a KV id."""

    if token in state.kv_states:
        return token
    mapped = state.values.get(f"{token}.kv_state_id")
    if isinstance(mapped, str):
        return mapped
    return token


class AlgebraOperator(StatefulOperator):
    """Compatibility adapter used by the package compiler/runtime."""

    op_type = OperatorType.NOOP

    def __init__(self, spec: OperatorSpec) -> None:
        super().__init__(
            name=spec.name,
            input_schema={},
            output_schema={},
            state_policy=spec.op_type.value,
            communication_pattern="runtime",
        )
        self.spec = spec

    def execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        start = time.perf_counter()
        state.add_trace("operator_start", name=self.spec.name, op_type=self.spec.op_type.value)
        result = self._execute(runtime, state)
        omega_ms = (time.perf_counter() - start) * 1000.0
        runtime.metrics.observe("framework_overhead_ms", omega_ms)
        runtime.metrics.mark_completed_op()
        runtime.metrics.update_live_memory(runtime.kv.live_bytes())
        result.metrics.setdefault("framework_overhead_ms", omega_ms)
        if result.value is not None:
            state.values[self.spec.name] = result.value
        if result.kv_state_id is not None:
            state.values[f"{self.spec.name}.kv_state_id"] = result.kv_state_id
        state.add_trace("operator_end", name=self.spec.name, result=result.kv_state_id)
        return result

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        return OperatorResult(name=self.spec.name)


class MaterializeOperator(AlgebraOperator):
    """Compatibility adapter for materialize specs."""

    op_type = OperatorType.MATERIALIZE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        operator = KVMaterializeOperator(
            name=self.spec.name,
            layer_count=int(self.spec.params.get("layer_count", 1)),
            head_count=int(self.spec.params.get("head_count", 1)),
            hidden_size=int(self.spec.params.get("hidden_size", runtime.config.bytes_per_token // 4)),
            dtype=str(self.spec.params.get("dtype", "float16")),
            device=str(self.spec.params.get("device", "cpu")),
            owner_node=str(self.spec.params.get("owner_node", "local")),
        )
        kv = operator.execute(
            prompt_text=self.spec.params.get("prompt_text"),
            token_count=int(self.spec.params.get("tokens", self.spec.params.get("token_count", 0))),
            model_id=str(self.spec.params.get("model_id", "mock-model")),
            tokenizer_id=str(self.spec.params.get("tokenizer_id", "mock-tokenizer")),
            model_config_hash=str(self.spec.params.get("model_config_hash", "mock-config")),
            metadata=self.spec.params,
        )
        runtime.kv.states[kv.state_id] = kv
        runtime.metrics.mark_kv_materialized()
        runtime.metrics.observe("recompute_cost_ms", float(self.spec.params.get("recompute_cost_ms", 0.0)))
        runtime.metrics.update_live_memory(runtime.kv.live_bytes())
        state.kv_states[kv.state_id] = kv
        return OperatorResult(self.spec.name, kv_state_id=kv.state_id)


class TransferOperator(AlgebraOperator):
    """Compatibility adapter for transfer specs."""

    op_type = OperatorType.TRANSFER

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        source = runtime.kv.get(source_id)
        target = str(self.spec.params.get("target", self.spec.params.get("target_node", "remote")))
        kv = KVTransferOperator(name=self.spec.name, transport=runtime.transport).execute(
            source,
            source_node=str(self.spec.params.get("source_node", source.owner_node)),
            target_node=target,
        )
        runtime.kv.states[kv.state_id] = kv
        state.kv_states[kv.state_id] = kv
        runtime.metrics.mark_kv_reuse()
        runtime.metrics.observe("transfer_cost_ms", float(kv.metadata.get("transfer_latency_ms", 0.0)))
        return OperatorResult(
            self.spec.name,
            kv_state_id=kv.state_id,
            metrics={"transfer_cost_ms": float(kv.metadata.get("transfer_latency_ms", 0.0))},
        )


class ForkOperator(AlgebraOperator):
    """Compatibility adapter for fork specs."""

    op_type = OperatorType.FORK

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        source = runtime.kv.get(source_id)
        branch = KVForkOperator(name=self.spec.name).execute(source, branch_count=1)[0]
        runtime.kv.states[branch.state_id] = branch
        state.kv_states[branch.state_id] = branch
        runtime.metrics.mark_kv_reuse()
        return OperatorResult(self.spec.name, kv_state_id=branch.state_id)


class RestrictedMergeOperator(AlgebraOperator):
    """Compatibility adapter for restricted merge specs."""

    op_type = OperatorType.RESTRICTED_MERGE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        states = [runtime.kv.get(_resolve_kv_id(state, item)) for item in self.spec.inputs]
        policy = str(self.spec.params.get("merge_policy", "segment_concat"))
        merged = KVMergeOperator(name=self.spec.name).execute(states, merge_policy=policy)
        if isinstance(merged, KVState):
            runtime.kv.states[merged.state_id] = merged
            state.kv_states[merged.state_id] = merged
            runtime.metrics.mark_kv_reuse()
            return OperatorResult(self.spec.name, kv_state_id=merged.state_id)
        state.values[self.spec.name] = merged
        return OperatorResult(self.spec.name, value=merged)


class EvictOperator(AlgebraOperator):
    """Compatibility adapter for evict specs."""

    op_type = OperatorType.EVICT

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        source = runtime.kv.get(source_id)
        record = KVEvictOperator(name=self.spec.name).execute(source)
        return OperatorResult(self.spec.name, value=record, kv_state_id=source.state_id)


class GenerateOperator(AlgebraOperator):
    """Generate text using reused or newly materialized KV state."""

    op_type = OperatorType.GENERATE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        prompt = str(self.spec.params.get("prompt", state.values.get("prompt", "")))
        kv_state_id = _resolve_kv_id(state, self.spec.inputs[0]) if self.spec.inputs else None
        with runtime.metrics.span("ttft_ms"):
            answer = runtime.generate(prompt, kv_state_id=kv_state_id)
        state.values[self.spec.name] = answer
        return OperatorResult(self.spec.name, value=answer, kv_state_id=kv_state_id)


class RetrieveOperator(AlgebraOperator):
    """Retrieve context from AAFLOW components when available or mock search."""

    op_type = OperatorType.RETRIEVE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        query = str(self.spec.params.get("query", state.values.get("query", "")))
        results = runtime.retrieve(query)
        state.values[self.spec.name] = results
        return OperatorResult(self.spec.name, value=results)


OPERATOR_REGISTRY = {
    OperatorType.EVICT: EvictOperator,
    OperatorType.FORK: ForkOperator,
    OperatorType.GENERATE: GenerateOperator,
    OperatorType.MATERIALIZE: MaterializeOperator,
    OperatorType.RESTRICTED_MERGE: RestrictedMergeOperator,
    OperatorType.RETRIEVE: RetrieveOperator,
    OperatorType.TRANSFER: TransferOperator,
}


def build_operator(spec: OperatorSpec) -> AlgebraOperator:
    """Construct an executable operator from a spec."""

    return OPERATOR_REGISTRY.get(spec.op_type, AlgebraOperator)(spec)
