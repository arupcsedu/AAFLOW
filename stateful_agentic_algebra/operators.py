"""Algebra operators for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: `AlgebraOperator` is the executable node.
  - KV materialize / transfer / fork / restricted merge / evict: each paper
    primitive is represented by a concrete operator class.
  - Metrics: operators record TTFT, transfer cost, recompute cost, memory,
    reuse ratio, throughput, and framework overhead Omega through the runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .state_objects import OperatorSpec, OperatorType, WorkflowState


def _resolve_kv_id(state: WorkflowState, token: str) -> str:
    """Resolve either a raw KV id or an operator output name to a KV id."""

    if token in state.kv_states:
        return token
    mapped = state.values.get(f"{token}.kv_state_id")
    if isinstance(mapped, str):
        return mapped
    return token


@dataclass
class OperatorResult:
    """Output from an operator execution."""

    name: str
    value: Any = None
    kv_state_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class AlgebraOperator:
    """Base executable operator."""

    op_type = OperatorType.NOOP

    def __init__(self, spec: OperatorSpec) -> None:
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
    """Materialize a new KV state, usually from prompt prefill."""

    op_type = OperatorType.MATERIALIZE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        tokens = int(self.spec.params.get("tokens", 0))
        recompute_ms = float(self.spec.params.get("recompute_cost_ms", tokens * 0.02))
        kv = runtime.kv.materialize(tokens=tokens, recompute_cost_ms=recompute_ms, metadata=self.spec.params)
        state.kv_states[kv.state_id] = kv
        return OperatorResult(self.spec.name, kv_state_id=kv.state_id, metrics={"recompute_cost_ms": recompute_ms})


class TransferOperator(AlgebraOperator):
    """Transfer a KV state to another worker/device."""

    op_type = OperatorType.TRANSFER

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        target = str(self.spec.params.get("target", "remote"))
        source = runtime.kv.get(source_id)
        transfer = runtime.transport.transfer(source.owner, target, source.bytes_size)
        kv = runtime.kv.transfer(source_id, target, transfer.transfer_cost_ms)
        state.kv_states[kv.state_id] = kv
        return OperatorResult(
            self.spec.name,
            value=transfer,
            kv_state_id=kv.state_id,
            metrics={"transfer_cost_ms": transfer.transfer_cost_ms},
        )


class ForkOperator(AlgebraOperator):
    """Fork an existing KV state for branch execution."""

    op_type = OperatorType.FORK

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        kv = runtime.kv.fork(source_id, owner=self.spec.params.get("owner"))
        state.kv_states[kv.state_id] = kv
        return OperatorResult(self.spec.name, kv_state_id=kv.state_id)


class RestrictedMergeOperator(AlgebraOperator):
    """Merge compatible KV branches under a restriction such as token limit."""

    op_type = OperatorType.RESTRICTED_MERGE

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        token_limit = self.spec.params.get("token_limit")
        kv = runtime.kv.restricted_merge(
            [_resolve_kv_id(state, item) for item in self.spec.inputs],
            token_limit=int(token_limit) if token_limit is not None else None,
        )
        state.kv_states[kv.state_id] = kv
        return OperatorResult(self.spec.name, kv_state_id=kv.state_id)


class EvictOperator(AlgebraOperator):
    """Evict a KV state from live memory."""

    op_type = OperatorType.EVICT

    def _execute(self, runtime: Any, state: WorkflowState) -> OperatorResult:
        source_id = _resolve_kv_id(state, self.spec.inputs[0])
        kv = runtime.kv.evict(source_id)
        return OperatorResult(self.spec.name, kv_state_id=kv.state_id)


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
