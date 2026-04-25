"""Stateful Agentic Algebra package.

This package is an independent research/benchmark layer on top of AAFLOW.  It
models the paper concepts as import-light Python objects: stateful operator
algebra, KV state objects, KV materialize/transfer/fork/restricted merge/evict,
and metrics for TTFT, transfer cost, recompute cost, throughput, memory, reuse
ratio, and framework overhead Omega.

All heavyweight integrations are lazy and optional.  Missing AAFLOW, vLLM,
SGLang, UCX, NCCL, CUDA, FAISS, or plotting libraries fall back to simulation
paths so the package remains importable in plain Python environments.
"""

from .state_objects import (
    KVBlock,
    KVState,
    KVStateStatus,
    OperatorSpec,
    OperatorType,
    StateCompatibilityError,
    WorkflowState,
)
from .operators import (
    AlgebraOperator,
    EvictOperator,
    ForkOperator,
    MaterializeOperator,
    RestrictedMergeOperator,
    TransferOperator,
)
from .compiler import AlgebraCompiler, ExecutionPlan
from .runtime import RuntimeConfig, StatefulAgenticRuntime
from .metrics_stateful import StatefulMetricsRecorder

__all__ = [
    "AlgebraCompiler",
    "AlgebraOperator",
    "EvictOperator",
    "ExecutionPlan",
    "ForkOperator",
    "KVBlock",
    "KVState",
    "KVStateStatus",
    "MaterializeOperator",
    "OperatorSpec",
    "OperatorType",
    "RestrictedMergeOperator",
    "RuntimeConfig",
    "StatefulAgenticRuntime",
    "StatefulMetricsRecorder",
    "StateCompatibilityError",
    "TransferOperator",
    "WorkflowState",
]
