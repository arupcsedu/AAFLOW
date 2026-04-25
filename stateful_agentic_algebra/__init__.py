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
    EvictionRecord,
    ForkOperator,
    KVForkOperator,
    KVEvictOperator,
    KVMergeOperator,
    KVMaterializeOperator,
    KVTransferOperator,
    MergeSummary,
    MaterializeOperator,
    RestrictedMergeOperator,
    StatefulOperator,
    TransferOperator,
)
from .compiler import (
    AlgebraCompiler,
    EdgeSpec,
    ExecutionPlan,
    NodeSpec,
    StatefulCompiler,
    StatefulExecutionGraph,
)
from .scheduler import CostModel, ScheduleDecision, StateAwareScheduler
from .transport import (
    AbstractTransport,
    LocalFileTransport,
    MockTransport,
    TransferMetrics,
    TransferResult,
    Transport,
    TransportConfig,
    UCXTransport,
)
from .runtime import RuntimeConfig, RuntimeMetrics, StatefulAgenticRuntime, StatefulRuntime
from .metrics_stateful import METRIC_FIELDS, MetricEvent, StatefulMetricsRecorder, aggregate_runs
_BASELINE_EXPORTS = {
    "AAFLOWTextBaseline",
    "BaselineAdapter",
    "BaselineResult",
    "DensePrefillBaseline",
    "DistServeStyleBaseline",
    "SGLangPrefixBaseline",
    "VLLMLocalPrefixBaseline",
    "get_baselines",
    "list_baselines",
}
_WORKLOAD_EXPORTS = {"GeneratedWorkload", "QueryWorkload", "WorkloadConfig"}
_AAFLOW_ADAPTER_EXPORTS = {
    "export_in_aaflow_style",
    "load_existing_metrics",
    "run_existing_rag_agent_if_available",
}


def __getattr__(name: str):
    """Lazily expose baseline/workload helpers without eager optional imports."""

    if name in _BASELINE_EXPORTS:
        from . import baselines

        value = getattr(baselines, name)
        globals()[name] = value
        return value
    if name in _WORKLOAD_EXPORTS:
        from . import workloads

        value = getattr(workloads, name)
        globals()[name] = value
        return value
    if name in _AAFLOW_ADAPTER_EXPORTS:
        from . import aaflow_adapter

        value = getattr(aaflow_adapter, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AlgebraCompiler",
    "AlgebraOperator",
    "AbstractTransport",
    "AAFLOWTextBaseline",
    "BaselineAdapter",
    "BaselineResult",
    "CostModel",
    "DensePrefillBaseline",
    "DistServeStyleBaseline",
    "EdgeSpec",
    "EvictOperator",
    "EvictionRecord",
    "ExecutionPlan",
    "ForkOperator",
    "GeneratedWorkload",
    "KVBlock",
    "KVForkOperator",
    "KVEvictOperator",
    "KVMergeOperator",
    "KVMaterializeOperator",
    "KVState",
    "KVStateStatus",
    "KVTransferOperator",
    "LocalFileTransport",
    "METRIC_FIELDS",
    "MaterializeOperator",
    "MergeSummary",
    "MetricEvent",
    "MockTransport",
    "NodeSpec",
    "OperatorSpec",
    "OperatorType",
    "QueryWorkload",
    "RestrictedMergeOperator",
    "RuntimeConfig",
    "RuntimeMetrics",
    "SGLangPrefixBaseline",
    "ScheduleDecision",
    "StatefulAgenticRuntime",
    "StatefulRuntime",
    "StateAwareScheduler",
    "StatefulMetricsRecorder",
    "StateCompatibilityError",
    "StatefulCompiler",
    "StatefulExecutionGraph",
    "StatefulOperator",
    "TransferOperator",
    "TransferMetrics",
    "TransferResult",
    "Transport",
    "TransportConfig",
    "UCXTransport",
    "VLLMLocalPrefixBaseline",
    "WorkloadConfig",
    "WorkflowState",
    "aggregate_runs",
    "export_in_aaflow_style",
    "get_baselines",
    "list_baselines",
    "load_existing_metrics",
    "run_existing_rag_agent_if_available",
]
