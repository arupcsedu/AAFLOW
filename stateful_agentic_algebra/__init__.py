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
_HF_BACKEND_EXPORTS = {"HFBackendConfig", "HFDecodeResult", "HFMeasurement", "HFPrefillResult", "HFKVBackend"}
_VLLM_BACKEND_EXPORTS = {"VLLMBackend", "VLLMBackendConfig", "VLLMRunResult"}
_VLLM_BENCHMARK_EXPORTS = {
    "check_vllm_available",
    "launch_vllm_server",
    "parse_vllm_results",
    "run_vllm_bench_serve",
    "wait_for_server",
}
_MODEL_REGISTRY_EXPORTS = {"ModelSpec", "default_model_registry", "get_model_spec", "list_models", "model_availability"}


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
    if name in _HF_BACKEND_EXPORTS:
        from . import hf_kv_backend

        value = getattr(hf_kv_backend, name)
        globals()[name] = value
        return value
    if name in _VLLM_BACKEND_EXPORTS:
        from . import vllm_backend

        value = getattr(vllm_backend, name)
        globals()[name] = value
        return value
    if name in _VLLM_BENCHMARK_EXPORTS:
        from . import vllm_benchmark

        value = getattr(vllm_benchmark, name)
        globals()[name] = value
        return value
    if name in _MODEL_REGISTRY_EXPORTS:
        from . import model_registry

        value = getattr(model_registry, name)
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
    "HFBackendConfig",
    "HFDecodeResult",
    "HFMeasurement",
    "HFPrefillResult",
    "HFKVBackend",
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
    "ModelSpec",
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
    "VLLMBackend",
    "VLLMBackendConfig",
    "VLLMRunResult",
    "WorkloadConfig",
    "WorkflowState",
    "aggregate_runs",
    "check_vllm_available",
    "default_model_registry",
    "export_in_aaflow_style",
    "get_model_spec",
    "get_baselines",
    "list_baselines",
    "list_models",
    "load_existing_metrics",
    "model_availability",
    "launch_vllm_server",
    "parse_vllm_results",
    "run_existing_rag_agent_if_available",
    "run_vllm_bench_serve",
    "wait_for_server",
]
