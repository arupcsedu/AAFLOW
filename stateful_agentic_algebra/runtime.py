"""Runtime facade for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: executes compiled operators over a runtime.
  - KV state object and lifecycle operations: exposes `KVStateManager`.
  - KV materialize / transfer / fork / restricted merge / evict: implemented
    through `kv_manager` and `transport`.
  - Metrics: records TTFT, transfer cost, recompute cost, throughput, memory,
    reuse ratio, and framework overhead Omega.

The runtime is independent from AAFLOW.  When AAFLOW modules are importable, it
can wrap them lazily; otherwise it uses deterministic mock retrieval and mock
generation.  Optional vLLM, SGLang, CUDA, UCX, and NCCL paths are detected but
not required.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .compiler import AlgebraCompiler, ExecutionPlan
from .kv_manager import KVManagerConfig, KVStateManager
from .metrics_stateful import StatefulMetricsRecorder
from .scheduler import AlgebraScheduler
from .state_objects import OperatorSpec, WorkflowState
from .transport import Transport, TransportConfig, optional_available


@dataclass
class RuntimeConfig:
    """Runtime configuration with mock-safe defaults."""

    prefer_aaflow: bool = False
    generation_backend: str = "mock"
    transport_backend: str = "auto"
    bytes_per_token: int = 2048
    mock_tokens_per_answer: int = 32
    optional_backends: Dict[str, bool] = field(default_factory=dict)


class StatefulAgenticRuntime:
    """Execute stateful algebra plans with optional AAFLOW reuse."""

    def __init__(self, config: Optional[RuntimeConfig] = None, aaflow_agent: Any = None) -> None:
        self.config = config or RuntimeConfig()
        self.metrics = StatefulMetricsRecorder()
        self.kv = KVStateManager(KVManagerConfig(bytes_per_token=self.config.bytes_per_token), self.metrics)
        self.transport = Transport(TransportConfig(backend=self.config.transport_backend))
        self.compiler = AlgebraCompiler()
        self.scheduler = AlgebraScheduler()
        self.aaflow_agent = aaflow_agent
        self.available_backends = self._detect_optional_backends()
        self.config.optional_backends = dict(self.available_backends)
        if (
            self.aaflow_agent is None
            and (self.config.prefer_aaflow or self.config.generation_backend == "aaflow")
        ):
            self.aaflow_agent = self._try_import_aaflow_agent()

    def _detect_optional_backends(self) -> Dict[str, bool]:
        return {
            "vllm": optional_available("vllm"),
            "sglang": optional_available("sglang"),
            "ucx": optional_available("ucp"),
            "nccl": optional_available("nccl"),
            "cuda_torch": self._torch_cuda_available(),
        }

    def _torch_cuda_available(self) -> bool:
        try:
            torch = importlib.import_module("torch")
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _try_import_aaflow_agent(self) -> Any:
        if importlib.util.find_spec("agents") is None:
            return None
        # AAFLOW objects are intentionally not constructed here because the
        # root modules may import heavy HF dependencies. Callers can pass a
        # pre-built RagAgent to reuse the existing pipeline.
        return None

    def compile(self, specs: List[OperatorSpec]) -> ExecutionPlan:
        return self.compiler.compile(specs)

    def run(self, specs: List[OperatorSpec], state: Optional[WorkflowState] = None) -> Dict[str, Any]:
        plan = self.compile(specs)
        workflow = state or WorkflowState()
        results = self.scheduler.run(plan, self, workflow)
        return {
            "workflow_id": workflow.workflow_id,
            "results": results,
            "trace": workflow.trace,
            "metrics": self.metrics.summary(),
        }

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if self.aaflow_agent is not None and hasattr(self.aaflow_agent, "build_context"):
            try:
                context, debug = self.aaflow_agent.build_context(query)
                return [{"text": context, "metadata": {"source": "aaflow"}, "debug": debug}]
            except Exception as exc:
                return [{"text": f"AAFLOW retrieval failed: {exc}", "metadata": {"source": "fallback"}}]
        return [{"text": f"mock context for: {query}", "score": 1.0, "metadata": {"source": "mock"}}]

    def generate(self, prompt: str, kv_state_id: Optional[str] = None) -> str:
        if kv_state_id and kv_state_id in self.kv.states:
            self.kv.states[kv_state_id].reuse_count += 1
            self.metrics.mark_kv_reuse()
        if self.aaflow_agent is not None and hasattr(self.aaflow_agent, "llm"):
            llm = getattr(self.aaflow_agent, "llm")
            if hasattr(llm, "generate"):
                try:
                    return str(llm.generate(prompt=prompt))
                except Exception:
                    pass
        tokens = self.config.mock_tokens_per_answer
        return " ".join(["stateful"] * max(1, tokens))
