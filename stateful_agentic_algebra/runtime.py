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
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .compiler import (
    AlgebraCompiler,
    ExecutionPlan,
    NodeSpec,
    StatefulCompiler,
    StatefulExecutionGraph,
)
from .kv_manager import KVManagerConfig, KVStateManager
from .metrics_stateful import StatefulMetricsRecorder
from .operators import (
    KVEvictOperator,
    KVForkOperator,
    KVMergeOperator,
    KVMaterializeOperator,
    StatefulOperator,
)
from .scheduler import AlgebraScheduler, StateAwareScheduler
from .state_objects import KVState, KVStateStatus, OperatorSpec, WorkflowState
from .transport import AbstractTransport, Transport, TransportConfig, optional_available


@dataclass
class RuntimeConfig:
    """Runtime configuration with mock-safe defaults."""

    prefer_aaflow: bool = False
    generation_backend: str = "mock"
    transport_backend: str = "auto"
    bytes_per_token: int = 2048
    mock_tokens_per_answer: int = 32
    optional_backends: Dict[str, bool] = field(default_factory=dict)
    mock_layer_count: int = 1
    mock_head_count: int = 1
    mock_hidden_size: int = 128
    mock_dtype: str = "float16"
    model_id: str = "mock-model"
    tokenizer_id: str = "mock-tokenizer"
    model_config_hash: str = "mock-config"
    position_encoding: str = "rope"


@dataclass
class RuntimeMetrics:
    """Per-run metrics emitted as the runtime metrics JSON payload."""

    ttft_sec: float = 0.0
    total_latency_sec: float = 0.0
    prefill_sec: float = 0.0
    decode_sec: float = 0.0
    transfer_sec: float = 0.0
    resume_sec: float = 0.0
    omega_sec: float = 0.0
    kv_bytes: int = 0
    reuse_ratio: float = 0.0
    num_agents: int = 1
    branch_factor: int = 1
    context_tokens: int = 0

    def to_json_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable metrics dictionary."""

        return {
            "ttft_sec": self.ttft_sec,
            "total_latency_sec": self.total_latency_sec,
            "prefill_sec": self.prefill_sec,
            "decode_sec": self.decode_sec,
            "transfer_sec": self.transfer_sec,
            "resume_sec": self.resume_sec,
            "omega_sec": self.omega_sec,
            "kv_bytes": self.kv_bytes,
            "reuse_ratio": self.reuse_ratio,
            "num_agents": self.num_agents,
            "branch_factor": self.branch_factor,
            "context_tokens": self.context_tokens,
        }


class StatefulRuntime:
    """Execute compiled stateful graphs in mock-safe CPU mode.

    The runtime consumes `StatefulExecutionGraph` objects from `compiler.py`
    and executes their explicit KV lifecycle nodes. It is graph-native, but it
    reuses the same KV operators, state manager, scheduler, transport, and
    metrics recorder used by the compatibility runtime below.
    """

    def __init__(
        self,
        compiler: Optional[StatefulCompiler] = None,
        scheduler: Optional[StateAwareScheduler] = None,
        kv_manager: Optional[KVStateManager] = None,
        transport: Optional[AbstractTransport] = None,
        metrics_collector: Optional[StatefulMetricsRecorder] = None,
        config: Optional[RuntimeConfig] = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.compiler = compiler or StatefulCompiler()
        self.scheduler = scheduler or StateAwareScheduler()
        self.metrics = metrics_collector or StatefulMetricsRecorder()
        self.kv_manager = kv_manager or KVStateManager(
            KVManagerConfig(
                bytes_per_token=self.config.bytes_per_token,
                model_id=self.config.model_id,
                tokenizer_id=self.config.tokenizer_id,
                model_config_hash=self.config.model_config_hash,
                position_encoding=self.config.position_encoding,
            ),
            self.metrics,
        )
        self.kv = self.kv_manager
        self.transport = transport or Transport(TransportConfig(backend=self.config.transport_backend))
        self.operator_registry: dict[str, StatefulOperator | Callable[..., Any]] = {
            "kv_materialize": KVMaterializeOperator(),
            "kv_fork": KVForkOperator(),
            "kv_merge": KVMergeOperator(),
            "kv_evict": KVEvictOperator(),
        }
        self._workflow = WorkflowState()
        self._state_bindings: dict[str, str] = {}
        self._data_bindings: dict[str, Any] = {}
        self._node_results: dict[str, Any] = {}
        self._run_metrics = RuntimeMetrics()
        self._first_generate_seen = False

    def register_operator(self, operator_type: str, operator: StatefulOperator | Callable[..., Any]) -> None:
        """Register or replace an operator implementation by graph type."""

        self.operator_registry[operator_type] = operator

    def execute_graph(
        self,
        graph: StatefulExecutionGraph,
        initial_values: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a compiled graph and return values, states, trace, metrics."""

        graph.validate()
        context = context or {}
        self._workflow = WorkflowState()
        self._state_bindings = {}
        self._data_bindings = dict(initial_values or {})
        self._node_results = {}
        self._run_metrics = RuntimeMetrics(
            num_agents=int(context.get("num_agents", 1)),
            branch_factor=int(context.get("branch_factor", 1)),
            context_tokens=int(context.get("context_tokens", self._data_bindings.get("token_count", 0) or 0)),
        )
        self._first_generate_seen = False
        run_start = time.perf_counter()

        for node_id in graph.topological_order():
            self.execute_node(graph.nodes[node_id])

        observed_total = time.perf_counter() - run_start
        synthetic_total = (
            self._run_metrics.prefill_sec
            + self._run_metrics.decode_sec
            + self._run_metrics.transfer_sec
            + self._run_metrics.resume_sec
            + self._run_metrics.omega_sec
        )
        self._run_metrics.total_latency_sec = max(observed_total, synthetic_total)
        self._run_metrics.kv_bytes = max(self._run_metrics.kv_bytes, self.kv_manager.live_bytes())
        self._run_metrics.reuse_ratio = self.metrics.reuse_ratio()
        metrics = self._run_metrics.to_json_dict()

        return {
            "workflow_id": self._workflow.workflow_id,
            "results": self._node_results,
            "values": self._data_bindings,
            "state_bindings": dict(self._state_bindings),
            "trace": self._workflow.trace,
            "metrics": metrics,
            "metrics_json": json.dumps(metrics, sort_keys=True),
        }

    def execute_node(self, node: NodeSpec) -> Any:
        """Execute one graph node using the current runtime context."""

        start = time.perf_counter()
        self._workflow.add_trace("node_start", node_id=node.node_id, operator_type=node.operator_type)
        if node.operator_type == "kv_materialize":
            result = self._execute_materialize(node)
        elif node.operator_type == "kv_fork":
            result = self._execute_fork(node)
        elif node.operator_type == "kv_transfer":
            result = self._execute_transfer(node)
        elif node.operator_type == "generate":
            result = self._execute_generate(node)
        elif node.operator_type == "kv_merge":
            result = self._execute_merge(node)
        elif node.operator_type == "kv_evict":
            result = self._execute_evict(node)
        elif node.operator_type == "retrieve":
            result = self._execute_retrieve(node)
        else:
            operator = self.operator_registry.get(node.operator_type)
            if operator is None:
                result = None
            elif hasattr(operator, "execute"):
                result = operator.execute(node=node, runtime=self)
            else:
                result = operator(node=node, runtime=self)

        omega = time.perf_counter() - start
        self._run_metrics.omega_sec += omega
        self.metrics.observe("framework_overhead_ms", omega * 1000.0)
        self.metrics.mark_completed_op()
        self._run_metrics.kv_bytes = max(self._run_metrics.kv_bytes, self.kv_manager.live_bytes())
        self.metrics.update_live_memory(self.kv_manager.live_bytes())
        self._node_results[node.node_id] = result
        self._workflow.add_trace("node_end", node_id=node.node_id)
        return result

    def run_linear_handoff(self, prompt: str, token_count: int) -> dict[str, Any]:
        """Compile and run a linear handoff workflow."""

        graph = self.compiler.compile_linear_handoff()
        return self.execute_graph(
            graph,
            initial_values={"prompt": prompt, "token_count": int(token_count), "query": prompt},
            context={"context_tokens": int(token_count), "num_agents": 1, "branch_factor": 1},
        )

    def run_branching(self, prompt: str, token_count: int, branch_count: int) -> dict[str, Any]:
        """Compile and run a branching multi-agent workflow."""

        branch_count = max(1, int(branch_count))
        graph = self.compiler.compile_branching_workflow(branch_count)
        return self.execute_graph(
            graph,
            initial_values={"prompt": prompt, "token_count": int(token_count), "query": prompt},
            context={"context_tokens": int(token_count), "num_agents": branch_count, "branch_factor": branch_count},
        )

    def run_tree_of_thought(
        self,
        prompt: str,
        token_count: int,
        depth: int,
        branch_factor: int,
    ) -> dict[str, Any]:
        """Compile and run a mock tree-of-thought workflow."""

        branch_factor = max(1, int(branch_factor))
        graph = self.compiler.compile_tree_of_thought(depth=depth, branch_factor=branch_factor)
        num_agents = branch_factor ** max(1, int(depth))
        return self.execute_graph(
            graph,
            initial_values={"prompt": prompt, "token_count": int(token_count), "query": prompt},
            context={"context_tokens": int(token_count), "num_agents": num_agents, "branch_factor": branch_factor},
        )

    def _execute_materialize(self, node: NodeSpec) -> KVState:
        tokens = self._node_token_count(node)
        prefill = self.scheduler.estimate_prefill(tokens)
        operator = KVMaterializeOperator(
            name=node.node_id,
            layer_count=int(node.metadata.get("layer_count", self.config.mock_layer_count)),
            head_count=int(node.metadata.get("head_count", self.config.mock_head_count)),
            hidden_size=int(node.metadata.get("hidden_size", self.config.mock_hidden_size)),
            dtype=str(node.metadata.get("dtype", self.config.mock_dtype)),
            position_encoding=str(node.metadata.get("position_encoding", self.config.position_encoding)),
            owner_node=str(node.metadata.get("owner_node", "local")),
            owner_device=str(node.metadata.get("owner_device", "cpu")),
            backend=str(node.metadata.get("backend", "mock")),
        )
        kv_state = operator.execute(
            prompt_text=str(self._data_bindings.get("prompt", "")),
            token_count=tokens,
            model_id=str(node.metadata.get("model_id", self.config.model_id)),
            tokenizer_id=str(node.metadata.get("tokenizer_id", self.config.tokenizer_id)),
            model_config_hash=str(node.metadata.get("model_config_hash", self.config.model_config_hash)),
            metadata={**node.metadata, "token_count": tokens, "prefill_sec": prefill},
        )
        self._store_state(kv_state)
        self._bind_state_outputs(node, [kv_state])
        for output in node.outputs:
            self._data_bindings[output] = self._data_bindings.get("prompt", "")
        self._run_metrics.prefill_sec += prefill
        self.metrics.mark_kv_materialized()
        self.metrics.observe("prefill_ms", prefill * 1000.0)
        self.metrics.observe("recompute_cost_ms", prefill * 1000.0)
        return kv_state

    def _execute_fork(self, node: NodeSpec) -> list[KVState]:
        source = self._state_from_symbol(node.state_inputs[0])
        branch_count = int(node.metadata.get("branch_count", len(node.state_outputs) or 1))
        branches = KVForkOperator(name=node.node_id).execute(source, branch_count)
        for branch in branches:
            self._store_state(branch)
            self.metrics.mark_kv_reuse()
        self._bind_state_outputs(node, branches)
        for idx, output in enumerate(node.outputs):
            self._data_bindings[output] = self._data_bindings.get("prompt", f"branch {idx}")
        return branches

    def _execute_transfer(self, node: NodeSpec) -> KVState:
        source = self._state_from_symbol(node.state_inputs[0])
        target = str(node.metadata.get("target_node", node.metadata.get("target", source.owner_node)))
        token_count = self._node_token_count(node, fallback=self._state_token_count(source))
        decision = self.scheduler.decide(source, token_count=token_count, target_node=target)
        self._data_bindings[f"{node.node_id}.schedule_decision"] = decision

        if decision.decision == "local_reuse":
            moved = source
            self.metrics.mark_kv_reuse()
        elif decision.decision == "transfer":
            moved = self.transport.send_state(source, target)
            self._store_state(moved)
            transfer_sec = float(moved.metadata.get("transfer_latency_sec", self.transport.estimate_transfer_time(source)))
            self._run_metrics.transfer_sec += transfer_sec
            self.metrics.mark_kv_reuse()
            self.metrics.observe("transfer_cost_ms", transfer_sec * 1000.0)
        else:
            moved = self._recompute_state(source, target, token_count)
            self._run_metrics.prefill_sec += self.scheduler.estimate_prefill(token_count)

        self._bind_state_outputs(node, [moved])
        for output in node.outputs:
            self._data_bindings[output] = self._first_input_value(node)
        return moved

    def _execute_generate(self, node: NodeSpec) -> str:
        kv_state = self._state_from_symbol(node.state_inputs[0]) if node.state_inputs else None
        prompt = str(self._first_input_value(node) or self._data_bindings.get("prompt", ""))
        resume = self.scheduler.cost_model.resume_overhead_sec if kv_state is not None else 0.0
        decode = self.scheduler.estimate_decode(self.config.mock_tokens_per_answer)
        answer = f"mock answer for {node.node_id}: {prompt}".strip()

        self._run_metrics.resume_sec += resume
        self._run_metrics.decode_sec += decode
        self.metrics.observe("resume_ms", resume * 1000.0)
        self.metrics.observe("decode_ms", decode * 1000.0)
        if kv_state is not None:
            kv_state.metadata["reuse_count"] = int(kv_state.metadata.get("reuse_count", 0)) + 1
            self.metrics.mark_kv_reuse()
        if not self._first_generate_seen:
            first_token_decode = self.scheduler.estimate_decode(1)
            self._run_metrics.ttft_sec = self._run_metrics.prefill_sec + self._run_metrics.transfer_sec + resume + first_token_decode
            self.metrics.observe("ttft_ms", self._run_metrics.ttft_sec * 1000.0)
            self._first_generate_seen = True

        for output in node.outputs:
            self._data_bindings[output] = answer
        return answer

    def _execute_merge(self, node: NodeSpec) -> Any:
        states = [self._state_from_symbol(symbol) for symbol in node.state_inputs]
        policy = str(node.metadata.get("merge_policy", "segment_concat"))
        merged = KVMergeOperator(name=node.node_id).execute(states, merge_policy=policy)
        if isinstance(merged, KVState):
            self._store_state(merged)
            self._bind_state_outputs(node, [merged])
        elif node.state_outputs and states:
            summary_state = states[0].fork(f"kv_{uuid.uuid4().hex[:12]}")
            summary_state.metadata.update(
                {
                    "status": KVStateStatus.MERGED.value,
                    "merge_policy": policy,
                    "summary_reduce": getattr(merged, "summary", str(merged)),
                }
            )
            self._store_state(summary_state)
            self._bind_state_outputs(node, [summary_state])
        for output in node.outputs:
            self._data_bindings[output] = merged
        self.metrics.mark_kv_reuse()
        return merged

    def _execute_evict(self, node: NodeSpec) -> Any:
        source = self._state_from_symbol(node.state_inputs[0])
        record = KVEvictOperator(name=node.node_id).execute(source)
        self.metrics.update_live_memory(self.kv_manager.live_bytes())
        for output in node.outputs:
            self._data_bindings[output] = record
        return record

    def _execute_retrieve(self, node: NodeSpec) -> list[dict[str, Any]]:
        query = str(self._data_bindings.get("query", self._data_bindings.get("prompt", "")))
        results = [{"text": f"mock context for: {query}", "score": 1.0, "metadata": {"source": "mock"}}]
        for output in node.outputs:
            self._data_bindings[output] = results
        return results

    def _recompute_state(self, source: KVState, target: str, token_count: int) -> KVState:
        prefill = self.scheduler.estimate_prefill(token_count)
        operator = KVMaterializeOperator(
            layer_count=int(source.metadata.get("layer_count", self.config.mock_layer_count)),
            head_count=int(source.metadata.get("head_count", self.config.mock_head_count)),
            hidden_size=int(source.metadata.get("hidden_size", self.config.mock_hidden_size)),
            dtype=str(source.metadata.get("dtype", self.config.mock_dtype)),
            position_encoding=source.position_encoding,
            owner_node=target,
            owner_device=source.owner_device,
            backend="mock_recompute",
        )
        recomputed = operator.execute(
            token_count=token_count,
            model_id=source.model_id,
            tokenizer_id=source.tokenizer_id,
            model_config_hash=source.model_config_hash,
            metadata={
                "status": KVStateStatus.MATERIALIZED.value,
                "recomputed_from": source.state_id,
                "target_node": target,
                "prefill_sec": prefill,
            },
        )
        self._store_state(recomputed)
        self.metrics.mark_kv_materialized()
        self.metrics.observe("recompute_cost_ms", prefill * 1000.0)
        return recomputed

    def _store_state(self, kv_state: KVState) -> None:
        self.kv_manager.states[kv_state.state_id] = kv_state
        self._workflow.kv_states[kv_state.state_id] = kv_state
        self._run_metrics.kv_bytes = max(self._run_metrics.kv_bytes, kv_state.total_bytes(), self.kv_manager.live_bytes())

    def _bind_state_outputs(self, node: NodeSpec, states: list[KVState]) -> None:
        for symbol, kv_state in zip(node.state_outputs, states):
            self._state_bindings[symbol] = kv_state.state_id

    def _state_from_symbol(self, symbol: str) -> KVState:
        state_id = self._state_bindings.get(symbol, symbol)
        return self.kv_manager.get(state_id)

    def _node_token_count(self, node: NodeSpec, fallback: Optional[int] = None) -> int:
        value = node.metadata.get("token_count", node.metadata.get("tokens"))
        if value is None:
            value = self._data_bindings.get("token_count", fallback if fallback is not None else 0)
        return max(0, int(value))

    def _state_token_count(self, state: KVState) -> int:
        start, end = state.token_span()
        return max(0, end - start)

    def _first_input_value(self, node: NodeSpec) -> Any:
        if not node.inputs:
            return self._data_bindings.get("prompt", "")
        return self._data_bindings.get(node.inputs[0], self._data_bindings.get("prompt", ""))


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
            state = self.kv.states[kv_state_id]
            state.metadata["reuse_count"] = int(state.metadata.get("reuse_count", 0)) + 1
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
