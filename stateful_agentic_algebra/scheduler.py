"""State-aware scheduler for KV transfer vs text recompute decisions.

Paper mapping:
  - Stateful operator algebra: schedules graph nodes while using explicit KV
    state edges to decide whether state should move or be recomputed.
  - KV transfer/recompute tradeoff:
      T_transfer = bytes / bandwidth + latency
      T_text = T_prefill + T_decode + omega_text
      T_state = T_transfer + T_resume + T_decode + omega_state
    The transfer decision compares state resume cost against text prefill cost:
      choose transfer if T_transfer + T_resume + omega_state
      < T_prefill + omega_text.
  - Metrics: decisions expose estimated time, memory, and reasons that can be
    aggregated as framework overhead Omega or scheduling diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .compiler import ExecutionPlan, StatefulExecutionGraph
from .operators import OperatorResult
from .state_objects import KVState, WorkflowState


@dataclass
class CostModel:
    """Cost parameters for transfer/recompute scheduling."""

    bandwidth_bytes_per_sec: float
    network_latency_sec: float
    prefill_time_per_token_sec: float
    decode_time_per_token_sec: float
    resume_overhead_sec: float
    omega_text_sec: float
    omega_state_sec: float
    memory_weight: float = 0.0


@dataclass
class ScheduleDecision:
    """Decision returned by the state-aware scheduler."""

    decision: str
    estimated_time_sec: float
    estimated_memory_bytes: int
    reason: str


@dataclass
class SchedulerConfig:
    """Legacy execution scheduler config retained for runtime compatibility."""

    mode: str = "sequential"


class StateAwareScheduler:
    """Decide whether to transfer KV state or recompute from text."""

    def __init__(self, cost_model: Optional[CostModel] = None, default_decode_tokens: int = 0) -> None:
        self.cost_model = cost_model or CostModel(
            bandwidth_bytes_per_sec=25_000_000_000,
            network_latency_sec=0.00005,
            prefill_time_per_token_sec=0.0002,
            decode_time_per_token_sec=0.00005,
            resume_overhead_sec=0.0001,
            omega_text_sec=0.00005,
            omega_state_sec=0.00005,
            memory_weight=0.0,
        )
        self.default_decode_tokens = max(0, int(default_decode_tokens))

    def estimate_prefill(self, token_count: int) -> float:
        """Estimate text prefill time."""

        return max(0, int(token_count)) * self.cost_model.prefill_time_per_token_sec

    def estimate_transfer(self, kv_state: KVState) -> float:
        """Estimate KV transfer time."""

        bandwidth = max(self.cost_model.bandwidth_bytes_per_sec, 1e-12)
        return kv_state.total_bytes() / bandwidth + self.cost_model.network_latency_sec

    def estimate_decode(self, new_tokens: int) -> float:
        """Estimate decode time for generated tokens."""

        return max(0, int(new_tokens)) * self.cost_model.decode_time_per_token_sec

    def decide(self, kv_state: KVState, token_count: int, target_node: str) -> ScheduleDecision:
        """Choose transfer, recompute, or local reuse."""

        decode_time = self.estimate_decode(self.default_decode_tokens)
        memory_penalty = kv_state.total_bytes() * self.cost_model.memory_weight

        if kv_state.owner_node == target_node:
            estimated = self.cost_model.resume_overhead_sec + decode_time + self.cost_model.omega_state_sec + memory_penalty
            return ScheduleDecision(
                decision="local_reuse",
                estimated_time_sec=estimated,
                estimated_memory_bytes=kv_state.total_bytes(),
                reason=f"KV state already resides on target node {target_node}",
            )

        transfer_time = self.estimate_transfer(kv_state)
        prefill_time = self.estimate_prefill(token_count)
        state_resume_side = transfer_time + self.cost_model.resume_overhead_sec + self.cost_model.omega_state_sec
        text_prefill_side = prefill_time + self.cost_model.omega_text_sec

        if state_resume_side < text_prefill_side:
            estimated = transfer_time + self.cost_model.resume_overhead_sec + decode_time + self.cost_model.omega_state_sec + memory_penalty
            return ScheduleDecision(
                decision="transfer",
                estimated_time_sec=estimated,
                estimated_memory_bytes=kv_state.total_bytes(),
                reason=(
                    "transfer chosen because "
                    f"T_transfer+T_resume+omega_state={state_resume_side:.6f}s "
                    f"< T_prefill+omega_text={text_prefill_side:.6f}s"
                ),
            )

        estimated = prefill_time + decode_time + self.cost_model.omega_text_sec
        return ScheduleDecision(
            decision="recompute",
            estimated_time_sec=estimated,
            estimated_memory_bytes=0,
            reason=(
                "recompute chosen because "
                f"T_transfer+T_resume+omega_state={state_resume_side:.6f}s "
                f">= T_prefill+omega_text={text_prefill_side:.6f}s"
            ),
        )

    def schedule_graph(self, graph: StatefulExecutionGraph) -> dict[str, Any]:
        """Annotate a graph with a topological schedule and transfer hints.

        The graph currently carries symbolic state ids, not concrete `KVState`
        objects, so this method returns node order and identifies transfer
        nodes where a runtime should call `decide()` once KV metadata is known.
        """

        order = graph.topological_order()
        transfer_nodes = [
            node_id
            for node_id in order
            if graph.nodes[node_id].operator_type == "kv_transfer"
        ]
        return {
            "topological_order": order,
            "transfer_nodes": transfer_nodes,
            "state_edges": [edge.edge_id for edge in graph.state_edges],
            "data_edges": [edge.edge_id for edge in graph.data_edges],
        }


class AlgebraScheduler:
    """Simple sequential execution adapter retained for existing runtime code."""

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()

    def run(self, plan: ExecutionPlan, runtime: Any, state: Optional[WorkflowState] = None) -> list[OperatorResult]:
        workflow = state or WorkflowState()
        results: list[OperatorResult] = []
        for operator in plan.operators:
            results.append(operator.execute(runtime, workflow))
        return results

