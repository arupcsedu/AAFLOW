"""Compiler for Stateful Agentic Algebra execution graphs.

Paper mapping:
  - Stateful operator algebra: `StatefulCompiler` lowers high-level workflow
    templates into an explicit stateful execution graph G_s = (V, E_d, E_s).
  - Data edges E_d: ordinary data/control dependencies between operators.
  - State edges E_s: explicit KV-state dependencies across materialize, fork,
    transfer, merge, and evict operators.
  - KV lifecycle: compiler methods insert `kv_materialize`, `kv_fork`,
    `kv_transfer`, `kv_merge`, and `kv_evict` nodes rather than hiding state
    movement in generic model calls.
  - Compatibility: the legacy `ExecutionPlan`/`AlgebraCompiler` interface is
    retained for the package runtime that executes `OperatorSpec` sequences.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .operators import AlgebraOperator, build_operator
from .state_objects import OperatorSpec


@dataclass
class NodeSpec:
    """Node in the stateful execution graph."""

    node_id: str
    operator_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    state_inputs: list[str] = field(default_factory=list)
    state_outputs: list[str] = field(default_factory=list)
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Data or state edge in G_s."""

    edge_id: str
    source: str
    target: str
    edge_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in {"data", "state"}:
            raise ValueError(f"edge_type must be 'data' or 'state', got {self.edge_type!r}")


@dataclass
class StatefulExecutionGraph:
    """Stateful graph G_s = (V, E_d, E_s)."""

    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    data_edges: list[EdgeSpec] = field(default_factory=list)
    state_edges: list[EdgeSpec] = field(default_factory=list)

    def add_node(self, node: NodeSpec) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_data_edge(self, edge: EdgeSpec | str, source: Optional[str] = None, target: Optional[str] = None, **metadata: Any) -> None:
        if isinstance(edge, EdgeSpec):
            data_edge = edge
        else:
            if source is None or target is None:
                raise ValueError("source and target are required when edge id is provided")
            data_edge = EdgeSpec(edge_id=edge, source=source, target=target, edge_type="data", metadata=metadata)
        if data_edge.edge_type != "data":
            raise ValueError("add_data_edge requires edge_type='data'")
        self.data_edges.append(data_edge)

    def add_state_edge(self, edge: EdgeSpec | str, source: Optional[str] = None, target: Optional[str] = None, **metadata: Any) -> None:
        if isinstance(edge, EdgeSpec):
            state_edge = edge
        else:
            if source is None or target is None:
                raise ValueError("source and target are required when edge id is provided")
            state_edge = EdgeSpec(edge_id=edge, source=source, target=target, edge_type="state", metadata=metadata)
        if state_edge.edge_type != "state":
            raise ValueError("add_state_edge requires edge_type='state'")
        self.state_edges.append(state_edge)

    def topological_order(self) -> list[str]:
        """Return a topological order over data and state edges."""

        self.validate()
        incoming = {node_id: 0 for node_id in self.nodes}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in [*self.data_edges, *self.state_edges]:
            incoming[edge.target] += 1
            outgoing[edge.source].append(edge.target)

        ready = [node_id for node_id, count in incoming.items() if count == 0]
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for target in outgoing[node_id]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return order

    def validate(self) -> None:
        """Validate endpoints, edge types, edge ids, and acyclicity."""

        seen_edges: set[str] = set()
        for edge in [*self.data_edges, *self.state_edges]:
            if edge.edge_id in seen_edges:
                raise ValueError(f"Duplicate edge_id: {edge.edge_id}")
            seen_edges.add(edge.edge_id)
            if edge.source not in self.nodes:
                raise ValueError(f"Edge {edge.edge_id} has unknown source {edge.source}")
            if edge.target not in self.nodes:
                raise ValueError(f"Edge {edge.edge_id} has unknown target {edge.target}")
            if edge.edge_type not in {"data", "state"}:
                raise ValueError(f"Invalid edge type on {edge.edge_id}: {edge.edge_type}")

        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        incoming = {node_id: 0 for node_id in self.nodes}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in [*self.data_edges, *self.state_edges]:
            incoming[edge.target] += 1
            outgoing[edge.source].append(edge.target)
        ready = [node_id for node_id, count in incoming.items() if count == 0]
        visited = 0
        while ready:
            node_id = ready.pop(0)
            visited += 1
            for target in outgoing[node_id]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        if visited != len(self.nodes):
            raise ValueError("Graph contains a cycle")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "data_edges": [asdict(edge) for edge in self.data_edges],
            "state_edges": [asdict(edge) for edge in self.state_edges],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "StatefulExecutionGraph":
        graph = cls()
        for node_id, raw in payload.get("nodes", {}).items():
            data = dict(raw)
            data.setdefault("node_id", node_id)
            graph.add_node(NodeSpec(**data))
        for raw in payload.get("data_edges", []):
            graph.add_data_edge(EdgeSpec(**raw))
        for raw in payload.get("state_edges", []):
            graph.add_state_edge(EdgeSpec(**raw))
        return graph

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")


class StatefulCompiler:
    """Compile high-level workflow templates into explicit stateful graphs."""

    def __init__(self, default_resources: Optional[dict[str, Any]] = None) -> None:
        self.default_resources = default_resources or {"cpu": 1, "gpu": 0}

    def compile_linear_handoff(self) -> StatefulExecutionGraph:
        graph = StatefulExecutionGraph()
        self._add_node(graph, "materialize_0", "kv_materialize", outputs=["prompt_tokens"], state_outputs=["kv_0"])
        self._add_node(
            graph,
            "transfer_0",
            "kv_transfer",
            inputs=["prompt_tokens"],
            outputs=["remote_prompt_tokens"],
            state_inputs=["kv_0"],
            state_outputs=["kv_1"],
            metadata={"source_node": "node_0", "target_node": "node_1"},
        )
        self._add_node(graph, "generate_0", "generate", inputs=["remote_prompt_tokens"], state_inputs=["kv_1"], outputs=["answer"])
        self._add_node(graph, "evict_0", "kv_evict", state_inputs=["kv_1"], outputs=["eviction_record"])
        self._state(graph, "materialize_0", "transfer_0", state_id="kv_0")
        self._data(graph, "materialize_0", "transfer_0", value="prompt_tokens")
        self._state(graph, "transfer_0", "generate_0", state_id="kv_1")
        self._data(graph, "transfer_0", "generate_0", value="remote_prompt_tokens")
        self._state(graph, "generate_0", "evict_0", state_id="kv_1")
        graph.validate()
        return graph

    def compile_branching_workflow(self, branch_count: int) -> StatefulExecutionGraph:
        graph = StatefulExecutionGraph()
        branch_count = max(1, int(branch_count))
        self._add_node(graph, "materialize_root", "kv_materialize", outputs=["root_prompt"], state_outputs=["kv_root"])
        self._add_node(
            graph,
            "fork_root",
            "kv_fork",
            inputs=["root_prompt"],
            outputs=[f"branch_prompt_{idx}" for idx in range(branch_count)],
            state_inputs=["kv_root"],
            state_outputs=[f"kv_branch_{idx}" for idx in range(branch_count)],
            metadata={"branch_count": branch_count},
        )
        self._state(graph, "materialize_root", "fork_root", state_id="kv_root")
        self._data(graph, "materialize_root", "fork_root", value="root_prompt")
        branch_nodes = []
        for idx in range(branch_count):
            transfer_id = f"transfer_branch_{idx}"
            gen_id = f"generate_branch_{idx}"
            branch_nodes.append(gen_id)
            self._add_node(
                graph,
                transfer_id,
                "kv_transfer",
                inputs=[f"branch_prompt_{idx}"],
                outputs=[f"remote_branch_prompt_{idx}"],
                state_inputs=[f"kv_branch_{idx}"],
                state_outputs=[f"kv_remote_branch_{idx}"],
                metadata={"target_node": f"agent_node_{idx}"},
            )
            self._add_node(
                graph,
                gen_id,
                "generate",
                inputs=[f"remote_branch_prompt_{idx}"],
                outputs=[f"branch_answer_{idx}"],
                state_inputs=[f"kv_remote_branch_{idx}"],
            )
            self._state(graph, "fork_root", transfer_id, state_id=f"kv_branch_{idx}")
            self._data(graph, "fork_root", transfer_id, value=f"branch_prompt_{idx}")
            self._state(graph, transfer_id, gen_id, state_id=f"kv_remote_branch_{idx}")
            self._data(graph, transfer_id, gen_id, value=f"remote_branch_prompt_{idx}")
        self._add_node(
            graph,
            "merge_branches",
            "kv_merge",
            inputs=[f"branch_answer_{idx}" for idx in range(branch_count)],
            outputs=["merged_answer"],
            state_inputs=[f"kv_remote_branch_{idx}" for idx in range(branch_count)],
            state_outputs=["kv_merged"],
            metadata={"merge_policy": "summary_reduce"},
        )
        for idx, gen_id in enumerate(branch_nodes):
            self._data(graph, gen_id, "merge_branches", value=f"branch_answer_{idx}")
            self._state(graph, gen_id, "merge_branches", state_id=f"kv_remote_branch_{idx}")
        self._add_node(graph, "evict_merged", "kv_evict", state_inputs=["kv_merged"], outputs=["eviction_record"])
        self._state(graph, "merge_branches", "evict_merged", state_id="kv_merged")
        graph.validate()
        return graph

    def compile_tree_of_thought(self, depth: int, branch_factor: int) -> StatefulExecutionGraph:
        graph = StatefulExecutionGraph()
        depth = max(1, int(depth))
        branch_factor = max(1, int(branch_factor))
        self._add_node(graph, "materialize_root", "kv_materialize", outputs=["thought_root"], state_outputs=["kv_root"])
        previous_level = [("materialize_root", "kv_root", "thought_root")]
        for level in range(depth):
            next_level: list[tuple[str, str, str]] = []
            for parent_idx, (parent_node, parent_state, parent_value) in enumerate(previous_level):
                fork_id = f"fork_l{level}_p{parent_idx}"
                self._add_node(
                    graph,
                    fork_id,
                    "kv_fork",
                    inputs=[parent_value],
                    outputs=[f"thought_l{level}_p{parent_idx}_b{b}" for b in range(branch_factor)],
                    state_inputs=[parent_state],
                    state_outputs=[f"kv_l{level}_p{parent_idx}_b{b}" for b in range(branch_factor)],
                    metadata={"branch_count": branch_factor, "level": level},
                )
                self._data(graph, parent_node, fork_id, value=parent_value)
                self._state(graph, parent_node, fork_id, state_id=parent_state)
                for branch in range(branch_factor):
                    gen_id = f"generate_l{level}_p{parent_idx}_b{branch}"
                    state_id = f"kv_l{level}_p{parent_idx}_b{branch}"
                    value_id = f"thought_l{level}_p{parent_idx}_b{branch}"
                    self._add_node(graph, gen_id, "generate", inputs=[value_id], outputs=[f"{value_id}_out"], state_inputs=[state_id])
                    self._data(graph, fork_id, gen_id, value=value_id)
                    self._state(graph, fork_id, gen_id, state_id=state_id)
                    next_level.append((gen_id, state_id, f"{value_id}_out"))
            previous_level = next_level
        self._add_node(
            graph,
            "merge_thoughts",
            "kv_merge",
            inputs=[value for _, _, value in previous_level],
            outputs=["final_thought"],
            state_inputs=[state_id for _, state_id, _ in previous_level],
            state_outputs=["kv_final"],
            metadata={"merge_policy": "summary_reduce"},
        )
        for node_id, state_id, value_id in previous_level:
            self._data(graph, node_id, "merge_thoughts", value=value_id)
            self._state(graph, node_id, "merge_thoughts", state_id=state_id)
        self._add_node(graph, "evict_final", "kv_evict", state_inputs=["kv_final"], outputs=["eviction_record"])
        self._state(graph, "merge_thoughts", "evict_final", state_id="kv_final")
        graph.validate()
        return graph

    def compile_rag_multi_agent(self, num_agents: int) -> StatefulExecutionGraph:
        graph = StatefulExecutionGraph()
        num_agents = max(1, int(num_agents))
        self._add_node(graph, "retrieve", "retrieve", inputs=["query"], outputs=["retrieved_context"])
        self._add_node(graph, "materialize_context", "kv_materialize", inputs=["retrieved_context"], outputs=["context_prompt"], state_outputs=["kv_context"])
        self._data(graph, "retrieve", "materialize_context", value="retrieved_context")
        self._add_node(
            graph,
            "fork_agents",
            "kv_fork",
            inputs=["context_prompt"],
            outputs=[f"agent_prompt_{idx}" for idx in range(num_agents)],
            state_inputs=["kv_context"],
            state_outputs=[f"kv_agent_{idx}" for idx in range(num_agents)],
            metadata={"branch_count": num_agents},
        )
        self._data(graph, "materialize_context", "fork_agents", value="context_prompt")
        self._state(graph, "materialize_context", "fork_agents", state_id="kv_context")
        for idx in range(num_agents):
            transfer_id = f"transfer_agent_{idx}"
            agent_id = f"agent_{idx}_generate"
            self._add_node(
                graph,
                transfer_id,
                "kv_transfer",
                inputs=[f"agent_prompt_{idx}"],
                outputs=[f"agent_remote_prompt_{idx}"],
                state_inputs=[f"kv_agent_{idx}"],
                state_outputs=[f"kv_agent_remote_{idx}"],
                metadata={"target_node": f"agent_{idx}"},
            )
            self._add_node(
                graph,
                agent_id,
                "generate",
                inputs=[f"agent_remote_prompt_{idx}"],
                outputs=[f"agent_answer_{idx}"],
                state_inputs=[f"kv_agent_remote_{idx}"],
            )
            self._data(graph, "fork_agents", transfer_id, value=f"agent_prompt_{idx}")
            self._state(graph, "fork_agents", transfer_id, state_id=f"kv_agent_{idx}")
            self._data(graph, transfer_id, agent_id, value=f"agent_remote_prompt_{idx}")
            self._state(graph, transfer_id, agent_id, state_id=f"kv_agent_remote_{idx}")
        self._add_node(
            graph,
            "merge_agent_answers",
            "kv_merge",
            inputs=[f"agent_answer_{idx}" for idx in range(num_agents)],
            outputs=["rag_answer"],
            state_inputs=[f"kv_agent_remote_{idx}" for idx in range(num_agents)],
            state_outputs=["kv_rag_answer"],
            metadata={"merge_policy": "summary_reduce"},
        )
        for idx in range(num_agents):
            self._data(graph, f"agent_{idx}_generate", "merge_agent_answers", value=f"agent_answer_{idx}")
            self._state(graph, f"agent_{idx}_generate", "merge_agent_answers", state_id=f"kv_agent_remote_{idx}")
        self._add_node(graph, "evict_rag", "kv_evict", state_inputs=["kv_rag_answer"], outputs=["eviction_record"])
        self._state(graph, "merge_agent_answers", "evict_rag", state_id="kv_rag_answer")
        graph.validate()
        return graph

    def compile_from_yaml(self, path: str | Path) -> StatefulExecutionGraph:
        """Compile a graph from a YAML or JSON workflow file.

        Expected schema:
          nodes:
            - node_id: ...
              operator_type: ...
          data_edges:
            - edge_id: ...
              source: ...
              target: ...
          state_edges:
            - edge_id: ...
              source: ...
              target: ...

        PyYAML is optional. If it is not installed, JSON content is accepted.
        """

        raw_text = Path(path).read_text(encoding="utf-8")
        payload = self._load_yaml_or_json(raw_text)
        graph = StatefulExecutionGraph()
        for raw in payload.get("nodes", []):
            graph.add_node(NodeSpec(**raw))
        for raw in payload.get("data_edges", []):
            raw = {**raw, "edge_type": "data"}
            graph.add_data_edge(EdgeSpec(**raw))
        for raw in payload.get("state_edges", []):
            raw = {**raw, "edge_type": "state"}
            graph.add_state_edge(EdgeSpec(**raw))
        graph.validate()
        return graph

    def _load_yaml_or_json(self, raw_text: str) -> dict[str, Any]:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(raw_text)
            return loaded or {}
        except Exception:
            return json.loads(raw_text)

    def _add_node(
        self,
        graph: StatefulExecutionGraph,
        node_id: str,
        operator_type: str,
        inputs: Optional[list[str]] = None,
        outputs: Optional[list[str]] = None,
        state_inputs: Optional[list[str]] = None,
        state_outputs: Optional[list[str]] = None,
        resources: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        graph.add_node(
            NodeSpec(
                node_id=node_id,
                operator_type=operator_type,
                inputs=inputs or [],
                outputs=outputs or [],
                state_inputs=state_inputs or [],
                state_outputs=state_outputs or [],
                resource_requirements=resources or dict(self.default_resources),
                metadata=metadata or {},
            )
        )

    def _data(self, graph: StatefulExecutionGraph, source: str, target: str, **metadata: Any) -> None:
        graph.add_data_edge(f"data_{len(graph.data_edges)}", source=source, target=target, **metadata)

    def _state(self, graph: StatefulExecutionGraph, source: str, target: str, **metadata: Any) -> None:
        graph.add_state_edge(f"state_{len(graph.state_edges)}", source=source, target=target, **metadata)


@dataclass
class ExecutionPlan:
    """Ordered executable algebra plan retained for runtime compatibility."""

    operators: list[AlgebraOperator] = field(default_factory=list)

    def names(self) -> list[str]:
        return [operator.spec.name for operator in self.operators]


class AlgebraCompiler:
    """Compile `OperatorSpec` sequences into a dependency-respecting plan."""

    def compile(self, specs: Iterable[OperatorSpec]) -> ExecutionPlan:
        pending = list(specs)
        emitted: list[OperatorSpec] = []
        seen = set()

        while pending:
            progressed = False
            for spec in list(pending):
                if spec.depends_on.issubset(seen):
                    emitted.append(spec)
                    seen.add(spec.name)
                    pending.remove(spec)
                    progressed = True
            if not progressed:
                names = ", ".join(spec.name for spec in pending)
                raise ValueError(f"Cyclic or missing dependencies in operator specs: {names}")

        return ExecutionPlan([build_operator(spec) for spec in emitted])

