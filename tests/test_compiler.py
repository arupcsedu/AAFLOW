import json

from stateful_agentic_algebra.compiler import (
    EdgeSpec,
    NodeSpec,
    StatefulCompiler,
    StatefulExecutionGraph,
)


def operator_types(graph):
    return {node.operator_type for node in graph.nodes.values()}


def test_graph_add_validate_and_topological_order():
    graph = StatefulExecutionGraph()
    graph.add_node(NodeSpec("a", "kv_materialize"))
    graph.add_node(NodeSpec("b", "kv_transfer"))
    graph.add_data_edge(EdgeSpec("d0", "a", "b", "data"))
    graph.add_state_edge(EdgeSpec("s0", "a", "b", "state"))

    graph.validate()
    assert graph.topological_order() == ["a", "b"]


def test_validate_rejects_unknown_edge_endpoint():
    graph = StatefulExecutionGraph()
    graph.add_node(NodeSpec("a", "kv_materialize"))
    graph.add_data_edge(EdgeSpec("d0", "a", "missing", "data"))

    try:
        graph.validate()
    except ValueError as exc:
        assert "unknown target" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_compile_linear_handoff_inserts_explicit_kv_ops():
    graph = StatefulCompiler().compile_linear_handoff()

    assert {"kv_materialize", "kv_transfer", "generate", "kv_evict"}.issubset(operator_types(graph))
    assert len(graph.state_edges) == 3
    assert graph.topological_order()[0] == "materialize_0"
    assert graph.topological_order()[-1] == "evict_0"


def test_compile_branching_workflow_inserts_fork_merge_transfer_evict():
    graph = StatefulCompiler().compile_branching_workflow(branch_count=3)
    types = operator_types(graph)

    assert {"kv_materialize", "kv_fork", "kv_transfer", "generate", "kv_merge", "kv_evict"}.issubset(types)
    assert graph.nodes["fork_root"].metadata["branch_count"] == 3
    assert len([node for node in graph.nodes.values() if node.operator_type == "kv_transfer"]) == 3
    assert len(graph.nodes["merge_branches"].state_inputs) == 3
    graph.validate()


def test_compile_tree_of_thought_scales_by_depth_and_branch_factor():
    graph = StatefulCompiler().compile_tree_of_thought(depth=2, branch_factor=2)

    assert "merge_thoughts" in graph.nodes
    assert graph.nodes["merge_thoughts"].operator_type == "kv_merge"
    # Level 0 has 2 generate nodes; level 1 has 4.
    assert len([n for n in graph.nodes.values() if n.operator_type == "generate"]) == 6
    assert "kv_evict" in operator_types(graph)
    graph.validate()


def test_compile_rag_multi_agent_inserts_retrieve_and_agent_state_ops():
    graph = StatefulCompiler().compile_rag_multi_agent(num_agents=2)

    assert graph.nodes["retrieve"].operator_type == "retrieve"
    assert graph.nodes["materialize_context"].operator_type == "kv_materialize"
    assert graph.nodes["fork_agents"].operator_type == "kv_fork"
    assert len([n for n in graph.nodes.values() if n.operator_type == "kv_transfer"]) == 2
    assert graph.nodes["merge_agent_answers"].operator_type == "kv_merge"
    assert graph.nodes["evict_rag"].operator_type == "kv_evict"
    graph.validate()


def test_compile_from_yaml_accepts_json_content(tmp_path):
    path = tmp_path / "workflow.yaml"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"node_id": "m", "operator_type": "kv_materialize"},
                    {"node_id": "e", "operator_type": "kv_evict", "state_inputs": ["kv"]},
                ],
                "data_edges": [],
                "state_edges": [{"edge_id": "s0", "source": "m", "target": "e", "metadata": {"state_id": "kv"}}],
            }
        ),
        encoding="utf-8",
    )

    graph = StatefulCompiler().compile_from_yaml(path)

    assert set(graph.nodes) == {"m", "e"}
    assert len(graph.state_edges) == 1
    assert graph.topological_order() == ["m", "e"]


def test_graph_json_round_trip():
    graph = StatefulCompiler().compile_linear_handoff()
    payload = graph.to_json_dict()
    restored = StatefulExecutionGraph.from_json_dict(payload)

    assert restored.to_json_dict() == payload
    assert restored.topological_order() == graph.topological_order()

