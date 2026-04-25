import json
import subprocess

from stateful_agentic_algebra.workloads import (
    WorkloadConfig,
    linear_handoff,
    multi_agent_debate,
    rag_shared_context,
    synthetic_branching_workload,
    transfer_recompute_crossover,
    tree_of_thought,
)


def test_workload_config_normalizes_counts():
    cfg = WorkloadConfig(
        workload_name="x",
        context_tokens=-1,
        output_tokens=-2,
        num_agents=0,
        branch_factor=0,
        depth=0,
        num_requests=0,
        seed=3,
    ).normalized()

    assert cfg.context_tokens == 0
    assert cfg.output_tokens == 0
    assert cfg.num_agents == 1
    assert cfg.branch_factor == 1
    assert cfg.depth == 1
    assert cfg.num_requests == 1


def test_linear_handoff_is_deterministic():
    cfg = WorkloadConfig(context_tokens=32, output_tokens=8, seed=11, num_requests=2)

    first = linear_handoff(cfg).to_json_dict()
    second = linear_handoff(cfg).to_json_dict()

    assert first == second
    assert first["config"]["workload_name"] == "linear_handoff"
    assert first["token_counts"] == [32, 32]


def test_multi_agent_and_tree_shapes():
    cfg = WorkloadConfig(context_tokens=16, output_tokens=4, num_agents=3, branch_factor=2, depth=3, num_requests=2)

    debate = multi_agent_debate(cfg)
    thought = tree_of_thought(cfg)

    assert len(debate.prompts) == 6
    assert debate.metadata["merge_policy"] == "summary_reduce"
    assert len(thought.prompts) == 2 * (1 + 2 + 4)
    assert thought.metadata["total_tree_nodes_per_request"] == 7


def test_rag_and_crossover_workloads_have_token_counts():
    cfg = WorkloadConfig(context_tokens=24, output_tokens=4, num_agents=2, branch_factor=2, seed=5)

    rag = rag_shared_context(cfg)
    crossover = transfer_recompute_crossover(cfg)

    assert rag.metadata["corpus_source"] in {"aaflow_local", "synthetic"}
    assert rag.token_counts
    assert crossover.token_counts == [6, 12, 24, 48, 96]
    assert crossover.config.num_requests == 5


def test_legacy_synthetic_branching_workload_still_emits_operator_specs():
    specs = synthetic_branching_workload(num_branches=2, shared_prefix_tokens=12)

    assert [spec.name for spec in specs] == [
        "materialize_prefix",
        "fork_0",
        "fork_1",
        "merge_branches",
        "generate_answer",
    ]


def test_workloads_demo_cli_prints_sample_configs():
    proc = subprocess.run(
        [
            "/scratch/djy8hg/env/drc_rag_bench_env/bin/python",
            "-m",
            "stateful_agentic_algebra.workloads",
            "--demo",
        ],
        cwd="/project/bi_dsc_community/drc_rag",
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert {item["config"]["workload_name"] for item in payload} == {
        "linear_handoff",
        "multi_agent_debate",
        "tree_of_thought",
        "rag_shared_context",
        "transfer_recompute_crossover",
    }
