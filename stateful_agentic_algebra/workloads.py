"""Workload generators for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: workloads emit operator specs.
  - KV lifecycle: generated plans include materialize, fork, transfer,
    restricted merge, generate, and evict.
  - Metrics: workloads are shaped to expose TTFT, transfer/recompute tradeoffs,
    throughput, memory pressure, reuse ratio, and framework overhead Omega.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .state_objects import OperatorSpec, OperatorType


@dataclass
class QueryWorkload:
    """A small query workload with shared prefix reuse."""

    prompts: List[str]
    shared_prefix_tokens: int = 128
    branch_tokens: int = 32


def synthetic_branching_workload(num_branches: int = 4, shared_prefix_tokens: int = 128) -> List[OperatorSpec]:
    """Create a branch/fork/merge workload for KV reuse experiments."""

    specs: List[OperatorSpec] = [
        OperatorSpec(
            name="materialize_prefix",
            op_type=OperatorType.MATERIALIZE,
            params={"tokens": shared_prefix_tokens, "recompute_cost_ms": shared_prefix_tokens * 0.02},
        )
    ]
    fork_names = []
    for idx in range(max(1, num_branches)):
        name = f"fork_{idx}"
        fork_names.append(name)
        specs.append(
            OperatorSpec(
                name=name,
                op_type=OperatorType.FORK,
                inputs=["materialize_prefix"],
                depends_on={"materialize_prefix"},
            )
        )
    specs.append(
        OperatorSpec(
            name="merge_branches",
            op_type=OperatorType.RESTRICTED_MERGE,
            inputs=fork_names,
            params={"token_limit": shared_prefix_tokens * 2},
            depends_on=set(fork_names),
        )
    )
    specs.append(
        OperatorSpec(
            name="generate_answer",
            op_type=OperatorType.GENERATE,
            inputs=["merge_branches"],
            params={"prompt": "Use the merged state to answer."},
            depends_on={"merge_branches"},
        )
    )
    return specs

