"""Baseline policies for Stateful Agentic Algebra experiments.

Paper mapping:
  - Recompute baseline: disables KV reuse and materializes every branch.
  - Reuse baseline: uses materialize/fork/restricted merge to quantify reuse
    ratio and memory/TTFT effects.
  - Framework overhead Omega: both baselines run through the same runtime so
    scheduler/operator overhead remains comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .runtime import StatefulAgenticRuntime
from .state_objects import OperatorSpec, OperatorType
from .workloads import synthetic_branching_workload


@dataclass
class BaselineResult:
    """Named baseline summary."""

    name: str
    metrics: Dict[str, float]


def recompute_baseline(num_branches: int = 4, tokens: int = 128) -> List[OperatorSpec]:
    """Build a plan that recomputes each branch instead of forking KV state."""

    specs: List[OperatorSpec] = []
    names = []
    for idx in range(max(1, num_branches)):
        name = f"materialize_branch_{idx}"
        names.append(name)
        specs.append(
            OperatorSpec(
                name=name,
                op_type=OperatorType.MATERIALIZE,
                params={"tokens": tokens, "recompute_cost_ms": tokens * 0.02},
            )
        )
    specs.append(
        OperatorSpec(
            name="merge_recomputed",
            op_type=OperatorType.RESTRICTED_MERGE,
            inputs=names,
            params={"token_limit": tokens * 2},
            depends_on=set(names),
        )
    )
    specs.append(
        OperatorSpec(
            name="generate_answer",
            op_type=OperatorType.GENERATE,
            inputs=["merge_recomputed"],
            params={"prompt": "Use recomputed state to answer."},
            depends_on={"merge_recomputed"},
        )
    )
    return specs


def run_default_baselines(runtime: StatefulAgenticRuntime | None = None) -> List[BaselineResult]:
    """Run reuse and recompute baselines with independent runtimes."""

    reuse_runtime = runtime or StatefulAgenticRuntime()
    reuse = reuse_runtime.run(synthetic_branching_workload())

    recompute_runtime = StatefulAgenticRuntime()
    recompute = recompute_runtime.run(recompute_baseline())
    return [
        BaselineResult("reuse", reuse["metrics"]),
        BaselineResult("recompute", recompute["metrics"]),
    ]

