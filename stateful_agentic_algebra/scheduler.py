"""Scheduler for Stateful Agentic Algebra execution plans.

Paper mapping:
  - Stateful operator algebra: schedules compiled operators.
  - KV state reuse: scheduler leaves KV lifecycle semantics to operators while
    recording framework overhead Omega at each operator boundary.
  - Throughput: completed operators are counted by the metrics recorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .compiler import ExecutionPlan
from .operators import OperatorResult
from .state_objects import WorkflowState


@dataclass
class SchedulerConfig:
    """Scheduler knobs. Parallel mode is reserved for future distributed runs."""

    mode: str = "sequential"


class AlgebraScheduler:
    """Simple scheduler that executes operators in plan order."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def run(self, plan: ExecutionPlan, runtime: Any, state: WorkflowState | None = None) -> List[OperatorResult]:
        workflow = state or WorkflowState()
        results: List[OperatorResult] = []
        for operator in plan.operators:
            results.append(operator.execute(runtime, workflow))
        return results

