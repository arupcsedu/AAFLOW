"""Compiler for Stateful Agentic Algebra plans.

Paper mapping:
  - Stateful operator algebra: validates and lowers symbolic `OperatorSpec`
    nodes into an `ExecutionPlan`.
  - KV lifecycle operators: preserves explicit materialize/transfer/fork/
    restricted merge/evict nodes for the scheduler/runtime.
  - Framework overhead Omega: compiler output is intentionally minimal so
    runtime overhead can be measured separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from .operators import AlgebraOperator, build_operator
from .state_objects import OperatorSpec


@dataclass
class ExecutionPlan:
    """Ordered executable algebra plan."""

    operators: List[AlgebraOperator] = field(default_factory=list)

    def names(self) -> List[str]:
        return [operator.spec.name for operator in self.operators]


class AlgebraCompiler:
    """Compile operator specs into a dependency-respecting sequential plan."""

    def compile(self, specs: Iterable[OperatorSpec]) -> ExecutionPlan:
        pending = list(specs)
        emitted: List[OperatorSpec] = []
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

