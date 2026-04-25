"""Command-line experiment runner for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: compiles and executes generated workloads.
  - KV lifecycle: experiments include materialize/fork/restricted merge/evict
    and can be extended with transfer.
  - Metrics: writes TTFT, transfer cost, recompute cost, throughput, memory,
    reuse ratio, and framework overhead Omega summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .runtime import RuntimeConfig, StatefulAgenticRuntime
from .workloads import synthetic_branching_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stateful Agentic Algebra experiments")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--shared-prefix-tokens", type=int, default=128)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--generation-backend", choices=["mock", "aaflow", "vllm", "sglang"], default="mock")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = StatefulAgenticRuntime(RuntimeConfig(generation_backend=args.generation_backend))
    specs = synthetic_branching_workload(args.branches, args.shared_prefix_tokens)
    return runtime.run(specs)


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    payload = {
        "workflow_id": result["workflow_id"],
        "metrics": result["metrics"],
        "trace": result["trace"],
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

