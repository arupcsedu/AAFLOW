"""Workload generators for Stateful Agentic Algebra.

Paper mapping:
  - Stateful operator algebra: workloads describe graph shapes such as linear
    handoff, debate fan-out, tree-of-thought branching, and RAG shared context.
  - KV state object and lifecycle operations: token counts and prompt layouts
    are shaped to stress materialize, transfer, fork, restricted merge, evict,
    and transfer-vs-recompute decisions.
  - Metrics: generated configs carry context/output token counts, request
    counts, agent counts, branch factors, and depth so runtimes and baselines
    can report TTFT, transfer/recompute cost, throughput, memory, reuse ratio,
    and framework overhead Omega with the same schema.

The first implementation is synthetic and deterministic. It optionally samples
local AAFLOW corpus/query files when present, but never requires external data.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .state_objects import OperatorSpec, OperatorType


@dataclass
class WorkloadConfig:
    """Shared workload configuration for runtimes and baselines."""

    workload_name: str = "synthetic"
    context_tokens: int = 128
    output_tokens: int = 32
    num_agents: int = 4
    branch_factor: int = 4
    depth: int = 2
    num_requests: int = 1
    model_id: str = "mock-model"
    tokenizer_id: str = "mock-tokenizer"
    seed: int = 0

    def normalized(self) -> "WorkloadConfig":
        """Return a copy with count fields clamped to useful ranges."""

        return replace(
            self,
            context_tokens=max(0, int(self.context_tokens)),
            output_tokens=max(0, int(self.output_tokens)),
            num_agents=max(1, int(self.num_agents)),
            branch_factor=max(1, int(self.branch_factor)),
            depth=max(1, int(self.depth)),
            num_requests=max(1, int(self.num_requests)),
            model_id=str(self.model_id),
            tokenizer_id=str(self.tokenizer_id),
            seed=int(self.seed),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return asdict(self.normalized())


@dataclass
class GeneratedWorkload:
    """Concrete deterministic workload instance."""

    config: WorkloadConfig
    prompts: list[str] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable representation."""

        return {
            "config": self.config.to_json_dict(),
            "prompts": list(self.prompts),
            "token_counts": list(self.token_counts),
            "metadata": dict(self.metadata),
        }


@dataclass
class QueryWorkload:
    """Compatibility query workload with shared prefix reuse."""

    prompts: List[str]
    shared_prefix_tokens: int = 128
    branch_tokens: int = 32


def linear_handoff(config: Optional[WorkloadConfig] = None, **overrides: Any) -> GeneratedWorkload:
    """Generate a deterministic linear handoff workload."""

    cfg = _config("linear_handoff", config, overrides, num_agents=2, branch_factor=1, depth=1)
    prompts = [
        _synthetic_prompt(
            cfg,
            request_idx=idx,
            role="handoff source agent prepares state for target agent",
            tokens=cfg.context_tokens,
        )
        for idx in range(cfg.num_requests)
    ]
    return GeneratedWorkload(
        config=cfg,
        prompts=prompts,
        token_counts=[cfg.context_tokens for _ in prompts],
        metadata={"shape": "linear", "state_transfers_per_request": 1},
    )


def multi_agent_debate(config: Optional[WorkloadConfig] = None, **overrides: Any) -> GeneratedWorkload:
    """Generate a fan-out/fan-in debate workload."""

    cfg = _config("multi_agent_debate", config, overrides)
    prompts = []
    token_counts = []
    for request_idx in range(cfg.num_requests):
        for agent_idx in range(cfg.num_agents):
            prompts.append(
                _synthetic_prompt(
                    cfg,
                    request_idx=request_idx,
                    role=f"debate agent {agent_idx} evaluates shared evidence",
                    tokens=cfg.context_tokens,
                )
            )
            token_counts.append(cfg.context_tokens)
    return GeneratedWorkload(
        config=cfg,
        prompts=prompts,
        token_counts=token_counts,
        metadata={"shape": "branching", "merge_policy": "summary_reduce"},
    )


def tree_of_thought(config: Optional[WorkloadConfig] = None, **overrides: Any) -> GeneratedWorkload:
    """Generate a deterministic tree-of-thought workload."""

    cfg = _config("tree_of_thought", config, overrides)
    prompts = []
    token_counts = []
    rng = random.Random(cfg.seed)
    for request_idx in range(cfg.num_requests):
        for level in range(cfg.depth):
            node_count = cfg.branch_factor ** level
            level_tokens = cfg.context_tokens + level * max(1, cfg.output_tokens // 2)
            for node_idx in range(node_count):
                prompts.append(
                    _synthetic_prompt(
                        cfg,
                        request_idx=request_idx,
                        role=f"thought level {level} node {node_idx} score {rng.randint(0, 999)}",
                        tokens=level_tokens,
                    )
                )
                token_counts.append(level_tokens)
    return GeneratedWorkload(
        config=cfg,
        prompts=prompts,
        token_counts=token_counts,
        metadata={
            "shape": "tree",
            "total_tree_nodes_per_request": sum(cfg.branch_factor ** level for level in range(cfg.depth)),
        },
    )


def rag_shared_context(config: Optional[WorkloadConfig] = None, **overrides: Any) -> GeneratedWorkload:
    """Generate a RAG shared-context workload with optional AAFLOW corpus text."""

    cfg = _config("rag_shared_context", config, overrides)
    corpus = _load_aaflow_texts(cfg.seed, max_items=max(cfg.num_requests, cfg.num_agents))
    prompts = []
    token_counts = []
    for request_idx in range(cfg.num_requests):
        context = corpus[request_idx % len(corpus)] if corpus else ""
        prompt = _synthetic_prompt(
            cfg,
            request_idx=request_idx,
            role="RAG agents answer from shared retrieved context",
            tokens=cfg.context_tokens,
            prefix=context,
        )
        prompts.append(prompt)
        token_counts.append(_approx_token_count(prompt))
    return GeneratedWorkload(
        config=cfg,
        prompts=prompts,
        token_counts=token_counts,
        metadata={
            "shape": "rag_shared_context",
            "corpus_source": "aaflow_local" if corpus else "synthetic",
            "retrieval_fanout": cfg.num_agents,
        },
    )


def transfer_recompute_crossover(config: Optional[WorkloadConfig] = None, **overrides: Any) -> GeneratedWorkload:
    """Generate token counts around the transfer/recompute crossover."""

    cfg = _config("transfer_recompute_crossover", config, overrides)
    base = max(1, cfg.context_tokens)
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    token_counts = [max(1, int(base * multiplier)) for multiplier in multipliers]
    prompts = [
        _synthetic_prompt(
            cfg,
            request_idx=idx,
            role=f"crossover point {idx} compares KV transfer with text recompute",
            tokens=tokens,
        )
        for idx, tokens in enumerate(token_counts)
    ]
    return GeneratedWorkload(
        config=replace(cfg, num_requests=len(token_counts)),
        prompts=prompts,
        token_counts=token_counts,
        metadata={"shape": "crossover", "sweep": "context_tokens"},
    )


def synthetic_branching_workload(num_branches: int = 4, shared_prefix_tokens: int = 128) -> List[OperatorSpec]:
    """Create a legacy operator-spec workload for compatibility."""

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


def demo_workloads() -> list[GeneratedWorkload]:
    """Return sample workloads for CLI/demo output."""

    base = WorkloadConfig(context_tokens=64, output_tokens=16, num_agents=3, branch_factor=2, depth=2, seed=7)
    return [
        linear_handoff(base),
        multi_agent_debate(base),
        tree_of_thought(base),
        rag_shared_context(base),
        transfer_recompute_crossover(base),
    ]


def _config(
    workload_name: str,
    config: Optional[WorkloadConfig],
    overrides: dict[str, Any],
    **defaults: Any,
) -> WorkloadConfig:
    base = config or WorkloadConfig()
    values = {**base.to_json_dict(), **defaults, **overrides, "workload_name": workload_name}
    return WorkloadConfig(**{key: values[key] for key in WorkloadConfig.__dataclass_fields__}).normalized()


def _synthetic_prompt(
    cfg: WorkloadConfig,
    *,
    request_idx: int,
    role: str,
    tokens: int,
    prefix: str = "",
) -> str:
    rng = random.Random(cfg.seed + request_idx * 7919 + len(role))
    vocabulary = [
        "state",
        "agent",
        "cache",
        "prefix",
        "transfer",
        "recompute",
        "memory",
        "context",
        "reason",
        "merge",
        "token",
        "latency",
    ]
    header = _words(f"{cfg.workload_name} request {request_idx} {role}")
    prefix_budget = max(0, tokens - len(header))
    prefix_tokens = _words(prefix)[:prefix_budget]
    remaining = max(0, tokens - len(prefix_tokens) - len(header))
    body = [vocabulary[rng.randrange(len(vocabulary))] for _ in range(remaining)]
    return " ".join([*prefix_tokens, *header, *body]).strip()


def _words(text: str) -> list[str]:
    return [part for part in text.replace("\n", " ").split(" ") if part]


def _approx_token_count(text: str) -> int:
    return len(_words(text))


def _load_aaflow_texts(seed: int, max_items: int) -> list[str]:
    """Load deterministic snippets from local AAFLOW corpus/query files."""

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "higress_agentic_benchmark" / "combined_wikitext2_2000" / "combined_2000_docs.txt",
        repo_root / "higress_agentic_benchmark" / "sample_wikitext2_2000",
    ]
    texts: list[str] = []
    for candidate in candidates:
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8", errors="ignore")
            texts.extend(_chunk_text(raw, max_items=max_items * 2))
        elif candidate.is_dir():
            files = sorted(candidate.glob("*.txt"))
            rng = random.Random(seed)
            rng.shuffle(files)
            for path in files[: max(1, max_items)]:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                if raw.strip():
                    texts.append(" ".join(_words(raw)[:160]))
        if texts:
            break
    return [text for text in texts if text][: max(1, max_items)]


def _chunk_text(raw: str, max_items: int) -> list[str]:
    words = _words(raw)
    if not words:
        return []
    chunk_size = 160
    chunks = []
    for start in range(0, min(len(words), max_items * chunk_size), chunk_size):
        chunks.append(" ".join(words[start : start + chunk_size]))
    return chunks


def _demo_payload() -> list[dict[str, Any]]:
    return [workload.to_json_dict() for workload in demo_workloads()]


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for workload demos."""

    parser = argparse.ArgumentParser(description="Stateful Agentic Algebra workload generator")
    parser.add_argument("--demo", action="store_true", help="Print sample workload configs")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.demo:
        print(json.dumps(_demo_payload(), indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
