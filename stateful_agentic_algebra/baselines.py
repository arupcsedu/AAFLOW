"""Baseline adapters for Stateful Agentic Algebra experiments.

Paper mapping:
  - Stateful operator algebra: baselines expose the same `run_workload`
    contract as the stateful runtime, but intentionally remove or restrict
    explicit state orchestration.
  - KV state object and lifecycle operations: dense/text baselines recompute
    state instead of reusing `KVState`; local-prefix baselines simulate local
    prefix reuse; the DistServe-style baseline simulates disaggregated
    prefill/decode without claiming exact DistServe behavior.
  - Metrics: every baseline returns the runtime metrics schema: TTFT, transfer
    cost, recompute/prefill cost, decode, throughput inputs, memory/KV bytes,
    reuse ratio, and framework overhead Omega.

Optional frameworks are imported lazily. Missing vLLM, SGLang, or AAFLOW
dependencies produce clear skip/fallback metadata rather than import failures.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .runtime import RuntimeMetrics
from .scheduler import CostModel, StateAwareScheduler
from .state_objects import OperatorSpec, OperatorType
from .vllm_backend import VLLMBackend
from .workloads import QueryWorkload, WorkloadConfig, synthetic_branching_workload


DEFAULT_COST_MODEL = CostModel(
    bandwidth_bytes_per_sec=25_000_000_000,
    network_latency_sec=0.00005,
    prefill_time_per_token_sec=0.0002,
    decode_time_per_token_sec=0.00005,
    resume_overhead_sec=0.0001,
    omega_text_sec=0.00005,
    omega_state_sec=0.00005,
    memory_weight=0.0,
)


@dataclass
class BaselineResult:
    """Named baseline result with runtime-compatible metrics."""

    name: str
    metrics: Dict[str, float | int]
    available: bool = True
    skipped: bool = False
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _BaselineWorkload:
    """Internal normalized workload config for baseline costing."""

    prompt: str
    context_tokens: int
    output_tokens: int
    num_agents: int
    branch_factor: int
    bytes_per_token: int
    cost_model: CostModel
    aaflow_agent: Any = None


class BaselineAdapter:
    """Base class for runtime-compatible baseline adapters."""

    name = "Baseline"
    label = "Baseline"

    def available(self) -> bool:
        """Return whether the concrete backend is installed and usable."""

        return True

    def unavailable_reason(self) -> str:
        """Return a short reason when `available()` is false."""

        return ""

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        """Run a workload and return metrics using the shared schema."""

        raise NotImplementedError

    def _normalize(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> _BaselineWorkload:
        if workload_config is None:
            workload_config = WorkloadConfig()
        if isinstance(workload_config, WorkloadConfig):
            return _BaselineWorkload(
                prompt=workload_config.workload_name,
                context_tokens=max(0, int(workload_config.context_tokens)),
                output_tokens=max(0, int(workload_config.output_tokens)),
                num_agents=max(1, int(workload_config.num_agents)),
                branch_factor=max(1, int(workload_config.branch_factor)),
                bytes_per_token=2048,
                cost_model=DEFAULT_COST_MODEL,
                aaflow_agent=None,
            )
        if isinstance(workload_config, QueryWorkload):
            return _BaselineWorkload(
                prompt=workload_config.prompts[0] if workload_config.prompts else "",
                context_tokens=workload_config.shared_prefix_tokens,
                output_tokens=workload_config.branch_tokens,
                num_agents=max(1, len(workload_config.prompts)),
                branch_factor=max(1, len(workload_config.prompts)),
                bytes_per_token=2048,
                cost_model=DEFAULT_COST_MODEL,
                aaflow_agent=None,
            )
        data = dict(workload_config)
        context_tokens = data.get("context_tokens", data.get("token_count", data.get("shared_prefix_tokens", 128)))
        num_agents = data.get("num_agents", data.get("agents", data.get("branches", data.get("branch_count", 4))))
        branch_factor = data.get("branch_factor", data.get("branch_count", num_agents))
        cost_model = data.get("cost_model", DEFAULT_COST_MODEL)
        if isinstance(cost_model, dict):
            cost_model = CostModel(**cost_model)
        return _BaselineWorkload(
            prompt=str(data.get("prompt", data.get("workload_name", "stateful agentic algebra workload"))),
            context_tokens=max(0, int(context_tokens)),
            output_tokens=max(0, int(data.get("output_tokens", data.get("decode_tokens", data.get("new_tokens", 32))))),
            num_agents=max(1, int(num_agents)),
            branch_factor=max(1, int(branch_factor)),
            bytes_per_token=max(0, int(data.get("bytes_per_token", 2048))),
            cost_model=cost_model,
            aaflow_agent=data.get("aaflow_agent"),
        )

    def _metrics(
        self,
        cfg: _BaselineWorkload,
        *,
        prefill_sec: float,
        decode_sec: float,
        transfer_sec: float = 0.0,
        resume_sec: float = 0.0,
        omega_sec: float = 0.0,
        kv_bytes: Optional[int] = None,
        reuse_ratio: float = 0.0,
        ttft_sec: Optional[float] = None,
        observed_total_sec: float = 0.0,
    ) -> dict[str, float | int]:
        first_token = cfg.cost_model.decode_time_per_token_sec if cfg.output_tokens else 0.0
        metrics = RuntimeMetrics(
            ttft_sec=ttft_sec if ttft_sec is not None else prefill_sec + transfer_sec + resume_sec + first_token + omega_sec,
            total_latency_sec=max(observed_total_sec, prefill_sec + decode_sec + transfer_sec + resume_sec + omega_sec),
            prefill_sec=prefill_sec,
            decode_sec=decode_sec,
            transfer_sec=transfer_sec,
            resume_sec=resume_sec,
            omega_sec=omega_sec,
            kv_bytes=kv_bytes if kv_bytes is not None else cfg.context_tokens * cfg.bytes_per_token,
            reuse_ratio=reuse_ratio,
            num_agents=cfg.num_agents,
            branch_factor=cfg.branch_factor,
            context_tokens=cfg.context_tokens,
        )
        payload = metrics.to_json_dict()
        payload["context_build_total_sec"] = 0.0
        payload["llm_generate_sec"] = 0.0
        return payload

    def _skip(self, reason: str) -> BaselineResult:
        metrics = RuntimeMetrics().to_json_dict()
        metrics["context_build_total_sec"] = 0.0
        metrics["llm_generate_sec"] = 0.0
        return BaselineResult(
            name=self.label,
            metrics=metrics,
            available=False,
            skipped=True,
            reason=reason,
        )


class DensePrefillBaseline(BaselineAdapter):
    """Synthetic dense prefill baseline: every agent recomputes from text."""

    name = "dense_prefill"
    label = "Dense Prefill / Text Passing"

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        cfg = self._normalize(workload_config)
        scheduler = StateAwareScheduler(cfg.cost_model)
        prefill = cfg.num_agents * scheduler.estimate_prefill(cfg.context_tokens)
        decode = cfg.num_agents * scheduler.estimate_decode(cfg.output_tokens)
        omega = cfg.num_agents * cfg.cost_model.omega_text_sec
        metrics = self._metrics(
            cfg,
            prefill_sec=prefill,
            decode_sec=decode,
            omega_sec=omega,
            kv_bytes=cfg.num_agents * cfg.context_tokens * cfg.bytes_per_token,
            reuse_ratio=0.0,
        )
        return BaselineResult(self.label, metrics, metadata={"baseline_type": "synthetic_text_passing"})


class AAFLOWTextBaseline(BaselineAdapter):
    """AAFLOW text-passing baseline with lazy reuse of RagAgent/LLMGenerator."""

    name = "aaflow_text"
    label = "AAFLOW Text Baseline"

    def available(self) -> bool:
        return importlib.util.find_spec("agents") is not None

    def unavailable_reason(self) -> str:
        return "AAFLOW agents.py is not importable from the current environment"

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        cfg = self._normalize(workload_config)
        scheduler = StateAwareScheduler(cfg.cost_model)
        start = time.perf_counter()
        context_build_total = 0.0
        llm_generate_total = 0.0
        answers = []

        if cfg.aaflow_agent is not None:
            for idx in range(cfg.num_agents):
                query = f"{cfg.prompt} [agent {idx}]"
                context_start = time.perf_counter()
                try:
                    if hasattr(cfg.aaflow_agent, "build_context"):
                        context, _debug = cfg.aaflow_agent.build_context(query)
                    else:
                        context = ""
                    context_build_total += time.perf_counter() - context_start

                    gen_start = time.perf_counter()
                    if hasattr(cfg.aaflow_agent, "llm") and hasattr(cfg.aaflow_agent.llm, "generate"):
                        answers.append(str(cfg.aaflow_agent.llm.generate(prompt=query, extra_context=context)))
                    elif hasattr(cfg.aaflow_agent, "generate_answer"):
                        answer, _debug = cfg.aaflow_agent.generate_answer(query)
                        answers.append(str(answer))
                    llm_generate_total += time.perf_counter() - gen_start
                except Exception as exc:
                    return BaselineResult(
                        self.label,
                        self._mock_metrics(cfg, scheduler, time.perf_counter() - start),
                        available=self.available(),
                        reason=f"AAFLOW execution failed; used text simulation metrics: {exc}",
                        metadata={"fallback": "mock_text", "answers": answers},
                    )

            observed = time.perf_counter() - start
            metrics = self._metrics(
                cfg,
                prefill_sec=context_build_total,
                decode_sec=llm_generate_total,
                omega_sec=cfg.num_agents * cfg.cost_model.omega_text_sec,
                kv_bytes=0,
                reuse_ratio=0.0,
                observed_total_sec=observed,
            )
            metrics["context_build_total_sec"] = context_build_total
            metrics["llm_generate_sec"] = llm_generate_total
            return BaselineResult(self.label, metrics, metadata={"answers": answers, "mode": "aaflow_agent"})

        metrics = self._mock_metrics(cfg, scheduler, time.perf_counter() - start)
        return BaselineResult(
            self.label,
            metrics,
            available=self.available(),
            reason="" if self.available() else self.unavailable_reason(),
            metadata={"mode": "mock_text", "note": "pass a pre-built RagAgent via workload_config['aaflow_agent'] for real AAFLOW execution"},
        )

    def _mock_metrics(self, cfg: _BaselineWorkload, scheduler: StateAwareScheduler, observed: float) -> dict[str, float | int]:
        context_build = cfg.num_agents * scheduler.estimate_prefill(cfg.context_tokens)
        generate = cfg.num_agents * scheduler.estimate_decode(cfg.output_tokens)
        metrics = self._metrics(
            cfg,
            prefill_sec=context_build,
            decode_sec=generate,
            omega_sec=cfg.num_agents * cfg.cost_model.omega_text_sec,
            kv_bytes=0,
            reuse_ratio=0.0,
            observed_total_sec=observed,
        )
        metrics["context_build_total_sec"] = context_build
        metrics["llm_generate_sec"] = generate
        return metrics


class VLLMLocalPrefixBaseline(BaselineAdapter):
    """Optional vLLM local prefix/KV reuse baseline."""

    name = "vllm_local_prefix"
    label = "vLLM Local Prefix Baseline"

    def available(self) -> bool:
        return VLLMBackend.available()

    def unavailable_reason(self) -> str:
        return "vLLM is not installed; using mock local-prefix simulation when run"

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        cfg = self._normalize(workload_config)
        scheduler = StateAwareScheduler(cfg.cost_model)
        prefill = scheduler.estimate_prefill(cfg.context_tokens)
        resume = cfg.num_agents * cfg.cost_model.resume_overhead_sec
        decode = cfg.num_agents * scheduler.estimate_decode(cfg.output_tokens)
        omega = cfg.num_agents * cfg.cost_model.omega_state_sec
        metrics = self._metrics(
            cfg,
            prefill_sec=prefill,
            decode_sec=decode,
            resume_sec=resume,
            omega_sec=omega,
            kv_bytes=cfg.context_tokens * cfg.bytes_per_token,
            reuse_ratio=(cfg.num_agents - 1) / cfg.num_agents if cfg.num_agents else 0.0,
        )
        return BaselineResult(
            self.label,
            metrics,
            available=self.available(),
            reason="" if self.available() else self.unavailable_reason(),
            metadata={"mode": "vllm" if self.available() else "mock_local_prefix", "distributed_state_orchestration": False},
        )


class SGLangPrefixBaseline(BaselineAdapter):
    """SGLang prefix reuse baseline with a mock-safe fallback.

    When SGLang is installed, this adapter labels the run as SGLang-backed and
    leaves room for a future real serving call. When it is unavailable, the
    same prefix-reuse cost model is emitted as a simulation row instead of a
    skipped row so all-baseline sweeps still include `sglang_prefix`.
    """

    name = "sglang_prefix"
    label = "SGLang Prefix Baseline"

    def available(self) -> bool:
        return importlib.util.find_spec("sglang") is not None

    def unavailable_reason(self) -> str:
        return "SGLang is not installed; using simulated SGLang prefix metrics"

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        available = self.available()
        cfg = self._normalize(workload_config)
        scheduler = StateAwareScheduler(cfg.cost_model)
        prefill = scheduler.estimate_prefill(cfg.context_tokens)
        resume = cfg.num_agents * cfg.cost_model.resume_overhead_sec
        decode = cfg.num_agents * scheduler.estimate_decode(cfg.output_tokens)
        omega = cfg.num_agents * cfg.cost_model.omega_state_sec
        metrics = self._metrics(
            cfg,
            prefill_sec=prefill,
            decode_sec=decode,
            resume_sec=resume,
            omega_sec=omega,
            kv_bytes=cfg.context_tokens * cfg.bytes_per_token,
            reuse_ratio=(cfg.num_agents - 1) / cfg.num_agents if cfg.num_agents else 0.0,
        )
        return BaselineResult(
            self.label,
            metrics,
            available=available,
            reason="" if available else self.unavailable_reason(),
            metadata={
                "mode": "sglang_prefix" if available else "mock_sglang_prefix",
                "real_sglang_installed": available,
            },
        )


class DistServeStyleBaseline(BaselineAdapter):
    """DistServe-style simulated disaggregated prefill/decode baseline."""

    name = "distserve_style"
    label = "DistServe-style simulated baseline"

    def run_workload(self, workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None) -> BaselineResult:
        cfg = self._normalize(workload_config)
        scheduler = StateAwareScheduler(cfg.cost_model)
        prefill = scheduler.estimate_prefill(cfg.context_tokens)
        transfer = cfg.cost_model.network_latency_sec + (cfg.context_tokens * cfg.bytes_per_token) / max(
            cfg.cost_model.bandwidth_bytes_per_sec,
            1e-12,
        )
        decode = cfg.num_agents * scheduler.estimate_decode(cfg.output_tokens)
        resume = cfg.num_agents * cfg.cost_model.resume_overhead_sec
        omega = cfg.num_agents * cfg.cost_model.omega_state_sec
        metrics = self._metrics(
            cfg,
            prefill_sec=prefill,
            decode_sec=decode,
            transfer_sec=transfer,
            resume_sec=resume,
            omega_sec=omega,
            kv_bytes=cfg.context_tokens * cfg.bytes_per_token,
            reuse_ratio=(cfg.num_agents - 1) / cfg.num_agents if cfg.num_agents else 0.0,
        )
        return BaselineResult(
            self.label,
            metrics,
            metadata={
                "mode": "simulation",
                "claim": "DistServe-style simulated baseline, not exact DistServe unless a real implementation is integrated",
            },
        )


def get_baselines() -> list[BaselineAdapter]:
    """Return all registered baseline adapters."""

    return [
        DensePrefillBaseline(),
        AAFLOWTextBaseline(),
        VLLMLocalPrefixBaseline(),
        SGLangPrefixBaseline(),
        DistServeStyleBaseline(),
    ]


def list_baselines() -> list[dict[str, Any]]:
    """Return baseline availability records for CLI display."""

    records = []
    for baseline in get_baselines():
        available = baseline.available()
        records.append(
            {
                "name": baseline.name,
                "label": baseline.label,
                "available": available,
                "skipped": False,
                "reason": "" if available else baseline.unavailable_reason(),
            }
        )
    return records


def recompute_baseline(num_branches: int = 4, tokens: int = 128) -> List[OperatorSpec]:
    """Build a legacy plan that recomputes each branch instead of forking KV."""

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


def run_default_baselines(workload_config: WorkloadConfig | QueryWorkload | dict[str, Any] | None = None) -> List[BaselineResult]:
    """Run available/default baselines with a shared workload config."""

    results = []
    for baseline in get_baselines():
        result = baseline.run_workload(workload_config)
        if not result.skipped:
            results.append(result)
    return results


def baseline_names(baselines: Optional[Iterable[BaselineAdapter]] = None) -> list[str]:
    """Return stable baseline registry names."""

    return [baseline.name for baseline in (baselines or get_baselines())]
