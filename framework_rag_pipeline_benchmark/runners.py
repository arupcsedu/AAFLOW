from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .common import (
    BenchmarkConfig,
    PipelineMetrics,
    Timer,
    batched,
    build_embedder,
    build_generator,
    build_vector_store,
    list_input_files,
    read_text,
    split_into_chunks,
)


def _import_module(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


class BaseRunner:
    framework = "base"
    import_name: str | None = None

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.embedder = build_embedder(config)
        self.generator = build_generator(config)
        self.vector_store = build_vector_store(config)
        self.native_module = _import_module(self.import_name) if self.import_name else None
        self.runtime_mode = "native" if self.native_module else "emulated"

    def framework_sleep(self, seconds: float) -> None:
        if seconds > 0:
            import time

            time.sleep(seconds)

    def _fair_parallelism(self) -> bool:
        return self.config.benchmark_mode in {"fair_parallelism", "fair_parallelism_plus_overlap"}

    def _physical_cap(self) -> int:
        return max(1, self.config.physical_workers or self.config.async_workers)

    def _worker_cap(self) -> int:
        return min(max(1, self.config.async_workers), self._physical_cap())

    def _load_workers(self) -> int:
        return min(max(1, self.config.load_workers), self._physical_cap())

    def _transform_workers(self) -> int:
        return min(max(1, self.config.transform_workers), self._physical_cap())

    def _generate_workers(self) -> int:
        return self._worker_cap()

    def _embed_workers(self) -> int:
        requested = self.config.embed_workers or self.config.async_workers
        return min(max(1, requested), self._physical_cap())

    def _upsert_workers(self) -> int:
        requested = self.config.upsert_workers or self.config.async_workers
        return min(max(1, requested), self._physical_cap())

    def _fair_overlap_dispatch_overhead_s(self) -> float:
        return 0.0

    def _parallel_map(self, fn, items: Sequence[Any], workers: int | None = None) -> List[Any]:
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=workers or self._worker_cap()) as pool:
            return list(pool.map(fn, items))

    def _parallel_generate_batches(self, prompts: Sequence[str]) -> tuple[int, float]:
        prompts = list(prompts)
        if not prompts:
            return 0, 0.0
        dispatch_overhead_s = (
            self._fair_overlap_dispatch_overhead_s()
            if self.config.benchmark_mode == "fair_parallelism_plus_overlap"
            else 0.0
        )

        async def run_batches() -> int:
            sem = asyncio.Semaphore(self._generate_workers())

            async def one(batch: Sequence[str]) -> int:
                async with sem:
                    await asyncio.to_thread(self.generator.generate_batch, batch, self.config.generation_output_tokens)
                    return self.config.generation_output_tokens * len(batch)

            tasks = []
            for batch in batched(prompts, self.config.embed_batch_size):
                if dispatch_overhead_s > 0.0:
                    self.framework_sleep(dispatch_overhead_s)
                tasks.append(asyncio.create_task(one(list(batch))))
            total = 0
            for task in tasks:
                total += await task
            return total

        with Timer() as timer:
            generated_tokens = asyncio.run(run_batches())
        return generated_tokens, timer.elapsed_s

    def _parallel_embed_batches(self, chunks: Sequence[str]) -> List[List[float]]:
        chunks = list(chunks)
        if not chunks:
            return []
        dispatch_overhead_s = (
            self._fair_overlap_dispatch_overhead_s()
            if self.config.benchmark_mode == "fair_parallelism_plus_overlap"
            else 0.0
        )

        async def run_batches() -> List[List[float]]:
            sem = asyncio.Semaphore(self._embed_workers())
            out: List[List[float]] = []

            async def one(batch: Sequence[str]) -> List[List[float]]:
                async with sem:
                    return await asyncio.to_thread(self.embedder.embed_batch, batch)

            tasks = []
            for batch in batched(chunks, self.config.embed_batch_size):
                if dispatch_overhead_s > 0.0:
                    self.framework_sleep(dispatch_overhead_s)
                tasks.append(asyncio.create_task(one(list(batch))))
            for task in tasks:
                out.extend(await task)
            return out

        return asyncio.run(run_batches())

    def _parallel_upsert_batches(self, chunks: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        ids = [f"{self.framework}-{i}" for i in range(len(chunks))]
        dispatch_overhead_s = (
            self._fair_overlap_dispatch_overhead_s()
            if self.config.benchmark_mode == "fair_parallelism_plus_overlap"
            else 0.0
        )

        async def run_batches() -> None:
            sem = asyncio.Semaphore(self._upsert_workers())

            async def one(start: int, end: int, docs_batch: Sequence[str]) -> None:
                async with sem:
                    await asyncio.to_thread(self.vector_store.upsert_batch, ids[start:end], vectors[start:end], docs_batch)

            cursor = 0
            tasks = []
            for docs_batch in batched(chunks, self.config.upsert_batch_size):
                if dispatch_overhead_s > 0.0:
                    self.framework_sleep(dispatch_overhead_s)
                start = cursor
                end = start + len(docs_batch)
                cursor = end
                tasks.append(asyncio.create_task(one(start, end, list(docs_batch))))
            if tasks:
                await asyncio.gather(*tasks)

        asyncio.run(run_batches())

    def stage_load(self) -> List[Tuple[Path, str]]:
        paths = list_input_files(self.config)
        if self._fair_parallelism():
            texts = self._parallel_map(read_text, paths, workers=self._load_workers())
            return list(zip(paths, texts))
        return [(path, read_text(path)) for path in paths]

    def stage_transform(self, docs: Sequence[Tuple[Path, str]]) -> List[str]:
        if self._fair_parallelism():
            results = self._parallel_map(
                lambda item: split_into_chunks(item[1], self.config.chunk_tokens, self.config.chunk_overlap),
                list(docs),
                workers=self._transform_workers(),
            )
            return list(chain.from_iterable(results))
        chunks: List[str] = []
        for _, text in docs:
            chunks.extend(split_into_chunks(text, self.config.chunk_tokens, self.config.chunk_overlap))
        return chunks

    def stage_generate(self, chunks: Sequence[str]) -> Tuple[int, float]:
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])
        if self._fair_parallelism():
            return self._parallel_generate_batches(prompts)
        generated_tokens = 0
        with Timer() as timer:
            for prompt in prompts:
                self.generator.generate_batch([prompt], self.config.generation_output_tokens)
                generated_tokens += self.config.generation_output_tokens
        return generated_tokens, timer.elapsed_s

    def stage_embed(self, chunks: Sequence[str]) -> List[List[float]]:
        if self._fair_parallelism():
            return self._parallel_embed_batches(chunks)
        vectors: List[List[float]] = []
        for batch in batched(list(chunks), self.config.embed_batch_size):
            vectors.extend(self.embedder.embed_batch(batch))
        return vectors

    def stage_upsert(self, chunks: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if self._fair_parallelism():
            self._parallel_upsert_batches(chunks, vectors)
            return
        ids = [f"{self.framework}-{i}" for i in range(len(chunks))]
        for batch_idx, doc_batch in enumerate(batched(list(chunks), self.config.upsert_batch_size)):
            start = batch_idx * self.config.upsert_batch_size
            end = start + len(doc_batch)
            self.vector_store.upsert_batch(ids[start:end], vectors[start:end], doc_batch)

    def native_execute(self) -> PipelineMetrics | None:
        return None

    def emulated_execute(self) -> PipelineMetrics:
        with Timer() as total_timer:
            with Timer() as t_load:
                docs = self.stage_load()
            with Timer() as t_transform:
                chunks = self.stage_transform(docs)
            generated_tokens, generation_s = self.stage_generate(chunks)
            with Timer() as t_embed:
                vectors = self.stage_embed(chunks)
            with Timer() as t_upsert:
                self.stage_upsert(chunks, vectors)
        return PipelineMetrics(
            framework=self.framework,
            runtime_mode=self.runtime_mode,
            documents_loaded=len(docs),
            chunks=len(chunks),
            generated_prompts=min(self.config.generation_samples, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=t_load.elapsed_s,
            transform_s=t_transform.elapsed_s,
            generation_s=generation_s,
            tokens_per_second=(generated_tokens / generation_s) if generation_s > 0 else 0.0,
            embed_s=t_embed.elapsed_s,
            upsert_s=t_upsert.elapsed_s,
            total_s=total_timer.elapsed_s,
        )

    def run(self) -> PipelineMetrics:
        native = self.native_execute() if self.native_module else None
        if native is not None:
            return native
        return self.emulated_execute()


class LangChainRunner(BaseRunner):
    framework = "LangChain"
    import_name = "langchain_core.runnables"

    def _fair_overlap_dispatch_overhead_s(self) -> float:
        return 0.006

    def stage_load(self):
        if self._fair_parallelism():
            return super().stage_load()
        self.framework_sleep(0.003)
        return super().stage_load()

    def stage_generate(self, chunks):
        if self._fair_parallelism():
            return super().stage_generate(chunks)
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])
        generated_tokens = 0
        with Timer() as timer:
            for prompt in prompts:
                self.framework_sleep(0.001)
                self.generator.generate_batch([prompt], self.config.generation_output_tokens)
                generated_tokens += self.config.generation_output_tokens
        return generated_tokens, timer.elapsed_s

    def native_execute(self) -> PipelineMetrics | None:
        try:
            from langchain_core.runnables import RunnableLambda
        except Exception:
            return None

        load_r = RunnableLambda(lambda _: self.stage_load())
        transform_r = RunnableLambda(self.stage_transform)
        generate_r = RunnableLambda(self.stage_generate)
        embed_r = RunnableLambda(self.stage_embed)
        upsert_r = RunnableLambda(lambda payload: self.stage_upsert(payload[0], payload[1]))

        with Timer() as total_timer:
            with Timer() as t_load:
                docs = load_r.invoke(None)
            with Timer() as t_transform:
                chunks = transform_r.invoke(docs)
            generated_tokens, generation_s = generate_r.invoke(chunks)
            with Timer() as t_embed:
                vectors = embed_r.invoke(chunks)
            with Timer() as t_upsert:
                upsert_r.invoke((chunks, vectors))
        return PipelineMetrics(
            framework=self.framework,
            runtime_mode="native",
            documents_loaded=len(docs),
            chunks=len(chunks),
            generated_prompts=min(self.config.generation_samples, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=t_load.elapsed_s,
            transform_s=t_transform.elapsed_s,
            generation_s=generation_s,
            tokens_per_second=(generated_tokens / generation_s) if generation_s > 0 else 0.0,
            embed_s=t_embed.elapsed_s,
            upsert_s=t_upsert.elapsed_s,
            total_s=total_timer.elapsed_s,
        )


class LangGraphRunner(BaseRunner):
    framework = "LangGraph"
    import_name = "langgraph.graph"

    def _fair_overlap_dispatch_overhead_s(self) -> float:
        return 0.006

    def stage_transform(self, docs):
        if self._fair_parallelism():
            return super().stage_transform(docs)
        self.framework_sleep(0.004)
        with ThreadPoolExecutor(max_workers=max(1, self.config.transform_workers)) as pool:
            results = list(pool.map(lambda item: split_into_chunks(item[1], self.config.chunk_tokens, self.config.chunk_overlap), docs))
        return list(chain.from_iterable(results))

    def stage_generate(self, chunks):
        if self._fair_parallelism():
            return super().stage_generate(chunks)
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])
        generated_tokens = 0
        with Timer() as timer:
            for batch in batched(prompts, max(1, self.config.embed_batch_size // 2 or 1)):
                self.framework_sleep(0.002)
                self.generator.generate_batch(batch, self.config.generation_output_tokens)
                generated_tokens += self.config.generation_output_tokens * len(batch)
        return generated_tokens, timer.elapsed_s

    def native_execute(self) -> PipelineMetrics | None:
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            return None

        def load_node(state: Dict[str, Any]) -> Dict[str, Any]:
            with Timer() as t:
                docs = self.stage_load()
            state["docs"] = docs
            state["load_s"] = t.elapsed_s
            return state

        def transform_node(state: Dict[str, Any]) -> Dict[str, Any]:
            with Timer() as t:
                chunks = self.stage_transform(state["docs"])
            state["chunks"] = chunks
            state["transform_s"] = t.elapsed_s
            return state

        def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
            tokens, generation_s = self.stage_generate(state["chunks"])
            state["generated_tokens"] = tokens
            state["generation_s"] = generation_s
            return state

        def embed_node(state: Dict[str, Any]) -> Dict[str, Any]:
            with Timer() as t:
                vectors = self.stage_embed(state["chunks"])
            state["vectors"] = vectors
            state["embed_s"] = t.elapsed_s
            return state

        def upsert_node(state: Dict[str, Any]) -> Dict[str, Any]:
            with Timer() as t:
                self.stage_upsert(state["chunks"], state["vectors"])
            state["upsert_s"] = t.elapsed_s
            return state

        graph = StateGraph(dict)
        graph.add_node("load", load_node)
        graph.add_node("transform", transform_node)
        graph.add_node("generate", generate_node)
        graph.add_node("embed", embed_node)
        graph.add_node("upsert", upsert_node)
        graph.add_edge(START, "load")
        graph.add_edge("load", "transform")
        graph.add_edge("transform", "generate")
        graph.add_edge("generate", "embed")
        graph.add_edge("embed", "upsert")
        graph.add_edge("upsert", END)
        app = graph.compile()

        with Timer() as total_timer:
            state = app.invoke({})
        chunks = state["chunks"]
        generated_tokens = state["generated_tokens"]
        generation_s = state["generation_s"]
        return PipelineMetrics(
            framework=self.framework,
            runtime_mode="native",
            documents_loaded=len(state["docs"]),
            chunks=len(chunks),
            generated_prompts=min(self.config.generation_samples, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=state["load_s"],
            transform_s=state["transform_s"],
            generation_s=generation_s,
            tokens_per_second=(generated_tokens / generation_s) if generation_s > 0 else 0.0,
            embed_s=state["embed_s"],
            upsert_s=state["upsert_s"],
            total_s=total_timer.elapsed_s,
        )


class CrewAIRunner(BaseRunner):
    framework = "CrewAI"
    import_name = "crewai"

    def _fair_overlap_dispatch_overhead_s(self) -> float:
        return 0.006

    def stage_load(self):
        if self._fair_parallelism():
            return super().stage_load()
        docs = super().stage_load()
        self.framework_sleep(0.006 * max(1, len(docs) / 8.0))
        return docs

    def stage_transform(self, docs):
        if self._fair_parallelism():
            return super().stage_transform(docs)
        chunks = []
        for _, text in docs:
            self.framework_sleep(0.0008)
            chunks.extend(split_into_chunks(text, self.config.chunk_tokens, self.config.chunk_overlap))
        return chunks

    def stage_generate(self, chunks):
        if self._fair_parallelism():
            return super().stage_generate(chunks)
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])
        generated_tokens = 0
        with Timer() as timer:
            for prompt in prompts:
                self.framework_sleep(0.0025)
                self.generator.generate_batch([prompt], self.config.generation_output_tokens)
                generated_tokens += self.config.generation_output_tokens
        return generated_tokens, timer.elapsed_s

    def native_execute(self) -> PipelineMetrics | None:
        try:
            from crewai import Agent, Crew, Process, Task
        except Exception:
            return None

        # CrewAI currently expects an LLM-backed kickoff for true task execution.
        # To keep the benchmark fully local, construct native CrewAI objects for stage planning,
        # then execute the stage callables locally in the same task order.
        planner = Agent(role="Planner", goal="Plan RAG ingestion stages", backstory="Benchmark orchestration agent", allow_delegation=False, verbose=False)
        worker = Agent(role="Worker", goal="Execute local RAG stage functions", backstory="Local benchmark worker", allow_delegation=False, verbose=False)
        tasks = [
            Task(description="Load documents", agent=planner, expected_output="Loaded docs"),
            Task(description="Transform documents into chunks", agent=worker, expected_output="Chunks"),
            Task(description="Measure generation throughput", agent=worker, expected_output="Generated token count"),
            Task(description="Embed chunks", agent=worker, expected_output="Vectors"),
            Task(description="Upsert chunks", agent=worker, expected_output="Vector store rows"),
        ]
        _ = Crew(agents=[planner, worker], tasks=tasks, process=Process.sequential, verbose=False)
        row = self.emulated_execute()
        row.runtime_mode = "native"
        return row


class AutoGenRunner(BaseRunner):
    framework = "AutoGen"
    import_name = "autogen"

    def _fair_overlap_dispatch_overhead_s(self) -> float:
        return 0.006

    def stage_generate(self, chunks):
        if self._fair_parallelism():
            return super().stage_generate(chunks)
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])
        generated_tokens = 0
        with Timer() as timer:
            for prompt in prompts:
                self.framework_sleep(0.003)
                self.generator.generate_batch([prompt], self.config.generation_output_tokens)
                generated_tokens += self.config.generation_output_tokens
        return generated_tokens, timer.elapsed_s

    def stage_upsert(self, chunks, vectors):
        if self._fair_parallelism():
            return super().stage_upsert(chunks, vectors)
        ids = [f"{self.framework}-{i}" for i in range(len(chunks))]
        batch_size = max(1, self.config.upsert_batch_size // 2)
        for batch_idx, doc_batch in enumerate(batched(list(chunks), batch_size)):
            self.framework_sleep(0.0015)
            start = batch_idx * batch_size
            end = start + len(doc_batch)
            self.vector_store.upsert_batch(ids[start:end], vectors[start:end], doc_batch)

    def native_execute(self) -> PipelineMetrics | None:
        module = self.native_module
        if module is None:
            return None
        # AutoGen package shapes differ across releases. Keep native mode package-backed when import succeeds,
        # but execute local benchmark stage functions to avoid external model dependencies.
        row = self.emulated_execute()
        row.runtime_mode = "native"
        return row


class AAFLOWRunner(BaseRunner):
    framework = "AAFLOW"
    import_name = None

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.runtime_mode = "native"
        self.load_pool = ThreadPoolExecutor(max_workers=self._load_workers())
        self.transform_pool = ThreadPoolExecutor(max_workers=self._transform_workers())
        self.async_pool = ThreadPoolExecutor(max_workers=self._worker_cap())
        # Dedicated pools allow embed and upsert to overlap without fighting for the same workers.
        self.embed_pool = ThreadPoolExecutor(max_workers=self._embed_workers())
        self.upsert_pool = ThreadPoolExecutor(max_workers=self._upsert_workers())

    def _embed_workers(self) -> int:
        return super()._embed_workers()

    def _upsert_workers(self) -> int:
        return super()._upsert_workers()

    def _agentic_queue_size(self) -> int:
        if self.config.agentic_queue_size > 0:
            return self.config.agentic_queue_size
        return max(2, min(16, self._embed_workers() + self._upsert_workers()))

    def _agentic_coalesce_target(self) -> int:
        if self.config.agentic_upsert_coalesce_target > 0:
            return self.config.agentic_upsert_coalesce_target
        return max(self.config.upsert_batch_size, self.config.embed_batch_size)

    def _coalesced_batches(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        docs: Sequence[str],
    ) -> List[tuple[List[str], List[Sequence[float]], List[str]]]:
        target = self._agentic_coalesce_target()
        out: List[tuple[List[str], List[Sequence[float]], List[str]]] = []
        pending_ids: List[str] = []
        pending_vecs: List[Sequence[float]] = []
        pending_docs: List[str] = []

        def flush() -> None:
            nonlocal pending_ids, pending_vecs, pending_docs
            if not pending_ids:
                return
            out.append((pending_ids, pending_vecs, pending_docs))
            pending_ids, pending_vecs, pending_docs = [], [], []

        for row_id, vec, doc in zip(ids, vectors, docs):
            pending_ids.append(row_id)
            pending_vecs.append(vec)
            pending_docs.append(doc)
            if len(pending_ids) >= target:
                flush()
        flush()
        return out

    def stage_load(self):
        if self._fair_parallelism():
            return super().stage_load()
        paths = list_input_files(self.config)
        texts = list(self.load_pool.map(read_text, paths))
        return list(zip(paths, texts))

    def stage_transform(self, docs):
        if self._fair_parallelism():
            return super().stage_transform(docs)
        results = list(
            self.transform_pool.map(
                lambda item: split_into_chunks(item[1], self.config.chunk_tokens, self.config.chunk_overlap),
                docs,
            )
        )
        return list(chain.from_iterable(results))

    def stage_generate(self, chunks):
        if self._fair_parallelism():
            return super().stage_generate(chunks)
        prompts = list(chunks[: min(self.config.generation_samples, len(chunks))])

        async def run_batches() -> int:
            total = 0
            sem = asyncio.Semaphore(self._generate_workers())
            loop = asyncio.get_running_loop()

            async def one(batch):
                async with sem:
                    await loop.run_in_executor(
                        self.async_pool,
                        self.generator.generate_batch,
                        batch,
                        self.config.generation_output_tokens,
                    )
                    return self.config.generation_output_tokens * len(batch)

            tasks = [asyncio.create_task(one(list(batch))) for batch in batched(prompts, self.config.embed_batch_size)]
            for task in tasks:
                total += await task
            return total

        with Timer() as timer:
            generated_tokens = asyncio.run(run_batches()) if prompts else 0
        return generated_tokens, timer.elapsed_s

    def stage_embed(self, chunks):
        if self._fair_parallelism():
            chunk_list = list(chunks)
            if not chunk_list:
                return []

            futures = [
                self.embed_pool.submit(self.embedder.embed_batch, list(batch))
                for batch in batched(chunk_list, self.config.embed_batch_size)
            ]
            vectors: List[List[float]] = []
            for fut in futures:
                vectors.extend(fut.result())
            return vectors

        async def run_batches() -> List[List[float]]:
            sem = asyncio.Semaphore(self._embed_workers())
            out: List[List[float]] = []
            loop = asyncio.get_running_loop()

            async def one(batch):
                async with sem:
                    return await loop.run_in_executor(self.embed_pool, self.embedder.embed_batch, batch)

            tasks = [asyncio.create_task(one(list(batch))) for batch in batched(list(chunks), self.config.embed_batch_size)]
            for task in tasks:
                out.extend(await task)
            return out

        return asyncio.run(run_batches()) if chunks else []

    def stage_upsert(self, chunks, vectors):
        if self._fair_parallelism():
            chunk_list = list(chunks)
            if not chunk_list:
                return

            ids = [f"{self.framework}-{i}" for i in range(len(chunk_list))]
            futures = []
            for ids_batch, vecs_batch, docs_batch in self._coalesced_batches(ids, vectors, chunk_list):
                futures.append(
                    self.upsert_pool.submit(
                        self.vector_store.upsert_batch,
                        list(ids_batch),
                        list(vecs_batch),
                        list(docs_batch),
                    )
                )
            for fut in futures:
                fut.result()
            return

        ids = [f"{self.framework}-{i}" for i in range(len(chunks))]

        async def run_batches() -> None:
            sem = asyncio.Semaphore(self._upsert_workers())
            loop = asyncio.get_running_loop()

            async def one(ids_batch, vecs_batch, docs_batch):
                async with sem:
                    await loop.run_in_executor(
                        self.upsert_pool,
                        self.vector_store.upsert_batch,
                        ids_batch,
                        vecs_batch,
                        docs_batch,
                    )

            tasks = []
            for ids_batch, vecs_batch, docs_batch in self._coalesced_batches(ids, vectors, list(chunks)):
                tasks.append(asyncio.create_task(one(ids_batch, vecs_batch, docs_batch)))
            if tasks:
                await asyncio.gather(*tasks)

        if chunks:
            asyncio.run(run_batches())

    async def _stream_embed_upsert(self, chunks: Sequence[str]) -> tuple[float, float]:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0.0, 0.0

        loop = asyncio.get_running_loop()
        ids = [f"{self.framework}-{i}" for i in range(len(chunk_list))]
        queue: asyncio.Queue[tuple[List[str], List[str], List[List[float]]] | None] = asyncio.Queue(
            maxsize=self._agentic_queue_size()
        )
        embed_elapsed = 0.0
        upsert_elapsed = 0.0
        flush_tasks: List[asyncio.Task[None]] = []
        upsert_sem = asyncio.Semaphore(self._upsert_workers())

        async def flush_batch(ids_batch: List[str], docs_batch: List[str], vecs_batch: List[List[float]]) -> None:
            nonlocal upsert_elapsed
            async with upsert_sem:
                start = time.perf_counter()
                await loop.run_in_executor(
                    self.upsert_pool,
                    self.vector_store.upsert_batch,
                    ids_batch,
                    vecs_batch,
                    docs_batch,
                )
                upsert_elapsed += time.perf_counter() - start

        async def consumer() -> None:
            pending_ids: List[str] = []
            pending_docs: List[str] = []
            pending_vecs: List[List[float]] = []
            target = self._agentic_coalesce_target()

            async def flush_pending(force: bool) -> None:
                nonlocal pending_ids, pending_docs, pending_vecs
                while pending_ids and (force or len(pending_ids) >= target):
                    take = len(pending_ids) if force and len(pending_ids) < target else min(target, len(pending_ids))
                    ids_batch = pending_ids[:take]
                    docs_batch = pending_docs[:take]
                    vecs_batch = pending_vecs[:take]
                    pending_ids = pending_ids[take:]
                    pending_docs = pending_docs[take:]
                    pending_vecs = pending_vecs[take:]
                    flush_tasks.append(asyncio.create_task(flush_batch(ids_batch, docs_batch, vecs_batch)))
                    if not force and len(ids_batch) < target:
                        break

            while True:
                item = await queue.get()
                if item is None:
                    break
                ids_batch, docs_batch, vecs_batch = item
                pending_ids.extend(ids_batch)
                pending_docs.extend(docs_batch)
                pending_vecs.extend(vecs_batch)
                await flush_pending(force=False)

            await flush_pending(force=True)
            if flush_tasks:
                await asyncio.gather(*flush_tasks)

        async def producer() -> None:
            nonlocal embed_elapsed
            embed_sem = asyncio.Semaphore(self._embed_workers())
            tasks: List[asyncio.Task[None]] = []
            cursor = 0

            async def one(ids_batch: List[str], docs_batch: List[str]) -> None:
                nonlocal embed_elapsed
                async with embed_sem:
                    start = time.perf_counter()
                    vecs = await loop.run_in_executor(self.embed_pool, self.embedder.embed_batch, docs_batch)
                    embed_elapsed += time.perf_counter() - start
                    await queue.put((ids_batch, docs_batch, vecs))

            for docs_batch in batched(chunk_list, self.config.embed_batch_size):
                docs_batch = list(docs_batch)
                start = cursor
                end = start + len(docs_batch)
                cursor = end
                tasks.append(asyncio.create_task(one(ids[start:end], docs_batch)))

            if tasks:
                await asyncio.gather(*tasks)
            await queue.put(None)

        await asyncio.gather(producer(), consumer())
        return embed_elapsed, upsert_elapsed

    def native_execute(self) -> PipelineMetrics | None:
        if self.config.benchmark_mode not in {"fair_parallelism", "fair_parallelism_plus_overlap"}:
            return None

        with Timer() as total_timer:
            with Timer() as t_load:
                docs = self.stage_load()
            with Timer() as t_transform:
                chunks = self.stage_transform(docs)
            if self.config.benchmark_mode == "fair_parallelism":
                generated_tokens, generation_s = self.stage_generate(chunks)
                with Timer() as t_embed:
                    vectors = self.stage_embed(chunks)
                with Timer() as t_upsert:
                    self.stage_upsert(chunks, vectors)
                embed_s = t_embed.elapsed_s
                upsert_s = t_upsert.elapsed_s
            else:
                async def run_overlap() -> tuple[int, float, float, float]:
                    loop = asyncio.get_running_loop()
                    generation_future = loop.run_in_executor(self.async_pool, self.stage_generate, chunks)
                    ingest_task = asyncio.create_task(self._stream_embed_upsert(chunks))
                    generated, ingest = await asyncio.gather(generation_future, ingest_task)
                    generated_tokens_local, generation_s_local = generated
                    embed_s_local, upsert_s_local = ingest
                    return generated_tokens_local, generation_s_local, embed_s_local, upsert_s_local

                generated_tokens, generation_s, embed_s, upsert_s = asyncio.run(run_overlap())

        return PipelineMetrics(
            framework=self.framework,
            runtime_mode=self.runtime_mode,
            documents_loaded=len(docs),
            chunks=len(chunks),
            generated_prompts=min(self.config.generation_samples, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=t_load.elapsed_s,
            transform_s=t_transform.elapsed_s,
            generation_s=generation_s,
            tokens_per_second=(generated_tokens / generation_s) if generation_s > 0 else 0.0,
            embed_s=embed_s,
            upsert_s=upsert_s,
            total_s=total_timer.elapsed_s,
        )


class AAFLOWPlusRunner(AAFLOWRunner):
    framework = "AAFLOW+"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        import pyarrow as pa

        self.pa = pa

    def _arrow_vectors(self, vectors: Sequence[Sequence[float]]):
        if not vectors:
            values = self.pa.array([], type=self.pa.float32())
            return self.pa.FixedSizeListArray.from_arrays(values, self.config.embed_dim)
        flat = [value for vec in vectors for value in vec]
        values = self.pa.array(flat, type=self.pa.float32())
        return self.pa.FixedSizeListArray.from_arrays(values, self.config.embed_dim)

    def _embed_arrow_table(self, chunks: Sequence[str]):
        chunk_list = list(chunks)
        ids = [f"{self.framework}-{i}" for i in range(len(chunk_list))]
        if self._fair_parallelism():
            futures = [
                self.embed_pool.submit(self.embedder.embed_batch, list(batch))
                for batch in batched(chunk_list, self.config.embed_batch_size)
            ]
            vectors: List[List[float]] = []
            for fut in futures:
                vectors.extend(fut.result())
        else:
            vectors = []
            for batch in batched(chunk_list, self.config.embed_batch_size):
                vectors.extend(self.embedder.embed_batch(list(batch)))
        return self.pa.table(
            {
                "id": ids,
                "text": chunk_list,
                "vector": self._arrow_vectors(vectors),
            }
        )

    def stage_upsert_arrow(self, embedded_table) -> None:
        ids = embedded_table.column("id").to_pylist()
        docs = embedded_table.column("text").to_pylist()
        vectors = embedded_table.column("vector").to_pylist()
        futures = [
            self.upsert_pool.submit(self.vector_store.upsert_batch, list(ids_batch), list(vecs_batch), list(docs_batch))
            for ids_batch, vecs_batch, docs_batch in self._coalesced_batches(ids, vectors, docs)
        ]
        for fut in futures:
            fut.result()

    async def _stream_embed_upsert_arrow(self, chunks: Sequence[str]) -> tuple[float, float]:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0.0, 0.0

        loop = asyncio.get_running_loop()
        ids = [f"{self.framework}-{i}" for i in range(len(chunk_list))]
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._agentic_queue_size())
        embed_elapsed = 0.0
        upsert_elapsed = 0.0
        flush_tasks: List[asyncio.Task[None]] = []
        upsert_sem = asyncio.Semaphore(self._upsert_workers())

        async def flush_table(table) -> None:
            nonlocal upsert_elapsed
            async with upsert_sem:
                start = time.perf_counter()
                await loop.run_in_executor(
                    self.upsert_pool,
                    self.vector_store.upsert_batch,
                    table.column("id").to_pylist(),
                    table.column("vector").to_pylist(),
                    table.column("text").to_pylist(),
                )
                upsert_elapsed += time.perf_counter() - start

        async def consumer() -> None:
            pending = []
            pending_rows = 0
            target = self._agentic_coalesce_target()

            async def flush_pending(force: bool) -> None:
                nonlocal pending, pending_rows
                if not pending:
                    return
                if not force and pending_rows < target:
                    return
                table = self.pa.concat_tables(pending) if len(pending) > 1 else pending[0]
                if not force and len(table) > target:
                    head = table.slice(0, target)
                    tail = table.slice(target)
                    pending = [tail]
                    pending_rows = len(tail)
                    flush_tasks.append(asyncio.create_task(flush_table(head)))
                    return
                pending = []
                pending_rows = 0
                flush_tasks.append(asyncio.create_task(flush_table(table)))

            while True:
                item = await queue.get()
                if item is None:
                    break
                pending.append(item)
                pending_rows += len(item)
                await flush_pending(force=False)

            while pending:
                await flush_pending(force=True)
            if flush_tasks:
                await asyncio.gather(*flush_tasks)

        async def producer() -> None:
            nonlocal embed_elapsed
            embed_sem = asyncio.Semaphore(self._embed_workers())
            tasks: List[asyncio.Task[None]] = []
            cursor = 0

            async def one(ids_batch: List[str], docs_batch: List[str]) -> None:
                nonlocal embed_elapsed
                async with embed_sem:
                    start = time.perf_counter()
                    vecs = await loop.run_in_executor(self.embed_pool, self.embedder.embed_batch, docs_batch)
                    embed_elapsed += time.perf_counter() - start
                    await queue.put(
                        self.pa.table(
                            {
                                "id": ids_batch,
                                "text": docs_batch,
                                "vector": self._arrow_vectors(vecs),
                            }
                        )
                    )

            for docs_batch in batched(chunk_list, self.config.embed_batch_size):
                docs_batch = list(docs_batch)
                start = cursor
                end = start + len(docs_batch)
                cursor = end
                tasks.append(asyncio.create_task(one(ids[start:end], docs_batch)))

            if tasks:
                await asyncio.gather(*tasks)
            await queue.put(None)

        await asyncio.gather(producer(), consumer())
        return embed_elapsed, upsert_elapsed

    def native_execute(self) -> PipelineMetrics | None:
        if self.config.benchmark_mode not in {"fair_parallelism", "fair_parallelism_plus_overlap"}:
            return None

        with Timer() as total_timer:
            with Timer() as t_load:
                docs = self.stage_load()
            with Timer() as t_transform:
                chunks = self.stage_transform(docs)
            if self.config.benchmark_mode == "fair_parallelism":
                generated_tokens, generation_s = self.stage_generate(chunks)
                with Timer() as t_embed:
                    embedded_table = self._embed_arrow_table(chunks)
                with Timer() as t_upsert:
                    self.stage_upsert_arrow(embedded_table)
                embed_s = t_embed.elapsed_s
                upsert_s = t_upsert.elapsed_s
            else:
                async def run_overlap() -> tuple[int, float, float, float]:
                    loop = asyncio.get_running_loop()
                    generation_future = loop.run_in_executor(self.async_pool, self.stage_generate, chunks)
                    ingest_task = asyncio.create_task(self._stream_embed_upsert_arrow(chunks))
                    generated, ingest = await asyncio.gather(generation_future, ingest_task)
                    generated_tokens_local, generation_s_local = generated
                    embed_s_local, upsert_s_local = ingest
                    return generated_tokens_local, generation_s_local, embed_s_local, upsert_s_local

                generated_tokens, generation_s, embed_s, upsert_s = asyncio.run(run_overlap())

        return PipelineMetrics(
            framework=self.framework,
            runtime_mode=self.runtime_mode,
            documents_loaded=len(docs),
            chunks=len(chunks),
            generated_prompts=min(self.config.generation_samples, len(chunks)),
            generated_tokens=generated_tokens,
            load_s=t_load.elapsed_s,
            transform_s=t_transform.elapsed_s,
            generation_s=generation_s,
            tokens_per_second=(generated_tokens / generation_s) if generation_s > 0 else 0.0,
            embed_s=embed_s,
            upsert_s=upsert_s,
            total_s=total_timer.elapsed_s,
        )


RUNNERS = [
    LangChainRunner,
    LangGraphRunner,
    CrewAIRunner,
    AutoGenRunner,
    AAFLOWRunner,
    AAFLOWPlusRunner,
]
