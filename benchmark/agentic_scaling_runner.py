#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_BENCH = SCRIPT_DIR / 'benchmark_configs_1_to_5.py'
SCALING_CONFIG_DIR = SCRIPT_DIR / 'scaling_configs'


def _tiered_choice(workers: int, tiers: List[tuple[int, int]], default: int) -> int:
    chosen = default
    for threshold, value in tiers:
        if workers >= threshold:
            chosen = value
    return chosen




def _config_path(profile: str, mode: str, physical_workers: int) -> Path:
    safe_mode = mode.replace('-', '_')
    return SCALING_CONFIG_DIR / f"{profile}_{safe_mode}_{physical_workers}w.json"


def _config_inputs(args: argparse.Namespace) -> Dict[str, int | float | str | bool]:
    if args.set5_sink_backend != 'faiss':
        agentic_embed_batch = embed_batch
        agentic_upsert_batch = upsert_batch

    return {
        'profile': args.profile,
        'mode': getattr(args, 'mode', 'agentic'),
        'physical_workers': args.physical_workers,
        'async_workers': args.async_workers,
        'stage_cpu_cap': args.stage_cpu_cap,
        'base_nodes': args.base_nodes,
        'base_files': args.base_files,
        'base_nodes_per_worker': args.base_nodes_per_worker,
        'node_chars': args.node_chars,
        'chunks_per_file': args.chunks_per_file,
        'dim': args.dim,
        'embedder_backend': args.embedder_backend,
        'set5_sink_backend': args.set5_sink_backend,
    }


def resolve_tuned_config(args: argparse.Namespace) -> tuple[Dict[str, int | float | str | bool], Path]:
    SCALING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = _config_path(args.profile, getattr(args, 'mode', 'agentic'), args.physical_workers)
    current_inputs = _config_inputs(args)
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        tuned_saved = saved.get('tuned')
        if (
            saved.get('inputs') == current_inputs
            and isinstance(tuned_saved, dict)
            and 'agentic_embed_batch' in tuned_saved
            and 'agentic_upsert_batch' in tuned_saved
        ):
            return tuned_saved, config_path
    tuned = build_tuned_config(args)
    payload = {
        'inputs': current_inputs,
        'tuned': tuned,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return tuned, config_path

def build_tuned_config(args: argparse.Namespace) -> Dict[str, int | float | str | bool]:
    logical_workers = max(1, args.async_workers)
    physical_cpus = max(1, args.stage_cpu_cap or 1)

    if args.profile == 'strong':
        nodes = args.base_nodes
        files = max(1, args.base_files)
    else:
        nodes = max(1, args.base_nodes_per_worker * logical_workers)
        files = max(1, int(math.ceil(nodes / max(1, args.chunks_per_file))))

    files = max(files, int(math.ceil(nodes / max(1, args.chunks_per_file))))

    load_workers = min(physical_cpus, _tiered_choice(logical_workers, [(64, 8), (256, 12), (1024, 16)], physical_cpus))
    transform_workers = load_workers
    embed_workers = min(physical_cpus, _tiered_choice(logical_workers, [(64, 8), (256, 12), (1024, 16)], physical_cpus))

    if args.set5_sink_backend == 'faiss':
        embed_batch = _tiered_choice(
            logical_workers,
            [(64, 512), (128, 1024), (256, 2048), (512, 4096), (1024, 8192), (2048, 12288), (4096, 16384)],
            512,
        )
        upsert_batch = _tiered_choice(
            logical_workers,
            [(64, 2048), (128, 4096), (256, 8192), (512, 12288), (1024, 16384), (2048, 24576), (4096, 32768)],
            2048,
        )
        if logical_workers >= 1024:
            agentic_embed_batch = embed_batch
            agentic_upsert_batch = upsert_batch
        else:
            agentic_embed_batch = embed_batch * 2
            agentic_upsert_batch = upsert_batch * 2
        # In weak-scaling FAISS runs at 256w, the larger Agentic embed batch
        # regressed embed time relative to Higress. Keep Agentic's embed batch
        # aligned with the shared thin baseline there, while preserving the
        # larger upsert batch that still helps amortize sink cost.
        if args.profile == 'weak' and logical_workers <= 256:
            agentic_embed_batch = embed_batch
        upsert_workers = 1
        upsert_shards = 1
        upsert_timeout_ms = 2.0
        coalesce_multiplier = 1
    else:
        embed_batch = _tiered_choice(
            logical_workers,
            [(64, 256), (128, 512), (256, 1024), (512, 2048), (1024, 4096), (2048, 6144), (4096, 8192)],
            256,
        )
        upsert_batch = _tiered_choice(
            logical_workers,
            [(64, 4096), (128, 8192), (256, 12288), (512, 16384), (1024, 24576), (2048, 32768), (4096, 49152)],
            4096,
        )
        upsert_workers = 1
        upsert_shards = min(8, physical_cpus)
        upsert_timeout_ms = _tiered_choice(logical_workers, [(512, 12), (1024, 16), (2048, 20), (4096, 24)], 8)
        coalesce_multiplier = _tiered_choice(logical_workers, [(128, 64), (512, 96), (1024, 128), (2048, 192), (4096, 256)], 48)

    ray_num_cpus = min(
        physical_cpus,
        _tiered_choice(
            logical_workers,
            [(64, 12), (128, 64), (256, 80), (512, 96), (1024, 128), (2048, 144), (4096, 160)],
            min(physical_cpus, 12),
        ),
    )
    ray_upsert_workers = _tiered_choice(logical_workers, [(64, 1), (128, 4), (256, 6), (512, 8), (1024, 12), (2048, 16)], 1)
    ray_upsert_workers = min(ray_upsert_workers, max(1, physical_cpus // 4))
    ray_shards = max(1, min(ray_upsert_workers, 8))
    ray_object_store_memory_mb = _tiered_choice(
        logical_workers,
        [(64, 1024), (128, 4096), (256, 8192), (512, 12288), (1024, 16384), (2048, 24576), (4096, 32768)],
        1024,
    )

    return {
        'profile': args.profile,
        'physical_workers': args.physical_workers,
        'async_workers': logical_workers,
        'stage_cpu_cap': physical_cpus,
        'nodes': nodes,
        'files': files,
        'chunks_per_file': args.chunks_per_file,
        'load_workers': load_workers,
        'transform_workers': transform_workers,
        'embed_workers': embed_workers,
        'upsert_workers': upsert_workers,
        'upsert_shards': upsert_shards,
        'upsert_timeout_ms': upsert_timeout_ms,
        'embed_batch': embed_batch,
        'upsert_batch': upsert_batch,
        'agentic_embed_batch': agentic_embed_batch,
        'agentic_upsert_batch': agentic_upsert_batch,
        'coalesce_multiplier': coalesce_multiplier,
        'ray_num_cpus': ray_num_cpus,
        'ray_upsert_workers': ray_upsert_workers,
        'ray_shards': ray_shards,
        'ray_object_store_memory_mb': ray_object_store_memory_mb,
    }


def main() -> int:
    p = argparse.ArgumentParser(description='AAFLOW strong/weak scaling wrapper with worker-aware tuning.')
    p.add_argument('--profile', choices=['strong', 'weak'], required=True)
    p.add_argument('--mode', choices=['agentic', 'ray'], default='agentic')
    p.add_argument('--async-workers', type=int, required=True)
    p.add_argument('--physical-workers', type=int, default=None)
    p.add_argument('--stage-cpu-cap', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')))
    p.add_argument('--base-nodes', type=int, default=100000)
    p.add_argument('--base-files', type=int, default=100)
    p.add_argument('--base-nodes-per-worker', type=int, default=64)
    p.add_argument('--node-chars', type=int, default=900)
    p.add_argument('--chunks-per-file', type=int, default=1000)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--embedder-backend', default='local-hash')
    p.add_argument('--set5-sink-backend', default='faiss')
    p.add_argument('--persist-dir', default='')
    p.add_argument('--run-dir', default='')
    p.add_argument('--python-bin', default=sys.executable)
    p.add_argument('--bench-script', default=str(BASE_BENCH))
    p.add_argument('--shared-corpus-root', default='')
    p.add_argument('--ray-input-format', choices=['raw', 'prechunked', 'preembedded'], default='raw')
    args = p.parse_args()

    if args.physical_workers is None:
        args.physical_workers = args.async_workers
    tuned, config_path = resolve_tuned_config(args)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'tuned_config.json').write_text(json.dumps(tuned, indent=2), encoding='utf-8')
        (run_dir / 'worker_config.json').write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')

    cmd = [
        args.python_bin,
        args.bench_script,
        '--no-cap-async-workers',
        '--nodes', str(tuned['nodes']),
        '--files', str(tuned['files']),
        '--chunks-per-file', str(tuned['chunks_per_file']),
        '--node-chars', str(args.node_chars),
        '--dim', str(args.dim),
        '--embedder-backend', args.embedder_backend,
        '--set5-sink-backend', args.set5_sink_backend,
        '--async-workers', str(tuned['async_workers']),
        '--set5-load-workers', str(tuned['load_workers']),
        '--set5-transform-workers', str(tuned['transform_workers']),
        '--set5-embed-workers', str(tuned['embed_workers']),
        '--set5-upsert-workers', str(tuned['upsert_workers']),
        '--set5-upsert-shards', str(tuned['upsert_shards']),
        '--set5-upsert-timeout-ms', str(tuned['upsert_timeout_ms']),
        '--set5-embed-batch', str(tuned['embed_batch']),
        '--set5-upsert-batch', str(tuned['upsert_batch']),
        '--set5-upsert-coalesce-multiplier', str(tuned['coalesce_multiplier']),
        '--no-scale-set5-batches',
        '--strict-stage-scaling',
    ]
    if args.mode == 'agentic':
        cmd.append('--only-agentic')
    else:
        cmd.extend([
            '--only-ray',
            '--run-ray-set6',
            '--sink-backend', args.set5_sink_backend,
            '--ray-num-cpus', str(tuned['ray_num_cpus']),
            '--upsert-workers-cap', str(tuned['ray_upsert_workers']),
            '--set45-upsert-shards', str(tuned['ray_shards']),
            '--ray-object-store-memory-mb', str(tuned['ray_object_store_memory_mb']),
        ])
    if args.persist_dir:
        cmd.extend(['--persist-dir', args.persist_dir])
    if args.shared_corpus_root:
        cmd.extend(['--shared-corpus-root', args.shared_corpus_root])
    if args.mode == 'ray':
        cmd.extend(['--ray-input-format', args.ray_input_format])

    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
