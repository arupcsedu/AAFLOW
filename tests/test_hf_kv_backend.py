import json
import subprocess

import pytest

from stateful_agentic_algebra.hf_kv_backend import HFBackendConfig, HFDecodeResult, HFKVBackend, HFPrefillResult


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_past_key_values_to_kv_state_uses_metadata_only_by_default():
    torch = pytest.importorskip("torch")
    key = torch.zeros((1, 2, 4, 8), dtype=torch.float32)
    value = torch.zeros((1, 2, 4, 8), dtype=torch.float32)

    state = HFKVBackend.past_key_values_to_kv_state(
        past_key_values=((key, value),),
        model_id="tiny",
        tokenizer_id="tiny-tokenizer",
        model_config_hash="hash",
        position_encoding="gpt2",
    )

    assert state.model_id == "tiny"
    assert len(state.blocks) == 1
    assert state.blocks[0].token_end == 4
    assert state.blocks[0].key_shape == (1, 2, 4, 8)
    assert state.blocks[0].value_shape == (1, 2, 4, 8)
    assert state.total_bytes() == key.numel() * key.element_size() + value.numel() * value.element_size()
    assert state.blocks[0].key_ref is None
    assert state.blocks[0].value_ref is None


def test_hf_backend_optional_available_returns_bool():
    assert isinstance(HFKVBackend.optional_available(), bool)


def test_measure_kv_bytes_uses_tensor_metadata():
    torch = pytest.importorskip("torch")
    key = torch.zeros((1, 2, 3, 4), dtype=torch.float16)
    value = torch.zeros((1, 2, 3, 4), dtype=torch.float16)

    assert HFKVBackend.measure_kv_bytes(((key, value),)) == key.numel() * key.element_size() * 2


def test_new_hf_result_types_are_importable():
    assert HFPrefillResult.__name__ == "HFPrefillResult"
    assert HFDecodeResult.__name__ == "HFDecodeResult"


def test_experiment_runner_accepts_hf_backend_without_crashing(tmp_path):
    output_dir = tmp_path / "hf"
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "stateful_agentic_algebra.experiment_runner",
            "--baseline",
            "ours_stateful",
            "--workload",
            "linear_handoff",
            "--backend",
            "hf",
            "--context-tokens",
            "8",
            "--output-tokens",
            "2",
            "--num-agents",
            "2",
            "--num-requests",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["output_dir"] == str(output_dir)
    assert (output_dir / "results.json").exists()
    assert (output_dir / "skipped_baselines.json").exists()
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["results"]
    skipped = json.loads((output_dir / "skipped_baselines.json").read_text(encoding="utf-8"))["skipped_baselines"]
    assert len(results) + len(skipped) == 1


def test_hf_backend_tiny_model_if_available_locally():
    pytest.importorskip("transformers")
    backend = HFKVBackend(
        HFBackendConfig(
            model_id="sshleifer/tiny-gpt2",
            tokenizer_id="sshleifer/tiny-gpt2",
            local_files_only=True,
        )
    )
    try:
        measurement = backend.measure("hello world", context_tokens=8, output_tokens=1)
    except Exception as exc:
        pytest.skip(f"tiny HF model is not available locally: {exc}")

    assert measurement.kv_state.total_bytes() > 0
    assert measurement.metrics["prefill_sec"] >= 0
    assert measurement.metrics["generated_tokens"] == 1
