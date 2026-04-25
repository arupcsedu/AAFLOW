import json
import subprocess

from stateful_agentic_algebra.model_registry import (
    ModelSpec,
    default_model_registry,
    get_model_spec,
    list_models,
    model_availability,
)


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_default_registry_contains_required_models():
    models = default_model_registry()
    ids = {model.model_id for model in models}

    assert {"gpt2", "distilgpt2"} & ids
    assert "mistralai/Mistral-7B-Instruct-v0.3" in ids
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    assert any(model.family == "llama" for model in models)
    assert any(model.tensor_parallel_size > 1 for model in models)


def test_list_models_adds_availability_without_downloads():
    listed = list_models()

    assert listed
    assert all("available" in item for item in listed)
    assert all("reason" in item for item in listed)
    gated = [item for item in listed if item["requires_auth"]]
    assert gated
    assert all("model_id" in item for item in listed)


def test_model_availability_marks_missing_auth_for_gated_model(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    spec = ModelSpec(
        model_id="meta-llama/test",
        family="llama",
        size_label="7B",
        backend_options=["mock"],
        requires_auth=True,
    )

    status = model_availability(spec)

    assert status["available"] is False
    assert "auth-required" in status["reason"]


def test_model_registry_cli_list_prints_json():
    proc = subprocess.run(
        [PYTHON, "-m", "stateful_agentic_algebra.model_registry", "--list"],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert "models" in payload
    assert any(model["model_id"] == "Qwen/Qwen2.5-7B-Instruct" for model in payload["models"])


def test_get_model_spec():
    spec = get_model_spec("distilgpt2")

    assert spec is not None
    assert spec.family == "gpt2"
