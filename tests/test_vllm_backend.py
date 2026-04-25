import importlib.util
import subprocess

from stateful_agentic_algebra.vllm_backend import VLLMBackend, VLLMBackendConfig
from stateful_agentic_algebra.vllm_backend import _extract_ttft, _output_text, _output_token_count


PYTHON = "/scratch/djy8hg/env/drc_rag_bench_env/bin/python"
CWD = "/project/bi_dsc_community/drc_rag"


def test_vllm_backend_available_reflects_import_spec():
    assert VLLMBackend.available() == (importlib.util.find_spec("vllm") is not None)


def test_vllm_backend_missing_dependency_raises_clear_runtime_error():
    if VLLMBackend.available():
        return

    try:
        VLLMBackend(VLLMBackendConfig(model_id="gpt2")).run_prompts("hello")
    except RuntimeError as exc:
        assert "vLLM is not installed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when vLLM is unavailable")


def test_vllm_kv_placeholders_are_explicit():
    backend = VLLMBackend()

    for method_name in ("export_kv_state", "import_kv_state"):
        try:
            getattr(backend, method_name)()
        except NotImplementedError as exc:
            assert "stable public KV cache" in str(exc)
        else:
            raise AssertionError(f"expected NotImplementedError from {method_name}")


def test_vllm_backend_import_from_package_is_lazy():
    proc = subprocess.run(
        [PYTHON, "-c", "from stateful_agentic_algebra import VLLMBackend; print(VLLMBackend.available())"],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.strip() in {"True", "False"}


def test_vllm_baseline_skipped_status_is_clean_when_missing():
    proc = subprocess.run(
        [PYTHON, "-m", "stateful_agentic_algebra.experiment_runner", "--list-baselines"],
        cwd=CWD,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "vllm_local_prefix" in proc.stdout
    if not VLLMBackend.available():
        assert "vLLM is not installed" in proc.stdout


def test_vllm_output_helpers_extract_text_tokens_and_ttft():
    class Metrics:
        time_to_first_token = 0.12

    class Completion:
        text = "hello world"
        token_ids = [1, 2, 3]

    class RequestOutput:
        outputs = [Completion()]
        metrics = Metrics()

    output = RequestOutput()

    assert _output_text(output) == "hello world"
    assert _output_token_count(output) == 3
    assert _extract_ttft([output]) == 0.12
