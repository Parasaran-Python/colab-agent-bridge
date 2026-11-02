import pytest

from colab_agent_bridge.backends import (
    LlamaCppBackend,
    OllamaBackend,
    VLLMBackend,
    get_backend,
)


def test_vllm_backend_script_generation():
    backend = get_backend("vllm")
    assert isinstance(backend, VLLMBackend)
    assert backend.name == "vllm"
    install_script = backend.generate_install_script()
    assert "vllm" in install_script

    start_script = backend.generate_start_script(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        port=8000,
        quantization="awq",
    )
    assert "vllm.entrypoints.openai.api_server" in start_script
    assert "--model Qwen/Qwen2.5-Coder-7B-Instruct" in start_script
    assert "--port 8000" in start_script
    assert "--quantization awq" in start_script


def test_ollama_backend_script_generation():
    backend = get_backend("ollama")
    assert isinstance(backend, OllamaBackend)
    assert backend.name == "ollama"
    install_script = backend.generate_install_script()
    assert "ollama" in install_script.lower()

    start_script = backend.generate_start_script(
        model_name="qwen2.5-coder:7b",
        port=8000,
    )
    assert "ollama serve" in start_script
    assert "ollama pull qwen2.5-coder:7b" in start_script


def test_llama_cpp_backend_script_generation():
    backend = get_backend("llama_cpp")
    assert isinstance(backend, LlamaCppBackend)
    assert backend.name == "llama_cpp"
    install_script = backend.generate_install_script()
    assert "llama-cpp-python" in install_script

    start_script = backend.generate_start_script(
        model_name="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        port=8000,
    )
    assert "llama_cpp.server" in start_script
    assert "8000" in start_script


def test_get_backend_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent")


def test_backend_case_insensitive_and_defaults():
    backend = get_backend("VLLM")
    assert isinstance(backend, VLLMBackend)
    start_script = backend.generate_start_script("test-model")
    assert "--max-model-len 8192" in start_script
    assert "--gpu-memory-utilization 0.9" in start_script
    assert "--quantization" not in start_script
