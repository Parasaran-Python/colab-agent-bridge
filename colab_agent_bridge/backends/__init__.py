from typing import Dict, Type
from colab_agent_bridge.backends.base import BaseBackend
from colab_agent_bridge.backends.vllm import VLLMBackend
from colab_agent_bridge.backends.ollama import OllamaBackend
from colab_agent_bridge.backends.llama_cpp import LlamaCppBackend

BACKENDS: Dict[str, Type[BaseBackend]] = {
    "vllm": VLLMBackend,
    "ollama": OllamaBackend,
    "llama_cpp": LlamaCppBackend,
}


def get_backend(name: str) -> BaseBackend:
    name_lower = name.lower()
    if name_lower not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name_lower]()


__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "BACKENDS",
    "get_backend",
]
