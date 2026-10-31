from typing import Any
from colab_agent_bridge.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "ollama"

    def generate_install_script(self) -> str:
        return """
import os, subprocess
print("[Ollama] Downloading and installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")
print("[Ollama] Ollama installation finished.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        # Note: Ollama standard port is 11434, we route to specified port via OLLAMA_HOST
        return f"""
import os, time

log_file = "/content/ollama_server.log"
cmd = f"nohup env OLLAMA_HOST=0.0.0.0:{port} ollama serve > {{log_file}} 2>&1 &"
print(f"[Ollama] Starting Ollama server on 0.0.0.0:{port}...")
os.system(cmd)
time.sleep(3)

print(f"[Ollama] Pulling model {model_name}...")
os.system(f"env OLLAMA_HOST=0.0.0.0:{port} ollama pull {model_name}")
os.system(f"env OLLAMA_HOST=0.0.0.0:{port} ollama cp {model_name} default 2>/dev/null || true")
print(f"[Ollama] Model {model_name} is ready on port {port}.")
"""
