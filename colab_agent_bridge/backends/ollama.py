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
os.system("apt-get update -qq && apt-get install -y -qq zstd pciutils && curl -fsSL https://ollama.com/install.sh | sh")
print("[Ollama] Ollama installation finished.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        # Note: Ollama standard port is 11434, we route to specified port via OLLAMA_HOST
        return f"""
import os, time

log_file = "/content/ollama_server.log"
cmd = f"nohup env PATH=$PATH:/usr/local/bin OLLAMA_HOST=0.0.0.0:{port} ollama serve > {{log_file}} 2>&1 &"
print(f"[Ollama] Starting Ollama server on 0.0.0.0:{port}...")
os.system(cmd)
time.sleep(3)

print(f"[Ollama] Pulling model {model_name}...")
res = os.system(f"env PATH=$PATH:/usr/local/bin OLLAMA_HOST=0.0.0.0:{port} ollama pull {model_name}")
actual_model = "{model_name}"
if res != 0 and "devstral" in "{model_name}":
    alt = "devstral:24b" if "small" in "{model_name}" else "devstral-small-2"
    print(f"[Ollama] Pulling alternate tag {{alt}}...")
    if os.system(f"env PATH=$PATH:/usr/local/bin OLLAMA_HOST=0.0.0.0:{port} ollama pull {{alt}}") == 0:
        actual_model = alt

for alias in ["default", "build", "chat", "{model_name}".replace(":", "-")]:
    os.system(f"env PATH=$PATH:/usr/local/bin OLLAMA_HOST=0.0.0.0:{port} ollama cp {{actual_model}} {{alias}} 2>/dev/null || true")
print(f"[Ollama] Model {{actual_model}} is ready on port {port}.")
"""
