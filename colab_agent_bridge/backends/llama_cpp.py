from typing import Any
from colab_agent_bridge.backends.base import BaseBackend


class LlamaCppBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "llama_cpp"

    def generate_install_script(self) -> str:
        return """
import subprocess, sys
print("[llama-cpp] Installing llama-cpp-python[server] with CUDA support...")
cmd = [
    sys.executable, "-m", "pip", "install", "-q",
    "llama-cpp-python[server]", "huggingface_hub"
]
subprocess.run(cmd, check=True)
print("[llama-cpp] Installed successfully.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        return f"""
import os, sys, time

log_file = "/content/llama_cpp_server.log"
cmd = (
    f"nohup {{sys.executable}} -m llama_cpp.server "
    f"--hf_model_repo_id {model_name} "
    f"--host 0.0.0.0 "
    f"--port {port} "
    f"--n_gpu_layers -1 "
    f"> {{log_file}} 2>&1 &"
)

print(f"[llama-cpp] Launching server for {model_name} on port {port}...")
os.system(cmd)
time.sleep(2)
print("[llama-cpp] Started in background.")
"""
