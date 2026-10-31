from typing import Any
from colab_agent_bridge.backends.base import BaseBackend


class VLLMBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "vllm"

    def generate_install_script(self) -> str:
        return """
import subprocess, sys
print("[vLLM] Installing vllm via uv/pip...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "accelerate"], check=True)
print("[vLLM] Installation completed.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        quant = kwargs.get("quantization")
        max_len = kwargs.get("max_model_len", 8192)
        gpu_util = kwargs.get("gpu_memory_utilization", 0.90)

        quant_flag = f"--quantization {quant}" if quant else ""

        return f"""
import subprocess, os, sys, time

log_file = "/content/vllm_server.log"
cmd = (
    f"nohup {{sys.executable}} -m vllm.entrypoints.openai.api_server "
    f"--model {model_name} "
    f"--port {port} "
    f"--host 0.0.0.0 "
    f"--max-model-len {max_len} "
    f"--gpu-memory-utilization {gpu_util} "
    f"{quant_flag} "
    f"> {{log_file}} 2>&1 &"
)

print(f"[vLLM] Launching vLLM OpenAI API server for {model_name} on port {port}...")
os.system(cmd)
time.sleep(2)
print("[vLLM] Background supervisor started. Logs streaming to /content/vllm_server.log")
"""
