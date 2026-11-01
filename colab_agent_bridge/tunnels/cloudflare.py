import re
from typing import Optional
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.tunnels.base import BaseTunnel


class CloudflareTunnel(BaseTunnel):
    def __init__(
        self,
        session_name: str,
        remote_port: int = 8000,
        config: Optional[BridgeConfig] = None,
        profile: Optional[str] = None,
    ):
        self.session_name = session_name
        self.remote_port = remote_port
        self.config = config or get_config()
        self.profile = profile
        self.executor = RemoteExecutor(self.config, profile=self.profile)
        self.tunnel_url: Optional[str] = None

    def generate_remote_script(self) -> str:
        return f"""
import os, time, subprocess, re

# Download cloudflared if not present
if not os.path.exists("/usr/local/bin/cloudflared"):
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")

log_path = "/content/cloudflared.log"
os.system(f"nohup /usr/local/bin/cloudflared tunnel --url http://localhost:{self.remote_port} > {{log_path}} 2>&1 &")

# Extract the generated trycloudflare url
url = None
for _ in range(30):
    time.sleep(1)
    if os.path.exists(log_path):
        content = open(log_path, 'r', errors='ignore').read()
        match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', content)
        if match:
            url = match.group(0)
            break

if url:
    print(f"CLOUDFLARE_URL:{{url}}")
else:
    print("CLOUDFLARE_FAILED")
"""

    def start(self) -> str:
        script = self.generate_remote_script()
        res = self.executor.run_code(self.session_name, script)
        stdout = res.get("stdout", "")
        match = re.search(r"CLOUDFLARE_URL:(https://[^\s]+)", stdout)
        if match:
            self.tunnel_url = match.group(1) + "/v1"
            return self.tunnel_url
        raise RuntimeError(f"Failed to start Cloudflare Tunnel on Colab VM: {stdout} {res.get('stderr')}")

    def stop(self) -> None:
        kill_code = "import os; os.system('pkill -f cloudflared')"
        self.executor.run_code(self.session_name, kill_code)
