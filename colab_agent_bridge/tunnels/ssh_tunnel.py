import subprocess
import time
from typing import Optional, List
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.tunnels.base import BaseTunnel


class SSHTunnel(BaseTunnel):
    def __init__(
        self,
        session_name: str,
        local_port: int = 8000,
        remote_port: int = 8000,
        config: Optional[BridgeConfig] = None,
    ):
        self.session_name = session_name
        self.local_port = local_port
        self.remote_port = remote_port
        self.config = config or get_config()
        self.process: Optional[subprocess.Popen] = None

    def build_ssh_command(self) -> List[str]:
        # Uses colab ssh with ProxyMode to set up direct local port forwarding
        return [
            self.config.colab_cli_path,
            f"--auth={self.config.auth_strategy}",
            "ssh",
            "-s",
            self.session_name,
            "--proxy-mode",
            "--",
            "-N",
            "-L",
            f"{self.local_port}:localhost:{self.remote_port}",
        ]

    def start(self) -> str:
        cmd = self.build_ssh_command()
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if self.process.poll() is not None:
            stderr = ""
            if self.process.stderr:
                err = self.process.stderr.read()
                stderr = err.decode("utf-8", errors="replace") if isinstance(err, bytes) else str(err)
            raise RuntimeError(
                f"SSH tunnel process exited immediately (code {self.process.returncode}): {stderr.strip()}"
            )
        return f"http://localhost:{self.local_port}/v1"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
