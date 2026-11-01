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
        # Uses standard OpenSSH with colab ssh --proxy-mode as ProxyCommand
        proxy_cmd = f"{self.config.colab_cli_path} --auth={self.config.auth_strategy} ssh --proxy-mode -s {self.session_name}"
        return [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            f"ProxyCommand={proxy_cmd}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{self.local_port}:localhost:{self.remote_port}",
            "root@colab-runtime",
        ]

    def start(self) -> str:
        cmd = self.build_ssh_command()
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
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

    @classmethod
    def cleanup_session_tunnels(cls, session_name: str) -> None:
        """Find and terminate any local SSH or ProxyCommand tunnel processes associated with the session."""
        import os
        import signal
        try:
            res = subprocess.run(
                ["pgrep", "-f", f"-s {session_name}"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                for pid_str in res.stdout.strip().split():
                    try:
                        pid = int(pid_str)
                        if pid != os.getpid():
                            os.kill(pid, signal.SIGTERM)
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
        except Exception:
            pass
