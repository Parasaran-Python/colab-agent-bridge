import subprocess
import time
from typing import List, Optional

from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.profile import ProfileManager
from colab_agent_bridge.tunnels.base import BaseTunnel


class SSHTunnel(BaseTunnel):
    def __init__(
        self,
        session_name: str,
        local_port: int = 8000,
        remote_port: int = 8000,
        config: Optional[BridgeConfig] = None,
        profile: Optional[str] = None,
        profile_mgr: Optional[ProfileManager] = None,
    ):
        self.session_name = session_name
        self.local_port = local_port
        self.remote_port = remote_port
        self.config = config or get_config()
        self.profile = profile
        self.profile_mgr = profile_mgr or ProfileManager()
        self.process: Optional[subprocess.Popen] = None

    def build_ssh_command(self) -> List[str]:
        # Uses standard OpenSSH with colab ssh --proxy-mode as ProxyCommand
        target_profile = self.profile if self.profile is not None else self.profile_mgr.get_active_profile()
        auth = "oauth2" if (target_profile and target_profile != "default") else self.config.auth_strategy
        proxy_cmd = f"{self.config.colab_cli_path} --auth={auth} ssh --proxy-mode -s {self.session_name}"
        if target_profile and target_profile != "default":
            p_path = self.profile_mgr.get_profile_path(target_profile)
            if p_path:
                proxy_cmd = f"env HOME={p_path} {proxy_cmd}"

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
        log_path = f"/tmp/colab_bridge_ssh_{self.session_name}.log"
        self._log_file = open(log_path, "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
            start_new_session=True,
        )
        time.sleep(2)
        if self.process.poll() is not None:
            stderr = ""
            if self.process.stderr is not None and hasattr(self.process.stderr, "read"):
                try:
                    err = self.process.stderr.read()
                    stderr = err.decode("utf-8", errors="replace") if isinstance(err, bytes) else str(err)
                except Exception:
                    pass
            if not stderr and hasattr(self, "_log_file"):
                try:
                    self._log_file.flush()
                    with open(log_path, "r") as f:
                        stderr = f.read()[-1000:]
                except Exception:
                    pass
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
        self.cleanup_session_tunnels(self.session_name)

    @classmethod
    def cleanup_session_tunnels(cls, session_name: str) -> None:
        """Find and terminate any local SSH or ProxyCommand tunnel processes associated with the session."""
        import os
        import signal

        try:
            res = subprocess.run(
                ["pgrep", "-f", "--", f"-s {session_name}"],
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
