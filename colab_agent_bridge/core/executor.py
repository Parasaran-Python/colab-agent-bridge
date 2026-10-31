import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config


class RemoteExecutor:
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or get_config()

    def _build_base_cmd(self) -> List[str]:
        return [self.config.colab_cli_path, f"--auth={self.config.auth_strategy}"]

    def run_code(self, session_name: str, code: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["exec", "-s", session_name]
        res = subprocess.run(cmd, input=code, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode,
        }

    def run_script(self, session_name: str, script_path: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["exec", "-s", session_name, "-f", script_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode,
        }

    def install_packages(self, session_name: str, packages: List[str]) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["install", "-s", session_name] + packages
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }
