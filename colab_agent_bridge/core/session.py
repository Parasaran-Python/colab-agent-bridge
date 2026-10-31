import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.hardware import validate_accelerator


class SessionManager:
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or get_config()

    def _build_base_cmd(self) -> List[str]:
        return [self.config.colab_cli_path, f"--auth={self.config.auth_strategy}"]

    def verify_auth(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return {"authenticated": True, "output": res.stdout.strip()}
            return {
                "authenticated": False,
                "error": res.stderr.strip() or res.stdout.strip(),
                "remediation": "Run: gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory",
            }
        except FileNotFoundError:
            return {
                "authenticated": False,
                "error": f"Colab binary not found at '{self.config.colab_cli_path}'",
                "remediation": "Ensure google-colab-cli is installed or set COLAB_CLI_PATH.",
            }

    def new_session(
        self,
        session_name: str,
        gpu: Optional[str] = None,
        tpu: Optional[str] = None,
        high_mem: bool = False,
    ) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["new", "-s", session_name]

        if gpu:
            spec = validate_accelerator(gpu)
            if spec.is_gpu:
                cmd.extend(["--gpu", spec.name])
        elif tpu:
            spec = validate_accelerator(tpu)
            if spec.is_tpu:
                cmd.extend(["--tpu", spec.name])

        if high_mem:
            cmd.append("--high-mem")

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode,
        }

    def stop_session(self, session_name: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["stop", "-s", session_name]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def list_sessions(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def get_session_status(self, session_name: Optional[str] = None) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["status"]
        if session_name:
            cmd.extend(["-s", session_name])
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }
