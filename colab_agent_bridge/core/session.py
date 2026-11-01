import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.hardware import validate_accelerator
from colab_agent_bridge.core.profile import ProfileManager


class SessionManager:
    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        profile: Optional[str] = None,
        profile_mgr: Optional[ProfileManager] = None,
    ):
        self.config = config or get_config()
        self.profile = profile
        self.profile_mgr = profile_mgr or ProfileManager()

    def _get_env(self) -> Dict[str, str]:
        return self.profile_mgr.get_profile_env(self.profile)

    def _get_auth_strategy(self) -> str:
        target = self.profile if self.profile is not None else self.profile_mgr.get_active_profile()
        if target and target != "default":
            return "oauth2"
        return self.config.auth_strategy

    def _build_base_cmd(self) -> List[str]:
        return [self.config.colab_cli_path, f"--auth={self._get_auth_strategy()}"]

    def verify_auth(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        env = self._get_env()
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return {"authenticated": True, "output": res.stdout.strip()}

            # If default strategy failed, check alternative strategy (e.g. oauth2 if adc failed)
            alt_strategy = "oauth2" if self.config.auth_strategy == "adc" else "adc"
            alt_cmd = [self.config.colab_cli_path, f"--auth={alt_strategy}", "sessions"]
            alt_res = subprocess.run(alt_cmd, env=env, capture_output=True, text=True, check=False)
            if alt_res.returncode == 0:
                self.config.auth_strategy = alt_strategy
                return {"authenticated": True, "output": alt_res.stdout.strip(), "auto_switched": alt_strategy}

            return {
                "authenticated": False,
                "error": res.stderr.strip() or res.stdout.strip(),
                "remediation": (
                    "Run: gcloud auth application-default login --scopes=openid,"
                    "https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,"
                    "https://www.googleapis.com/auth/colaboratory\n"
                    "Or run: colab --auth=oauth2 sessions to authorize via browser."
                ),
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

        res = subprocess.run(cmd, env=self._get_env(), capture_output=True, text=True, check=False)
        out = (res.stderr + " " + res.stdout).lower()
        already_exists = "already exists" in out or "allocation refused" in out
        if not (res.returncode == 0 or already_exists):
            try:
                sessions_res = self.list_sessions()
                if sessions_res.get("success") and session_name in sessions_res.get("stdout", ""):
                    return {
                        "success": True,
                        "session_name": session_name,
                        "stdout": f"Session '{session_name}' is already active.",
                        "stderr": "",
                        "returncode": 0,
                    }
            except Exception:
                pass
        return {
            "success": res.returncode == 0 or already_exists,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode,
        }

    def stop_session(self, session_name: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["stop", "-s", session_name]
        res = subprocess.run(cmd, env=self._get_env(), capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def list_sessions(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        res = subprocess.run(cmd, env=self._get_env(), capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def get_session_status(self, session_name: Optional[str] = None) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["status"]
        if session_name:
            cmd.extend(["-s", session_name])
        res = subprocess.run(cmd, env=self._get_env(), capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def get_usage(self) -> Dict[str, Any]:
        return self.profile_mgr.query_usage(self.profile)
