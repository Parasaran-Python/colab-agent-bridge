import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from colab_agent_bridge.config import resolve_colab_binary

DEFAULT_BRIDGE_CONFIG_DIR = Path.home() / ".config" / "colab-bridge"
DEFAULT_PROFILES_DIR = DEFAULT_BRIDGE_CONFIG_DIR / "profiles"
ACTIVE_PROFILE_FILE = DEFAULT_BRIDGE_CONFIG_DIR / "active_profile"


@dataclass
class ProfileInfo:
    name: str
    path: Path
    email: Optional[str] = None
    authenticated: bool = False
    cu_balance: float = 0.0
    cu_usage_rate: float = 0.0
    active_sessions: List[str] = field(default_factory=list)
    assignments: int = 0
    is_active: bool = False


class ProfileManager:
    def __init__(
        self,
        profiles_dir: Optional[Path] = None,
        colab_cli_path: Optional[str] = None,
        config_dir: Optional[Path] = None,
    ):
        self.config_dir = config_dir or (profiles_dir.parent if profiles_dir else DEFAULT_BRIDGE_CONFIG_DIR)
        self.profiles_dir = profiles_dir or (self.config_dir / "profiles")
        self.active_profile_file = self.config_dir / "active_profile"
        self.colab_cli_path = colab_cli_path or resolve_colab_binary()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def get_active_profile(self) -> str:
        if self.active_profile_file.exists():
            try:
                name = self.active_profile_file.read_text(encoding="utf-8").strip()
                if name:
                    return name
            except Exception:
                pass
        return "default"

    def set_active_profile(self, name: str) -> bool:
        if name != "default" and not (self.profiles_dir / name).exists():
            raise ValueError(f"Profile '{name}' does not exist.")
        self.active_profile_file.write_text(name, encoding="utf-8")
        return True

    def get_profile_path(self, name: Optional[str] = None) -> Optional[Path]:
        if not name or name == "default":
            return None
        return self.profiles_dir / name

    def get_profile_env(self, name: Optional[str] = None) -> Dict[str, str]:
        env = os.environ.copy()
        target = name if name is not None else self.get_active_profile()
        p_path = self.get_profile_path(target)
        if p_path is not None:
            env["HOME"] = str(p_path)
        return env

    def create_profile(self, name: str) -> Path:
        if name == "default":
            raise ValueError("Profile name 'default' is reserved for the system default account.")
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                f"Profile name '{name}' is invalid. Use alphanumeric characters, dashes, and underscores only."
            )
        p_path = self.profiles_dir / name
        p_path.mkdir(parents=True, exist_ok=True)
        (p_path / ".config" / "colab-cli").mkdir(parents=True, exist_ok=True)

        # Symlink .ssh from real home so colab ssh works without prompting
        real_ssh = Path.home() / ".ssh"
        profile_ssh = p_path / ".ssh"
        if real_ssh.exists() and not profile_ssh.exists():
            try:
                profile_ssh.symlink_to(real_ssh)
            except Exception:
                pass
        return p_path

    def remove_profile(self, name: str) -> bool:
        if name == "default":
            raise ValueError("Cannot remove the 'default' profile.")
        p_path = self.profiles_dir / name
        if not p_path.exists():
            return False
        shutil.rmtree(p_path)
        if self.get_active_profile() == name:
            self.set_active_profile("default")
        return True

    def fetch_email_from_token(self, token_file: Path, cache_file: Optional[Path] = None) -> Optional[str]:
        # Check cache first
        if cache_file and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if "email" in data and data["email"]:
                    return data["email"]
            except Exception:
                pass

        if not token_file.exists():
            return None

        try:
            with open(token_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            token = d.get("token")
            if not token:
                return None
            req = urllib.request.Request(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                info = json.loads(resp.read().decode("utf-8"))
                email = info.get("email")
                if email and cache_file:
                    try:
                        cache_file.write_text(json.dumps({"email": email}), encoding="utf-8")
                    except Exception:
                        pass
                return email
        except Exception:
            return None

    def get_profile_info(self, name: str) -> ProfileInfo:
        active_name = self.get_active_profile()
        is_active = name == active_name

        if name == "default":
            token_file = Path.home() / ".config" / "colab-cli" / "token.json"
            sessions_file = Path.home() / ".config" / "colab-cli" / "sessions.json"
            cache_file = self.config_dir / "default_profile_info.json"
            p_path = Path.home()
        else:
            p_path = self.profiles_dir / name
            token_file = p_path / ".config" / "colab-cli" / "token.json"
            sessions_file = p_path / ".config" / "colab-cli" / "sessions.json"
            cache_file = p_path / "profile_info.json"

        authenticated = token_file.exists()
        email = self.fetch_email_from_token(token_file, cache_file) if authenticated else None

        active_sessions: List[str] = []
        if sessions_file.exists():
            try:
                s_data = json.loads(sessions_file.read_text(encoding="utf-8"))
                active_sessions = list(s_data.keys())
            except Exception:
                pass

        cu_balance = 0.0
        cu_usage_rate = 0.0
        assignments = 0
        if authenticated:
            usage = self.query_usage(name)
            cu_balance = usage.get("balance", 0.0)
            cu_usage_rate = usage.get("rate", 0.0)
            assignments = usage.get("assignments", 0)

        # If email was not resolved (e.g. token expired and was refreshed during query_usage), retry once
        if authenticated and not email:
            email = self.fetch_email_from_token(token_file, cache_file)

        return ProfileInfo(
            name=name,
            path=p_path,
            email=email,
            authenticated=authenticated,
            cu_balance=cu_balance,
            cu_usage_rate=cu_usage_rate,
            active_sessions=active_sessions,
            assignments=assignments,
            is_active=is_active,
        )

    def list_profiles(self) -> List[ProfileInfo]:
        profiles = [self.get_profile_info("default")]
        if self.profiles_dir.exists():
            for entry in sorted(self.profiles_dir.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    profiles.append(self.get_profile_info(entry.name))
        return profiles

    def query_usage(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        env = self.get_profile_env(profile_name)
        target = profile_name if profile_name is not None else self.get_active_profile()
        auth_modes = ["oauth2"] if target != "default" else ["oauth2", "adc"]
        for auth_mode in auth_modes:
            try:
                res = subprocess.run(
                    [self.colab_cli_path, f"--auth={auth_mode}", "usage"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if res.returncode == 0:
                    balance = 0.0
                    rate = 0.0
                    assignments = 0
                    for line in res.stdout.splitlines():
                        lower = line.lower()
                        if "current balance:" in lower:
                            parts = line.split(":")
                            if len(parts) > 1:
                                val_str = parts[1].replace("compute units", "").strip()
                                try:
                                    balance = float(val_str)
                                except ValueError:
                                    pass
                        elif "usage rate:" in lower:
                            parts = line.split(":")
                            if len(parts) > 1:
                                val_str = parts[1].replace("/hr", "").strip()
                                try:
                                    rate = float(val_str)
                                except ValueError:
                                    pass
                        elif "active assignments:" in lower:
                            parts = line.split(":")
                            if len(parts) > 1:
                                try:
                                    assignments = int(parts[1].strip())
                                except ValueError:
                                    pass
                    return {"balance": balance, "rate": rate, "assignments": assignments, "success": True}
            except Exception:
                pass
        return {"balance": 0.0, "rate": 0.0, "assignments": 0, "success": False}
