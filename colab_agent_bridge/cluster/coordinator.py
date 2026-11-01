import os
import time
import shutil
import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import resolve_llama_server_binary, BridgeConfig, get_config
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.core.profile import ProfileManager
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.cluster.models import ClusterNode
from colab_agent_bridge.cluster.worker import generate_coordinator_start_script, generate_worker_install_script


class ClusterCoordinator:
    def __init__(
        self,
        nodes: List[ClusterNode],
        model_name: str,
        coordinator_port: int = 9000,
        coordinator_bin: Optional[str] = None,
        tensor_split: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self.nodes = nodes
        self.model_name = model_name
        self.coordinator_port = coordinator_port
        self.coordinator_bin = coordinator_bin or resolve_llama_server_binary()
        self.tensor_split = tensor_split
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.tunnel: Optional[SSHTunnel] = None
        self.is_remote: bool = False
        self._log_file = None

    @property
    def pid(self) -> Optional[int]:
        if self.process:
            return self.process.pid
        return None

    def build_command(self) -> List[str]:
        rpc_arg = ",".join([f"127.0.0.1:{node.local_rpc_port}" for node in self.nodes])
        split_arg = self.tensor_split if self.tensor_split else ",".join(["1"] * len(self.nodes))
        
        is_file = os.path.exists(self.model_name) or self.model_name.endswith(".gguf")
        if is_file:
            model_args = ["-m", self.model_name]
        else:
            repo = self.model_name
            if "gguf" not in repo.lower():
                repo = f"{repo}-GGUF"
            quant = ":Q4_K_M" if ":" not in repo else ""
            model_args = ["--hf-repo", f"{repo}{quant}"]

        cmd = [
            self.coordinator_bin,
            *model_args,
            "--alias",
            "default",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.coordinator_port),
            "--rpc",
            rpc_arg,
            "--tensor-split",
            split_arg,
            "-ngl",
            "999",
        ]
        cmd.extend(self.extra_args)
        return cmd

    def start_remote(
        self,
        coordinator_node: ClusterNode,
        worker_tunnel_url: Optional[str] = None,
        remote_port: int = 8000,
        local_port: int = 8000,
        timeout: int = 600,
        config: Optional[BridgeConfig] = None,
        profile_mgr: Optional[ProfileManager] = None,
    ) -> Dict[str, Any]:
        """Start llama-server directly on the remote Colab coordinator node."""
        cfg = config or get_config()
        executor = RemoteExecutor(cfg, profile=coordinator_node.profile)
        
        # 1. Install dependencies on coordinator if needed
        executor.run_code(
            coordinator_node.session_name,
            generate_worker_install_script(),
            timeout=timeout,
        )

        # 2. Launch coordinator script on remote node
        tensor_split = self.tensor_split or ",".join(["1"] * len(self.nodes))
        start_script = generate_coordinator_start_script(
            model_name=self.model_name,
            worker_tunnel_url=worker_tunnel_url,
            port=remote_port,
            tensor_split=tensor_split,
        )
        res = executor.run_code(
            coordinator_node.session_name,
            start_script,
            timeout=timeout,
        )
        if not res.get("success"):
            raise RuntimeError(
                f"Failed to launch remote coordinator on node '{coordinator_node.session_name}': "
                f"{res.get('stderr') or res.get('stdout')}"
            )

        # 3. Establish SSH tunnel forwarding local_port to remote_port
        tunnel = SSHTunnel(
            session_name=coordinator_node.session_name,
            local_port=local_port,
            remote_port=remote_port,
            config=cfg,
            profile=coordinator_node.profile,
            profile_mgr=profile_mgr,
        )
        tunnel.start()
        self.tunnel = tunnel
        self.is_remote = True
        coordinator_node.tunnel = tunnel
        coordinator_node.status = "active"

        return {
            "session_name": coordinator_node.session_name,
            "profile": coordinator_node.profile,
            "remote_port": remote_port,
            "local_port": local_port,
            "status": "active",
        }

    def start(self) -> None:
        # Check if coordinator binary exists
        bin_path = shutil.which(self.coordinator_bin)
        if not bin_path and not os.path.exists(self.coordinator_bin):
            raise FileNotFoundError(
                f"Cluster coordinator binary '{self.coordinator_bin}' not found in PATH. "
                "Ensure llama.cpp with RPC support is installed or configure --coordinator-bin / LLAMA_SERVER_PATH."
            )

        cmd = self.build_command()
        self._log_file = open("/tmp/cluster_coordinator.log", "a")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(2)
        if self.process.poll() is not None:
            self._log_file.flush()
            err_output = ""
            try:
                with open("/tmp/cluster_coordinator.log", "r") as f:
                    err_output = f.read()[-1000:]
            except Exception:
                pass
            raise RuntimeError(
                f"Cluster coordinator exited prematurely (code {self.process.returncode}):\n{err_output.strip()}"
            )

    def stop(self) -> None:
        if self.is_remote and self.tunnel:
            try:
                self.tunnel.stop()
            except Exception:
                pass
            self.tunnel = None

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def is_alive(self) -> bool:
        if self.is_remote:
            return self.tunnel is not None
        return self.process is not None and self.process.poll() is None
