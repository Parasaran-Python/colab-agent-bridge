import os
import time
import shutil
import subprocess
from typing import Optional, List
from colab_agent_bridge.config import resolve_llama_server_binary
from colab_agent_bridge.cluster.models import ClusterNode


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
        return self.process is not None and self.process.poll() is None
