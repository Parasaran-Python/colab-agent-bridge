import threading
import time
from typing import Optional, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.backends import get_backend
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.tunnels.cloudflare import CloudflareTunnel
from colab_agent_bridge.client.health import check_health, keepalive_ping
from colab_agent_bridge.client.harness import generate_agent_config
from colab_agent_bridge.adapter import start_adapter_server


class ColabBridge:
    def __init__(
        self,
        session_name: str = "agent-bridge",
        gpu: Optional[str] = "T4",
        tpu: Optional[str] = None,
        high_mem: bool = False,
        backend: str = "vllm",
        tunnel_type: str = "ssh",
        local_port: int = 8000,
        config: Optional[BridgeConfig] = None,
        profile: Optional[str] = None,
    ):
        self.session_name = session_name
        self.gpu = gpu
        self.tpu = tpu
        self.high_mem = high_mem
        self.backend_name = backend
        self.tunnel_type = tunnel_type
        self.local_port = local_port
        self.config = config or get_config()
        self.profile = profile

        self.session_mgr = SessionManager(self.config, profile=self.profile)
        self.executor = RemoteExecutor(self.config, profile=self.profile)
        self.tunnel = None
        self.adapter_server = None
        self.endpoint_url: Optional[str] = None
        self._keepalive_active = False
        self._keepalive_thread: Optional[threading.Thread] = None

    def get_endpoint(self) -> Optional[str]:
        return self.endpoint_url

    def deploy(self, model_name: str, wait_for_ready: bool = True, timeout: int = 600) -> Dict[str, Any]:
        # 0. Pre-flight auth check & auto-resolution
        auth_info = self.session_mgr.verify_auth()
        if not auth_info.get("authenticated"):
            raise RuntimeError(
                f"Colab CLI authentication failed: {auth_info.get('error')}\n{auth_info.get('remediation')}"
            )

        # 1. Provision session
        sess_res = self.session_mgr.new_session(
            session_name=self.session_name,
            gpu=self.gpu,
            tpu=self.tpu,
            high_mem=self.high_mem,
        )
        if not sess_res["success"]:
            raise RuntimeError(f"Colab VM allocation failed: {sess_res['stderr']}")

        # 2. Install backend prerequisites
        backend = get_backend(self.backend_name)
        install_script = backend.generate_install_script()
        install_res = self.executor.run_code(self.session_name, install_script, timeout=timeout)
        if not install_res.get("success"):
            raise RuntimeError(
                f"Backend installation failed on Colab VM: {install_res.get('stderr') or install_res.get('stdout')}"
            )

        # 3. Launch model server
        start_script = backend.generate_start_script(model_name=model_name, port=8000)
        self.executor.run_code(self.session_name, start_script, timeout=timeout)

        # 4. Open tunnel and smart adapter
        if self.tunnel_type == "cloudflare":
            self.tunnel = CloudflareTunnel(
                self.session_name,
                remote_port=8000,
                config=self.config,
                profile=self.profile,
            )
            self.endpoint_url = self.tunnel.start()
        else:
            tunnel_port = self.local_port + 1000
            self.tunnel = SSHTunnel(
                self.session_name,
                local_port=tunnel_port,
                remote_port=8000,
                config=self.config,
                profile=self.profile,
            )
            self.tunnel.start()
            self.adapter_server = start_adapter_server(
                listen_port=self.local_port, target_port=tunnel_port
            )
            self.endpoint_url = f"http://localhost:{self.local_port}/v1"

        # 5. Wait for readiness
        health_info = {"ready": True}
        if wait_for_ready:
            health_info = check_health(self.endpoint_url, timeout_seconds=timeout)

        # 6. Start keepalive loop in background
        self._start_keepalive()

        harness_conf = generate_agent_config(self.endpoint_url, model_name)
        return {
            "session_name": self.session_name,
            "endpoint_url": self.endpoint_url,
            "model_name": model_name,
            "health": health_info,
            "harness": harness_conf,
        }

    def _start_keepalive(self):
        self._keepalive_active = True

        def loop():
            while self._keepalive_active and self.endpoint_url:
                time.sleep(180)
                if self._keepalive_active:
                    keepalive_ping(self.endpoint_url)

        self._keepalive_thread = threading.Thread(target=loop, daemon=True)
        self._keepalive_thread.start()

    def teardown(self) -> Dict[str, Any]:
        self._keepalive_active = False
        if self.adapter_server:
            try:
                self.adapter_server.shutdown()
                self.adapter_server.server_close()
            except Exception:
                pass
        if self.tunnel:
            self.tunnel.stop()
        return self.session_mgr.stop_session(self.session_name)
