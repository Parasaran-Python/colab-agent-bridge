import subprocess
from typing import Optional, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.core.profile import ProfileManager
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.cluster.models import ClusterNode


def generate_worker_install_script() -> str:
    return """
import subprocess, sys
print("[cluster-worker] Installing llama-cpp-python[server] and RPC components...")
cmd = [
    sys.executable, "-m", "pip", "install", "-q",
    "llama-cpp-python[server]", "huggingface_hub"
]
subprocess.run(cmd, check=True)
print("[cluster-worker] Worker dependencies installed successfully.")
"""


def generate_worker_start_script(port: int = 50052) -> str:
    return f"""
import os, sys, time, shutil

log_file = "/content/cluster_rpc_server.log"

# Locate rpc-server binary or launch via llama-cpp-python server
rpc_bin = shutil.which("rpc-server") or shutil.which("llama-rpc-server")
if not rpc_bin:
    py_bin = os.path.join(os.path.dirname(sys.executable), "rpc-server")
    if os.path.exists(py_bin):
        rpc_bin = py_bin
    else:
        # Fallback python RPC daemon wrapper
        rpc_bin = f"{{sys.executable}} -m llama_cpp.server.rpc"

cmd = (
    f"nohup {{rpc_bin}} "
    f"-p {port} "
    f"-H 0.0.0.0 "
    f"> {{log_file}} 2>&1 &"
)

print(f"[cluster-worker] Launching remote RPC server daemon on port {port}...")
os.system(cmd)
time.sleep(2)
print("[cluster-worker] Remote RPC server daemon active.")
"""


def provision_cluster_worker(
    node: ClusterNode,
    config: Optional[BridgeConfig] = None,
    timeout: int = 600,
    high_mem: bool = False,
    profile_mgr: Optional[ProfileManager] = None,
) -> Dict[str, Any]:
    cfg = config or get_config()
    sm = SessionManager(cfg, profile=node.profile)
    executor = RemoteExecutor(cfg, profile=node.profile)

    # 1. Pre-flight auth check
    auth = sm.verify_auth()
    if not auth.get("authenticated"):
        raise RuntimeError(
            f"Authentication failed for profile '{node.profile}': {auth.get('error')}\n{auth.get('remediation')}"
        )

    # 2. Provision VM
    sess_res = sm.new_session(
        session_name=node.session_name,
        gpu=node.gpu,
        high_mem=high_mem,
    )
    if not sess_res.get("success"):
        raise RuntimeError(
            f"Colab VM allocation failed for node '{node.session_name}' on profile '{node.profile}': {sess_res.get('stderr')}"
        )

    # 3. Install worker dependencies
    install_res = executor.run_code(
        node.session_name,
        generate_worker_install_script(),
        timeout=timeout,
    )
    if not install_res.get("success"):
        raise RuntimeError(
            f"Worker installation failed on node '{node.session_name}': {install_res.get('stderr') or install_res.get('stdout')}"
        )

    # 4. Start RPC daemon
    start_res = executor.run_code(
        node.session_name,
        generate_worker_start_script(port=node.remote_rpc_port),
        timeout=timeout,
    )
    if not start_res.get("success"):
        raise RuntimeError(
            f"Worker RPC daemon startup failed on node '{node.session_name}': {start_res.get('stderr') or start_res.get('stdout')}"
        )

    # 5. Establish SSH tunnel
    tunnel = SSHTunnel(
        session_name=node.session_name,
        local_port=node.local_rpc_port,
        remote_port=node.remote_rpc_port,
        config=cfg,
        profile=node.profile,
        profile_mgr=profile_mgr,
    )
    tunnel.start()
    node.tunnel = tunnel
    node.status = "active"

    return {
        "profile": node.profile,
        "session_name": node.session_name,
        "local_rpc_port": node.local_rpc_port,
        "remote_rpc_port": node.remote_rpc_port,
        "status": "active",
    }


def stop_cluster_worker(
    node: ClusterNode,
    config: Optional[BridgeConfig] = None,
) -> Dict[str, Any]:
    cfg = config or get_config()
    if node.tunnel:
        try:
            node.tunnel.stop()
        except Exception:
            pass
        node.tunnel = None

    SSHTunnel.cleanup_session_tunnels(node.session_name)
    sm = SessionManager(cfg, profile=node.profile)
    res = sm.stop_session(node.session_name)
    node.status = "stopped"
    return res
