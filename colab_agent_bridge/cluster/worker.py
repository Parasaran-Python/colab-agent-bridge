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
import subprocess, sys, os, urllib.request, tarfile, glob

print("[cluster-worker] Installing RPC worker components...")
dest_dir = "/content/llama-bin"
os.makedirs(dest_dir, exist_ok=True)

# 1. Configure system dynamic linker with CUDA and Nvidia driver libraries
nvidia_libs = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
cuda_dirs = [dest_dir, "/usr/lib64-nvidia", "/usr/local/cuda/lib64"] + nvidia_libs
try:
    with open("/etc/ld.so.conf.d/nvidia-llama.conf", "w") as f:
        for d in cuda_dirs:
            if os.path.exists(d):
                f.write(f"{d}\\n")
    subprocess.run(["ldconfig"], check=False)
except Exception:
    pass

# Fast-path: if already installed and executable exists, skip re-download
installed = False
if os.path.exists(os.path.join(dest_dir, "ggml-rpc-server")) or os.path.exists("/usr/local/bin/ggml-rpc-server"):
    installed = True
else:
    for item in os.listdir("/content"):
        if item.startswith("llama-b") and item != "llama-bin":
            if os.path.exists(os.path.join("/content", item, "ggml-rpc-server")):
                # Ensure symlinks to dest_dir are created
                src = os.path.join("/content", item)
                for f in os.listdir(src):
                    s = os.path.join(src, f)
                    d = os.path.join(dest_dir, f)
                    if not os.path.exists(d):
                        try:
                            os.symlink(s, d)
                        except Exception:
                            pass
                installed = True
                break

if installed:
    print("[cluster-worker] Pre-built binary package already ready.")
    sys.exit(0)

# Download pre-built CUDA binary release for maximum performance
cuda_tar = "https://github.com/ggml-org/llama.cpp/releases/download/b12032/llama-b12032-bin-ubuntu-cuda-12.8-x64.tar.gz"
cpu_tar = "https://github.com/ggml-org/llama.cpp/releases/download/b12032/llama-b12032-bin-ubuntu-x64.tar.gz"
tar_path = "/tmp/llama-bin.tar.gz"

downloaded = False
for url in [cuda_tar, cpu_tar]:
    try:
        print(f"[cluster-worker] Downloading release from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        downloaded = True
        break
    except Exception as e:
        print(f"[cluster-worker] Download error ({e}), trying fallback...")

if downloaded:
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("/content")
        for item in os.listdir("/content"):
            if item.startswith("llama-b") and item != "llama-bin":
                src = os.path.join("/content", item)
                for f in os.listdir(src):
                    s = os.path.join(src, f)
                    d = os.path.join(dest_dir, f)
                    if not os.path.exists(d):
                        try:
                            os.symlink(s, d) if os.path.isfile(s) else None
                        except Exception:
                            pass
                break
        with open("/etc/ld.so.conf.d/llama.conf", "w") as f:
            f.write(f"{dest_dir}\\n")
        subprocess.run(["ldconfig"], check=False)
        for b in ["ggml-rpc-server", "rpc-server"]:
            b_path = os.path.join(dest_dir, b)
            if os.path.exists(b_path):
                target = f"/usr/local/bin/{b}"
                if not os.path.exists(target):
                    try:
                        os.symlink(b_path, target)
                    except Exception:
                        pass
        print("[cluster-worker] Pre-built binary package ready.")
    except Exception as e:
        print(f"[cluster-worker] Extraction fallback: {e}")

# Pre-download websocat and cloudflared for direct P2P cloud bridge
websocat_bin = "/content/websocat"
if not os.path.exists(websocat_bin):
    try:
        urllib.request.urlretrieve("https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl", websocat_bin)
        os.chmod(websocat_bin, 0o755)
    except Exception:
        pass

cloudflared_bin = "/content/cloudflared"
if not os.path.exists(cloudflared_bin):
    try:
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cloudflared_bin)
        os.chmod(cloudflared_bin, 0o755)
    except Exception:
        pass

print("[cluster-worker] Worker dependencies installed successfully.")
"""


def generate_worker_start_script(port: int = 50052, enable_tunnel: bool = True) -> str:
    tunnel_setup = ""
    if enable_tunnel:
        tunnel_setup = f"""
# Kill any existing websocat or cloudflared processes
os.system("pkill -9 -f 'websocat' || true")
os.system("pkill -9 -f 'cloudflared' || true")
time.sleep(1)

# Start websocat bridge (TCP port -> WebSocket port 18080)
if os.path.exists("/content/websocat"):
    os.system(f"nohup /content/websocat -b ws-l:127.0.0.1:18080 tcp:127.0.0.1:{port} > /content/websocat.log 2>&1 &")

# Start cloudflared tunnel
if os.path.exists("/content/cloudflared"):
    os.system("nohup /content/cloudflared tunnel --url http://127.0.0.1:18080 > /content/cloudflared.log 2>&1 &")

# Poll for trycloudflare.com URL
tunnel_url = None
for _ in range(30):
    time.sleep(1)
    if os.path.exists("/content/cloudflared.log"):
        with open("/content/cloudflared.log", "r") as f:
            content = f.read()
            m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", content)
            if m:
                tunnel_url = m.group(0)
                print(f"[cluster-worker] TUNNEL_URL: {{tunnel_url}}")
                break
"""

    return f"""
import os, sys, time, shutil, glob, urllib.request, re

log_file = "/content/cluster_rpc_server.log"
ld_path = "/content/llama-bin"

# Collect all nvidia cuda library directories
nvidia_libs = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
cuda_ld_paths = ":".join([ld_path, "/usr/lib64-nvidia", "/usr/local/cuda/lib64"] + nvidia_libs)

# Locate rpc-server binary
rpc_bin = (
    shutil.which("ggml-rpc-server")
    or shutil.which("rpc-server")
    or shutil.which("llama-rpc-server")
    or "/content/llama-bin/ggml-rpc-server"
    or "/content/llama-bin/rpc-server"
    or "/usr/local/bin/ggml-rpc-server"
)

if not rpc_bin or not os.path.exists(rpc_bin):
    for item in os.listdir("/content"):
        if item.startswith("llama-b") and item != "llama-bin":
            cand = os.path.join("/content", item, "ggml-rpc-server")
            if os.path.exists(cand):
                rpc_bin = cand
                ld_path = os.path.join("/content", item)
                break

if not rpc_bin or not os.path.exists(rpc_bin):
    rpc_bin = f"{{sys.executable}} -m llama_cpp.server.rpc"

# Terminate any previously running rpc-server instances
os.system("pkill -9 -f 'rpc-server' || true")
time.sleep(1)

cmd = (
    f"export LD_LIBRARY_PATH={{cuda_ld_paths}}:$LD_LIBRARY_PATH && "
    f"nohup {{rpc_bin}} "
    f"-p {port} "
    f"-H 0.0.0.0 "
    f"> {{log_file}} 2>&1 &"
)

print(f"[cluster-worker] Launching remote RPC server daemon on port {port}...")
os.system(cmd)
time.sleep(2)
print("[cluster-worker] Remote RPC server daemon active.")
{tunnel_setup}
"""


def generate_coordinator_start_script(
    model_name: str,
    worker_tunnel_url: Optional[str] = None,
    port: int = 8000,
    tensor_split: str = "1,1",
    ngl: int = 999,
) -> str:
    # Format HF repo identifier
    if model_name.endswith(".gguf") or ":" in model_name:
        hf_repo = model_name
    elif not model_name.endswith("-GGUF"):
        hf_repo = f"{model_name}-GGUF:Q4_K_M"
    else:
        hf_repo = f"{model_name}:Q4_K_M"

    tunnel_client_setup = ""
    rpc_arg = ""
    if worker_tunnel_url:
        wss_url = worker_tunnel_url.replace("https://", "wss://").replace("http://", "ws://")
        tunnel_client_setup = f"""
# Install websocat on coordinator
websocat_bin = "/content/websocat"
if not os.path.exists(websocat_bin):
    try:
        urllib.request.urlretrieve("https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl", websocat_bin)
        os.chmod(websocat_bin, 0o755)
    except Exception as e:
        print(f"[cluster-coordinator] websocat download error: {{e}}")

os.system("pkill -9 -f 'websocat' || true")
time.sleep(1)

# Wait until Cloudflare P2P tunnel edge is active and reachable
print("[cluster-coordinator] Waiting for Cloudflare P2P tunnel edge to become reachable...")
for _ in range(30):
    try:
        urllib.request.urlopen("{worker_tunnel_url}", timeout=2)
        break
    except urllib.error.HTTPError:
        # HTTP 400 Bad Request is expected from websocat ws-l listener
        break
    except Exception:
        time.sleep(1)

# Connect to worker's Cloudflare tunnel endpoint and expose local port 50052
if os.path.exists(websocat_bin):
    print("[cluster-coordinator] Establishing P2P WebSocket bridge to worker...")
    os.system(f"nohup /content/websocat -b tcp-l:127.0.0.1:50052 {wss_url} > /content/websocat_client.log 2>&1 &")
    time.sleep(3)
"""
        rpc_arg = "--rpc 127.0.0.1:50052"

    return f"""
import os, sys, time, shutil, glob, urllib.request

log_file = "/content/cluster_coordinator.log"
ld_path = "/content/llama-bin"

# Collect all nvidia cuda library directories
nvidia_libs = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
cuda_ld_paths = ":".join([ld_path, "/usr/lib64-nvidia", "/usr/local/cuda/lib64"] + nvidia_libs)

# Locate llama-server binary
server_bin = (
    shutil.which("llama-server")
    or "/content/llama-bin/llama-server"
    or "/usr/local/bin/llama-server"
)

if not server_bin or not os.path.exists(server_bin):
    for item in os.listdir("/content"):
        if item.startswith("llama-b") and item != "llama-bin":
            cand = os.path.join("/content", item, "llama-server")
            if os.path.exists(cand):
                server_bin = cand
                ld_path = os.path.join("/content", item)
                break

if not server_bin or not os.path.exists(server_bin):
    raise RuntimeError("llama-server binary not found on coordinator node.")

# Terminate any previously running llama-server
os.system("pkill -9 -f 'llama-server' || true")
time.sleep(1)

{tunnel_client_setup}

rpc_flag = "{rpc_arg}"
split_flag = "--tensor-split {tensor_split}" if rpc_flag else ""
cmd = (
    f"export LD_LIBRARY_PATH={{cuda_ld_paths}}:$LD_LIBRARY_PATH && "
    f"nohup {{server_bin}} "
    f"--hf-repo {hf_repo} "
    f"--alias default "
    f"--host 0.0.0.0 "
    f"--port {port} "
    f"{{rpc_flag}} "
    f"{{split_flag}} "
    f"-ngl {ngl} "
    f"> {{log_file}} 2>&1 &"
)

print(f"[cluster-coordinator] Launching llama-server coordinator on port {port}...")
os.system(cmd)
time.sleep(3)
print("[cluster-coordinator] Coordinator process launched.")
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
        generate_worker_start_script(port=node.remote_rpc_port, enable_tunnel=True),
        timeout=timeout,
    )
    if not start_res.get("success"):
        raise RuntimeError(
            f"Worker RPC daemon startup failed on node '{node.session_name}': {start_res.get('stderr') or start_res.get('stdout')}"
        )

    # Check for Cloudflare tunnel URL in stdout
    import re
    stdout_text = start_res.get("stdout") or ""
    m = re.search(r"TUNNEL_URL:\s*(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", stdout_text)
    if m:
        node.public_tunnel_url = m.group(1)

    # 5. Establish SSH tunnel
    SSHTunnel.cleanup_session_tunnels(node.session_name)
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
        "public_tunnel_url": node.public_tunnel_url,
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
