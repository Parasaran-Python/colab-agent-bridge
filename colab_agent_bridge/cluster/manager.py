import os
import time
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Union

from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.profile import ProfileManager, ProfileInfo
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.hardware import validate_accelerator, SUPPORTED_GPUS
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.adapter import start_adapter_server
from colab_agent_bridge.client.health import check_health, keepalive_ping
from colab_agent_bridge.client.harness import generate_agent_config

from colab_agent_bridge.cluster.models import ClusterNode, ClusterInfo
from colab_agent_bridge.cluster.state import ClusterStateStore
from colab_agent_bridge.cluster.worker import provision_cluster_worker, stop_cluster_worker
from colab_agent_bridge.cluster.coordinator import ClusterCoordinator


def validate_cluster_profiles(
    profiles: Union[List[str], str],
    profile_mgr: Optional[ProfileManager] = None,
) -> List[ProfileInfo]:
    if isinstance(profiles, str):
        profile_list = [p.strip() for p in profiles.split(",") if p.strip()]
    else:
        profile_list = [str(p).strip() for p in profiles if str(p).strip()]

    if len(profile_list) < 2:
        raise ValueError(
            "Distributed cluster weight sharding requires at least two profiles (e.g. --profiles default,personal_2)."
        )

    if len(profile_list) != len(set(profile_list)):
        raise ValueError(
            f"Duplicate profiles detected in {profile_list}. Each cluster node requires a distinct Google Colab profile."
        )

    mgr = profile_mgr or ProfileManager()
    all_profiles = {p.name: p for p in mgr.list_profiles()}

    verified: List[ProfileInfo] = []
    for name in profile_list:
        if name not in all_profiles:
            available = ", ".join(all_profiles.keys())
            raise ValueError(f"Profile '{name}' does not exist. Available profiles: {available}")
        info = all_profiles[name]
        if not info.authenticated:
            raise ValueError(
                f"Profile '{name}' is not authenticated. Authenticate it using 'colab-bridge profile add {name}'."
            )
        verified.append(info)

    return verified


def recommend_cluster_strategy(
    model: str,
    gpu: str = "T4",
    num_nodes: int = 2,
) -> Dict[str, Any]:
    spec = validate_accelerator(gpu)
    per_node_vram = spec.vram_gb
    total_pooled_vram = per_node_vram * num_nodes
    # 1.5 GB per node estimated for CUDA context + framework overhead
    usable_vram = max(0, total_pooled_vram - (num_nodes * 1.5))

    # Memory requirement heuristics based on parameter count
    lower_model = model.lower()
    if "70b" in lower_model:
        req_fp16 = 144
        req_quant = 56
    elif "32b" in lower_model:
        req_fp16 = 68
        req_quant = 26
    elif "14b" in lower_model:
        req_fp16 = 32
        req_quant = 14
    elif "7b" in lower_model or "8b" in lower_model or "9b" in lower_model:
        req_fp16 = 18
        req_quant = 8
    else:
        req_fp16 = 30
        req_quant = 16

    fits_fp16 = usable_vram >= req_fp16
    fits_quant = usable_vram >= req_quant
    needs_quant = not fits_fp16

    recommended_quant = []
    if needs_quant:
        recommended_quant = ["Q4_K_M (GGUF)", "AWQ-INT4", "GPTQ-4bit"]

    tensor_split = ",".join(["1"] * num_nodes)

    warning = None
    if not fits_quant:
        warning = (
            f"Model {model} requires ~{req_quant}GB VRAM in 4-bit, but {num_nodes}x {gpu} "
            f"only yields ~{usable_vram:.1f}GB usable VRAM. Consider adding more profiles/nodes."
        )

    return {
        "model": model,
        "gpu": gpu,
        "num_nodes": num_nodes,
        "per_node_vram_gb": per_node_vram,
        "total_pooled_vram_gb": total_pooled_vram,
        "usable_vram_gb": usable_vram,
        "fits_unquantized": fits_fp16,
        "fits_quantized": fits_quant,
        "needs_quantization": needs_quant,
        "recommended_quantization": recommended_quant,
        "tensor_split": tensor_split,
        "warning": warning,
    }


class ClusterBridge:
    def __init__(
        self,
        session_name: str = "cluster-bridge",
        profiles: Optional[Union[List[str], str]] = None,
        gpu: str = "T4",
        high_mem: bool = False,
        local_port: int = 8000,
        base_rpc_port: int = 50051,
        remote_rpc_port: int = 50052,
        coordinator_bin: Optional[str] = None,
        tensor_split: Optional[str] = None,
        remote_coordinator: bool = True,
        config: Optional[BridgeConfig] = None,
        profile_mgr: Optional[ProfileManager] = None,
        state_store: Optional[ClusterStateStore] = None,
    ):
        self.session_name = session_name
        if isinstance(profiles, str):
            self.profile_names = [p.strip() for p in profiles.split(",") if p.strip()]
        elif profiles:
            self.profile_names = [str(p).strip() for p in profiles if str(p).strip()]
        else:
            self.profile_names = []

        self.gpu = gpu
        self.high_mem = high_mem
        self.local_port = local_port
        self.base_rpc_port = base_rpc_port
        self.remote_rpc_port = remote_rpc_port
        self.coordinator_bin = coordinator_bin
        self.tensor_split = tensor_split
        self.remote_coordinator = remote_coordinator
        self.config = config or get_config()
        self.profile_mgr = profile_mgr or ProfileManager()
        self.state_store = state_store or ClusterStateStore()

        self.nodes: List[ClusterNode] = []
        self.coordinator: Optional[ClusterCoordinator] = None
        self.adapter_server = None
        self.endpoint_url: Optional[str] = None
        self._keepalive_active = False
        self._keepalive_thread: Optional[threading.Thread] = None

    def get_endpoint(self) -> Optional[str]:
        return self.endpoint_url

    def deploy(
        self,
        model_name: str,
        wait_for_ready: bool = True,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        # 1. Validate profiles
        verified = validate_cluster_profiles(self.profile_names, self.profile_mgr)

        # 2. Build cluster nodes
        self.nodes = []
        for idx, p_info in enumerate(verified):
            node_session = f"{self.session_name}-{p_info.name}"
            local_rpc_port = self.base_rpc_port + idx
            node = ClusterNode(
                profile=p_info.name,
                session_name=node_session,
                local_rpc_port=local_rpc_port,
                remote_rpc_port=self.remote_rpc_port,
                gpu=self.gpu,
            )
            self.nodes.append(node)

        coordinator_port = self.local_port + 1000
        tensor_split = self.tensor_split or ",".join(["1"] * len(self.nodes))

        if self.remote_coordinator:
            # Cloud-Native Distributed Coordinator Architecture
            # Node 0 is the Coordinator, Node 1..N are Workers
            self.nodes[0].role = "coordinator"
            for n in self.nodes[1:]:
                n.role = "worker"
            worker_nodes = self.nodes[1:]

            coord_node = self.nodes[0]
            sm = SessionManager(self.config, profile=coord_node.profile)
            auth = sm.verify_auth()
            if not auth.get("authenticated"):
                raise RuntimeError(
                    f"Authentication failed for coordinator profile '{coord_node.profile}': {auth.get('error')}"
                )

            # Provision coordinator VM and worker nodes concurrently in parallel
            def _provision_coord_vm():
                sess_res = sm.new_session(
                    session_name=coord_node.session_name,
                    gpu=coord_node.gpu,
                    high_mem=self.high_mem,
                )
                if not sess_res.get("success"):
                    raise RuntimeError(
                        f"Colab VM allocation failed for coordinator node '{coord_node.session_name}': {sess_res.get('stderr')}"
                    )
                return sess_res

            with ThreadPoolExecutor(max_workers=max(1, len(self.nodes))) as executor:
                coord_fut = executor.submit(_provision_coord_vm)
                worker_futures = {
                    executor.submit(
                        provision_cluster_worker,
                        node=n,
                        config=self.config,
                        timeout=timeout,
                        high_mem=self.high_mem,
                        profile_mgr=self.profile_mgr,
                    ): n for n in worker_nodes
                }
                coord_fut.result()
                for fut in as_completed(worker_futures):
                    node = worker_futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Worker provisioning failed for node '{node.session_name}': {exc}"
                        ) from exc

            worker_tunnel_url = worker_nodes[0].public_tunnel_url if worker_nodes else None
            self.coordinator = ClusterCoordinator(
                nodes=self.nodes,
                model_name=model_name,
                coordinator_port=coordinator_port,
                tensor_split=tensor_split,
            )
            self.coordinator.start_remote(
                coordinator_node=coord_node,
                worker_tunnel_url=worker_tunnel_url,
                remote_port=8000,
                local_port=coordinator_port,
                timeout=timeout,
                config=self.config,
                profile_mgr=self.profile_mgr,
            )
        else:
            # Fallback Local Coordinator Architecture
            def _provision(node: ClusterNode):
                provision_cluster_worker(
                    node=node,
                    config=self.config,
                    timeout=timeout,
                    high_mem=self.high_mem,
                    profile_mgr=self.profile_mgr,
                )

            if len(self.nodes) > 1:
                with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                    futures = {executor.submit(_provision, node): node for node in self.nodes}
                    for fut in as_completed(futures):
                        node = futures[fut]
                        try:
                            fut.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Worker provisioning failed for node '{node.session_name}': {exc}"
                            ) from exc
            else:
                for node in self.nodes:
                    _provision(node)

            import socket
            for node in self.nodes:
                ready = False
                start_t = time.time()
                while time.time() - start_t < 45:
                    try:
                        s = socket.create_connection(("127.0.0.1", node.local_rpc_port), timeout=1)
                        s.close()
                        ready = True
                        break
                    except Exception:
                        time.sleep(1)
                if not ready:
                    raise RuntimeError(
                        f"Local RPC tunnel port {node.local_rpc_port} failed to become reachable for node '{node.session_name}'."
                    )

            self.coordinator = ClusterCoordinator(
                nodes=self.nodes,
                model_name=model_name,
                coordinator_port=coordinator_port,
                coordinator_bin=self.coordinator_bin,
                tensor_split=tensor_split,
            )
            self.coordinator.start()

        # Start smart proxy adapter
        self.adapter_server = start_adapter_server(
            listen_port=self.local_port,
            target_port=coordinator_port,
        )
        self.endpoint_url = f"http://localhost:{self.local_port}/v1"

        # 6. Verify health
        health_info = {"ready": True}
        if wait_for_ready:
            health_info = check_health(self.endpoint_url, timeout_seconds=timeout)
            if not health_info.get("ready"):
                raise RuntimeError(
                    f"Cluster coordinator health check failed: {health_info.get('error', 'Timed out waiting for /v1/models')}"
                )

        # 7. Start background keepalive
        self._start_keepalive()

        # 8. Compute pooled VRAM and persist state
        spec = validate_accelerator(self.gpu)
        total_vram = spec.vram_gb * len(self.nodes)
        usable_vram = int(max(0, total_vram - (len(self.nodes) * 1.5)))

        cluster_info = ClusterInfo(
            session_name=self.session_name,
            model_name=model_name,
            profiles=[n.profile for n in self.nodes],
            gpu=self.gpu,
            nodes=self.nodes,
            coordinator_port=coordinator_port,
            listen_port=self.local_port,
            endpoint_url=self.endpoint_url,
            status="active",
            total_vram_gb=total_vram,
            usable_vram_gb=usable_vram,
            tensor_split=tensor_split,
            coordinator_pid=self.coordinator.pid,
        )
        self.state_store.save(cluster_info)

        harness_conf = generate_agent_config(self.endpoint_url, model_name)
        return {
            "session_name": self.session_name,
            "endpoint_url": self.endpoint_url,
            "model_name": model_name,
            "nodes": [n.to_dict() for n in self.nodes],
            "total_vram_gb": total_vram,
            "usable_vram_gb": usable_vram,
            "tensor_split": tensor_split,
            "health": health_info,
            "harness": harness_conf,
        }

    def _start_keepalive(self) -> None:
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

        # 1. Stop adapter server
        if self.adapter_server:
            try:
                self.adapter_server.shutdown()
                self.adapter_server.server_close()
            except Exception:
                pass
            self.adapter_server = None

        # 2. Stop coordinator process
        if self.coordinator:
            try:
                self.coordinator.stop()
            except Exception:
                pass
            self.coordinator = None

        # 3. Stop workers and release sessions concurrently
        node_results: Dict[str, Any] = {}
        if self.nodes:
            with ThreadPoolExecutor(max_workers=max(1, len(self.nodes))) as executor:
                futures = {
                    executor.submit(stop_cluster_worker, node, config=self.config): node
                    for node in self.nodes
                }
                for fut in as_completed(futures):
                    node = futures[fut]
                    try:
                        node_results[node.session_name] = fut.result()
                    except Exception as exc:
                        node_results[node.session_name] = {"success": False, "error": str(exc)}

        # 4. Remove from state store
        self.state_store.delete(self.session_name)

        return {
            "session_name": self.session_name,
            "success": all(r.get("success", False) for r in node_results.values()) if node_results else True,
            "nodes_released": list(node_results.keys()),
            "details": node_results,
        }


class ClusterManager:
    @staticmethod
    def list_clusters(state_store: Optional[ClusterStateStore] = None) -> List[ClusterInfo]:
        store = state_store or ClusterStateStore()
        return store.list_all()

    @staticmethod
    def get_cluster_status(
        session_name: Optional[str] = None,
        state_store: Optional[ClusterStateStore] = None,
    ) -> Optional[ClusterInfo]:
        store = state_store or ClusterStateStore()
        if session_name:
            return store.get(session_name)
        clusters = store.list_all()
        return clusters[0] if clusters else None

    @staticmethod
    def teardown_cluster(
        session_name: str,
        profiles: Optional[Union[List[str], str]] = None,
        config: Optional[BridgeConfig] = None,
        state_store: Optional[ClusterStateStore] = None,
    ) -> Dict[str, Any]:
        cfg = config or get_config()
        store = state_store or ClusterStateStore()
        cluster = store.get(session_name)

        # Clean adapter daemon processes if running
        try:
            subprocess.run(["pkill", "-f", "colab_agent_bridge.adapter"], capture_output=True)
        except Exception:
            pass

        # Clean coordinator if tracked
        if cluster and cluster.coordinator_pid:
            try:
                os.kill(cluster.coordinator_pid, 15)
            except Exception:
                pass

        # Determine target nodes
        target_nodes: List[ClusterNode] = []
        if cluster and cluster.nodes:
            target_nodes = cluster.nodes
        elif profiles:
            p_list = [p.strip() for p in profiles.split(",")] if isinstance(profiles, str) else list(profiles)
            for p in p_list:
                target_nodes.append(
                    ClusterNode(
                        profile=p,
                        session_name=f"{session_name}-{p}",
                        local_rpc_port=50051,
                    )
                )

        results: Dict[str, Any] = {}
        for node in target_nodes:
            res = stop_cluster_worker(node, config=cfg)
            results[node.session_name] = res

        store.delete(session_name)
        return {
            "session_name": session_name,
            "success": all(r.get("success", False) for r in results.values()) if results else True,
            "nodes_released": list(results.keys()),
            "details": results,
        }

    @staticmethod
    def teardown_all(
        config: Optional[BridgeConfig] = None,
        state_store: Optional[ClusterStateStore] = None,
    ) -> Dict[str, Any]:
        cfg = config or get_config()
        store = state_store or ClusterStateStore()
        clusters = store.list_all()

        total_released = []
        for c in clusters:
            res = ClusterManager.teardown_cluster(
                session_name=c.session_name,
                config=cfg,
                state_store=store,
            )
            total_released.extend(res.get("nodes_released", []))

        # Clear state file
        store.clear()
        return {
            "clusters_terminated": len(clusters),
            "nodes_released": total_released,
        }
