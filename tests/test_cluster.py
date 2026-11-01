import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner

from colab_agent_bridge.cli import app
from colab_agent_bridge.cluster.models import ClusterNode, ClusterInfo
from colab_agent_bridge.cluster.state import ClusterStateStore
from colab_agent_bridge.cluster.worker import (
    generate_worker_install_script,
    generate_worker_start_script,
    generate_coordinator_start_script,
    provision_cluster_worker,
    stop_cluster_worker,
)
from colab_agent_bridge.cluster.coordinator import ClusterCoordinator
from colab_agent_bridge.cluster.manager import (
    ClusterBridge,
    ClusterManager,
    validate_cluster_profiles,
    recommend_cluster_strategy,
)
from colab_agent_bridge.core.profile import ProfileInfo

runner = CliRunner()


# --- Models & State Tests ---

def test_cluster_models():
    node = ClusterNode(
        profile="personal_2",
        session_name="cluster-test-personal_2",
        local_rpc_port=50052,
        remote_rpc_port=50052,
        gpu="T4",
        status="active",
    )
    d = node.to_dict()
    assert d["profile"] == "personal_2"
    assert d["local_rpc_port"] == 50052
    assert d["status"] == "active"

    restored_node = ClusterNode.from_dict(d)
    assert restored_node.profile == "personal_2"
    assert restored_node.session_name == "cluster-test-personal_2"

    info = ClusterInfo(
        session_name="cluster-test",
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        profiles=["default", "personal_2"],
        gpu="T4",
        nodes=[node],
        coordinator_port=9000,
        listen_port=8000,
        endpoint_url="http://localhost:8000/v1",
        status="active",
        total_vram_gb=32,
        usable_vram_gb=29,
        tensor_split="1,1",
        coordinator_pid=12345,
    )
    info_dict = info.to_dict()
    assert info_dict["session_name"] == "cluster-test"
    assert info_dict["total_vram_gb"] == 32
    assert info_dict["coordinator_pid"] == 12345

    restored_info = ClusterInfo.from_dict(info_dict)
    assert restored_info.session_name == "cluster-test"
    assert len(restored_info.nodes) == 1
    assert restored_info.nodes[0].profile == "personal_2"


def test_cluster_state_store(tmp_path):
    state_file = tmp_path / "clusters.json"
    store = ClusterStateStore(state_file=state_file)

    assert store.list_all() == []
    assert store.get("nonexistent") is None

    node = ClusterNode(
        profile="default",
        session_name="c1-default",
        local_rpc_port=50051,
    )
    c1 = ClusterInfo(
        session_name="c1",
        model_name="test-model",
        profiles=["default", "p2"],
        gpu="T4",
        nodes=[node],
        coordinator_port=9000,
        listen_port=8000,
        endpoint_url="http://localhost:8000/v1",
    )
    store.save(c1)

    loaded = store.get("c1")
    assert loaded is not None
    assert loaded.session_name == "c1"
    assert loaded.model_name == "test-model"

    all_clusters = store.list_all()
    assert len(all_clusters) == 1
    assert all_clusters[0].session_name == "c1"

    # Delete
    deleted = store.delete("c1")
    assert deleted is True
    assert store.get("c1") is None
    assert store.delete("c1") is False

    # Clear
    store.save(c1)
    store.clear()
    assert store.list_all() == []


def test_cluster_state_store_corrupted(tmp_path):
    state_file = tmp_path / "clusters.json"
    state_file.write_text("{bad json")
    store = ClusterStateStore(state_file=state_file)
    assert store.list_all() == []
    assert store.get("any") is None


# --- Profile Validation Tests ---

def test_validate_cluster_profiles_success():
    mock_pm = MagicMock()
    mock_pm.list_profiles.return_value = [
        ProfileInfo(name="default", path=Path("/home"), authenticated=True),
        ProfileInfo(name="personal_2", path=Path("/profiles/p2"), authenticated=True),
    ]

    verified = validate_cluster_profiles("default,personal_2", profile_mgr=mock_pm)
    assert len(verified) == 2
    assert verified[0].name == "default"
    assert verified[1].name == "personal_2"

    verified_list = validate_cluster_profiles(["default", "personal_2"], profile_mgr=mock_pm)
    assert len(verified_list) == 2


def test_validate_cluster_profiles_too_few():
    mock_pm = MagicMock()
    with pytest.raises(ValueError, match="at least two profiles"):
        validate_cluster_profiles("default", profile_mgr=mock_pm)


def test_validate_cluster_profiles_duplicate():
    mock_pm = MagicMock()
    with pytest.raises(ValueError, match="Duplicate profiles detected"):
        validate_cluster_profiles(["default", "default"], profile_mgr=mock_pm)


def test_validate_cluster_profiles_not_found():
    mock_pm = MagicMock()
    mock_pm.list_profiles.return_value = [
        ProfileInfo(name="default", path=Path("/home"), authenticated=True),
    ]
    with pytest.raises(ValueError, match="Profile 'unknown' does not exist"):
        validate_cluster_profiles(["default", "unknown"], profile_mgr=mock_pm)


def test_validate_cluster_profiles_unauthenticated():
    mock_pm = MagicMock()
    mock_pm.list_profiles.return_value = [
        ProfileInfo(name="default", path=Path("/home"), authenticated=True),
        ProfileInfo(name="personal_2", path=Path("/profiles/p2"), authenticated=False),
    ]
    with pytest.raises(ValueError, match="not authenticated"):
        validate_cluster_profiles(["default", "personal_2"], profile_mgr=mock_pm)


# --- Strategy Recommendation Tests ---

def test_recommend_cluster_strategy_2x_t4():
    rec = recommend_cluster_strategy("Qwen/Qwen2.5-Coder-32B-Instruct", gpu="T4", num_nodes=2)
    assert rec["num_nodes"] == 2
    assert rec["total_pooled_vram_gb"] == 32
    assert rec["usable_vram_gb"] == 29.0
    assert rec["fits_quantized"] is True
    assert rec["fits_unquantized"] is False
    assert rec["needs_quantization"] is True
    assert "Q4_K_M (GGUF)" in rec["recommended_quantization"]
    assert rec["tensor_split"] == "1,1"


def test_recommend_cluster_strategy_4x_t4_llama70b():
    rec = recommend_cluster_strategy("meta-llama/Llama-3.3-70B-Instruct", gpu="T4", num_nodes=4)
    assert rec["num_nodes"] == 4
    assert rec["total_pooled_vram_gb"] == 64
    assert rec["usable_vram_gb"] == 58.0
    assert rec["fits_quantized"] is True
    assert rec["tensor_split"] == "1,1,1,1"


def test_recommend_cluster_strategy_insufficient():
    rec = recommend_cluster_strategy("meta-llama/Llama-3.3-70B-Instruct", gpu="T4", num_nodes=2)
    assert rec["fits_quantized"] is False
    assert rec["warning"] is not None


# --- Worker Scripts & Lifecycle Tests ---

def test_worker_scripts():
    install_script = generate_worker_install_script()
    assert "Installing RPC worker components" in install_script
    assert "llama-bin" in install_script

    start_script = generate_worker_start_script(port=50052, enable_tunnel=True)
    assert "port 50052" in start_script
    assert "-H 0.0.0.0" in start_script
    assert "rpc-server" in start_script
    assert "websocat" in start_script
    assert "cloudflared" in start_script

    coord_script = generate_coordinator_start_script(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        worker_tunnel_url="https://test-worker.trycloudflare.com",
    )
    assert "llama-server" in coord_script
    assert "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M" in coord_script
    assert "wss://test-worker.trycloudflare.com" in coord_script
    assert "--rpc 127.0.0.1:50052" in coord_script


@patch("colab_agent_bridge.cluster.worker.SSHTunnel")
@patch("colab_agent_bridge.cluster.worker.RemoteExecutor")
@patch("colab_agent_bridge.cluster.worker.SessionManager")
def test_provision_cluster_worker_success(mock_sm_cls, mock_exec_cls, mock_ssh_cls):
    mock_sm = MagicMock()
    mock_sm.verify_auth.return_value = {"authenticated": True}
    mock_sm.new_session.return_value = {"success": True}
    mock_sm_cls.return_value = mock_sm

    mock_exec = MagicMock()
    mock_exec.run_code.return_value = {"success": True}
    mock_exec_cls.return_value = mock_exec

    mock_tunnel = MagicMock()
    mock_ssh_cls.return_value = mock_tunnel

    node = ClusterNode(
        profile="personal_2",
        session_name="cluster-node-0",
        local_rpc_port=50051,
        remote_rpc_port=50052,
    )

    res = provision_cluster_worker(node)
    assert res["status"] == "active"
    assert node.status == "active"
    mock_tunnel.start.assert_called_once()


@patch("colab_agent_bridge.cluster.worker.SessionManager")
def test_provision_cluster_worker_auth_fail(mock_sm_cls):
    mock_sm = MagicMock()
    mock_sm.verify_auth.return_value = {"authenticated": False, "error": "ADC missing", "remediation": "login"}
    mock_sm_cls.return_value = mock_sm

    node = ClusterNode(
        profile="personal_2",
        session_name="cluster-node-0",
        local_rpc_port=50051,
    )
    with pytest.raises(RuntimeError, match="Authentication failed"):
        provision_cluster_worker(node)


@patch("colab_agent_bridge.cluster.worker.SSHTunnel")
@patch("colab_agent_bridge.cluster.worker.SessionManager")
def test_stop_cluster_worker(mock_sm_cls, mock_ssh_cls):
    mock_sm = MagicMock()
    mock_sm.stop_session.return_value = {"success": True}
    mock_sm_cls.return_value = mock_sm

    mock_tunnel = MagicMock()
    node = ClusterNode(
        profile="personal_2",
        session_name="cluster-node-0",
        local_rpc_port=50051,
        tunnel=mock_tunnel,
    )

    res = stop_cluster_worker(node)
    assert res["success"] is True
    assert node.status == "stopped"
    mock_tunnel.stop.assert_called_once()
    mock_ssh_cls.cleanup_session_tunnels.assert_called_once_with("cluster-node-0")


# --- Cluster Coordinator Tests ---

def test_coordinator_command_generation():
    nodes = [
        ClusterNode(profile="default", session_name="c-default", local_rpc_port=50051),
        ClusterNode(profile="personal_2", session_name="c-personal_2", local_rpc_port=50052),
    ]
    coord = ClusterCoordinator(
        nodes=nodes,
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        coordinator_port=9000,
        coordinator_bin="llama-server",
    )
    cmd = coord.build_command()
    assert "llama-server" in cmd
    assert "--port" in cmd
    assert "9000" in cmd
    assert "--rpc" in cmd
    rpc_idx = cmd.index("--rpc")
    assert cmd[rpc_idx + 1] == "127.0.0.1:50051,127.0.0.1:50052"
    assert "--tensor-split" in cmd
    split_idx = cmd.index("--tensor-split")
    assert cmd[split_idx + 1] == "1,1"


def test_coordinator_command_custom_split():
    nodes = [
        ClusterNode(profile="default", session_name="c-default", local_rpc_port=50051),
        ClusterNode(profile="personal_2", session_name="c-personal_2", local_rpc_port=50052),
    ]
    coord = ClusterCoordinator(
        nodes=nodes,
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        tensor_split="2,1",
    )
    cmd = coord.build_command()
    split_idx = cmd.index("--tensor-split")
    assert cmd[split_idx + 1] == "2,1"


@patch("shutil.which")
@patch("subprocess.Popen")
def test_coordinator_lifecycle(mock_popen, mock_which):
    mock_which.return_value = "/usr/local/bin/llama-server"
    mock_proc = MagicMock()
    mock_proc.pid = 4321
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    nodes = [ClusterNode(profile="default", session_name="n0", local_rpc_port=50051)]
    coord = ClusterCoordinator(nodes=nodes, model_name="test")

    coord.start()
    assert coord.pid == 4321
    assert coord.is_alive() is True

    coord.stop()
    mock_proc.terminate.assert_called_once()


@patch("colab_agent_bridge.cluster.coordinator.SSHTunnel")
@patch("colab_agent_bridge.cluster.coordinator.RemoteExecutor")
def test_coordinator_remote_lifecycle(mock_exec_cls, mock_tunnel_cls):
    mock_exec = MagicMock()
    mock_exec.run_code.return_value = {"success": True}
    mock_exec_cls.return_value = mock_exec

    mock_tunnel = MagicMock()
    mock_tunnel_cls.return_value = mock_tunnel

    node = ClusterNode(profile="default", session_name="c-default", local_rpc_port=50051)
    coord = ClusterCoordinator(nodes=[node], model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    res = coord.start_remote(
        coordinator_node=node,
        worker_tunnel_url="https://test.trycloudflare.com",
        remote_port=8000,
        local_port=9000,
    )

    assert res["status"] == "active"
    assert coord.is_alive() is True
    assert coord.is_remote is True
    mock_tunnel.start.assert_called_once()

    coord.stop()
    mock_tunnel.stop.assert_called_once()
    assert coord.tunnel is None


def test_coordinator_missing_binary():
    nodes = [ClusterNode(profile="default", session_name="n0", local_rpc_port=50051)]
    coord = ClusterCoordinator(nodes=nodes, model_name="test", coordinator_bin="non_existent_binary_xyz")
    with pytest.raises(FileNotFoundError, match="not found in PATH"):
        coord.start()


# --- ClusterBridge Tests ---

@patch("colab_agent_bridge.cluster.manager.SessionManager")
@patch("colab_agent_bridge.cluster.manager.check_health")
@patch("colab_agent_bridge.cluster.manager.start_adapter_server")
@patch("colab_agent_bridge.cluster.manager.ClusterCoordinator")
@patch("colab_agent_bridge.cluster.manager.provision_cluster_worker")
@patch("colab_agent_bridge.cluster.manager.validate_cluster_profiles")
def test_cluster_bridge_deploy_success(
    mock_val,
    mock_prov,
    mock_coord_cls,
    mock_adapter,
    mock_health,
    mock_sm_cls,
    tmp_path,
):
    mock_val.return_value = [
        ProfileInfo(name="default", path=Path("/h"), authenticated=True),
        ProfileInfo(name="personal_2", path=Path("/p"), authenticated=True),
    ]
    mock_prov.return_value = {"status": "active", "public_tunnel_url": "https://worker.trycloudflare.com"}

    mock_sm = MagicMock()
    mock_sm.verify_auth.return_value = {"authenticated": True}
    mock_sm.new_session.return_value = {"success": True}
    mock_sm_cls.return_value = mock_sm

    mock_coord = MagicMock()
    mock_coord.pid = 9999
    mock_coord_cls.return_value = mock_coord

    mock_server = MagicMock()
    mock_adapter.return_value = mock_server

    mock_health.return_value = {"ready": True, "elapsed": 1.5}

    store = ClusterStateStore(state_file=tmp_path / "clusters.json")
    bridge = ClusterBridge(
        session_name="test-cluster",
        profiles=["default", "personal_2"],
        gpu="T4",
        state_store=store,
        remote_coordinator=True,
    )

    res = bridge.deploy("Qwen/Qwen2.5-Coder-32B-Instruct", wait_for_ready=True)

    assert res["session_name"] == "test-cluster"
    assert res["endpoint_url"] == "http://localhost:8000/v1"
    assert res["total_vram_gb"] == 32
    assert len(res["nodes"]) == 2
    mock_coord.start_remote.assert_called_once()
    mock_adapter.assert_called_once_with(listen_port=8000, target_port=9000)

    # Verify state was saved
    saved = store.get("test-cluster")
    assert saved is not None
    assert saved.model_name == "Qwen/Qwen2.5-Coder-32B-Instruct"

    # Test teardown
    with patch("colab_agent_bridge.cluster.manager.stop_cluster_worker") as mock_stop:
        mock_stop.return_value = {"success": True}
        td_res = bridge.teardown()
        assert td_res["success"] is True
        assert len(td_res["nodes_released"]) == 2
        mock_coord.stop.assert_called_once()
        mock_server.shutdown.assert_called_once()
        assert store.get("test-cluster") is None


@patch("colab_agent_bridge.cluster.manager.SessionManager")
@patch("colab_agent_bridge.cluster.manager.check_health")
@patch("colab_agent_bridge.cluster.manager.start_adapter_server")
@patch("colab_agent_bridge.cluster.manager.ClusterCoordinator")
@patch("colab_agent_bridge.cluster.manager.provision_cluster_worker")
@patch("colab_agent_bridge.cluster.manager.validate_cluster_profiles")
def test_cluster_bridge_deploy_health_failure(
    mock_val,
    mock_prov,
    mock_coord_cls,
    mock_adapter,
    mock_health,
    mock_sm_cls,
    tmp_path,
):
    mock_val.return_value = [
        ProfileInfo(name="default", path=Path("/h"), authenticated=True),
        ProfileInfo(name="personal_2", path=Path("/p"), authenticated=True),
    ]
    mock_sm = MagicMock()
    mock_sm.verify_auth.return_value = {"authenticated": True}
    mock_sm.new_session.return_value = {"success": True}
    mock_sm_cls.return_value = mock_sm

    mock_coord_cls.return_value = MagicMock()
    mock_adapter.return_value = MagicMock()
    mock_health.return_value = {"ready": False, "error": "Cluster node unreachable"}

    store = ClusterStateStore(state_file=tmp_path / "clusters.json")
    bridge = ClusterBridge(
        session_name="test-fail",
        profiles=["default", "personal_2"],
        state_store=store,
    )

    with pytest.raises(RuntimeError, match="Cluster node unreachable"):
        bridge.deploy("Qwen/Qwen2.5-Coder-32B-Instruct", wait_for_ready=True)


# --- ClusterManager Methods Tests ---

def test_cluster_manager_methods(tmp_path):
    state_file = tmp_path / "clusters.json"
    store = ClusterStateStore(state_file=state_file)

    node = ClusterNode(profile="p1", session_name="s1-p1", local_rpc_port=50051)
    c = ClusterInfo(
        session_name="s1",
        model_name="model1",
        profiles=["p1", "p2"],
        gpu="T4",
        nodes=[node],
        coordinator_port=9000,
        listen_port=8000,
        endpoint_url="http://localhost:8000/v1",
    )
    store.save(c)

    # list
    clusters = ClusterManager.list_clusters(state_store=store)
    assert len(clusters) == 1

    # status
    stat = ClusterManager.get_cluster_status("s1", state_store=store)
    assert stat is not None
    assert stat.session_name == "s1"

    # teardown
    with patch("colab_agent_bridge.cluster.manager.stop_cluster_worker") as mock_stop:
        mock_stop.return_value = {"success": True}
        res = ClusterManager.teardown_cluster("s1", state_store=store)
        assert res["success"] is True
        assert store.get("s1") is None

    # teardown_all
    store.save(c)
    with patch("colab_agent_bridge.cluster.manager.stop_cluster_worker") as mock_stop:
        mock_stop.return_value = {"success": True}
        all_res = ClusterManager.teardown_all(state_store=store)
        assert all_res["clusters_terminated"] == 1
        assert store.list_all() == []


# --- CLI Cluster Subcommands Tests ---

def test_cli_cluster_help():
    res = runner.invoke(app, ["cluster", "--help"])
    assert res.exit_code == 0
    assert "up" in res.stdout
    assert "down" in res.stdout
    assert "list" in res.stdout
    assert "status" in res.stdout


def test_cli_cluster_up_success():
    with patch("colab_agent_bridge.cli.ClusterBridge") as mock_bridge_cls:
        mock_bridge = MagicMock()
        mock_bridge.deploy.return_value = {
            "session_name": "cluster-bridge",
            "endpoint_url": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "total_vram_gb": 32,
            "usable_vram_gb": 29,
            "tensor_split": "1,1",
            "nodes": [
                {"session_name": "cluster-bridge-default", "profile": "default", "local_rpc_port": 50051},
                {"session_name": "cluster-bridge-personal_2", "profile": "personal_2", "local_rpc_port": 50052},
            ],
            "harness": {"aider_command": "aider --model test"},
        }
        mock_bridge_cls.return_value = mock_bridge

        result = runner.invoke(
            app,
            [
                "cluster",
                "up",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "--profiles",
                "default,personal_2",
                "--gpu",
                "T4",
            ],
        )
        assert result.exit_code == 0
        assert "Distributed Cluster is LIVE" in result.stdout
        assert "32 GB" in result.stdout
        assert "http://localhost:8000/v1" in result.stdout
        assert "aider --model test" in result.stdout


def test_cli_cluster_up_failure():
    with patch("colab_agent_bridge.cli.ClusterBridge") as mock_bridge_cls:
        mock_bridge = MagicMock()
        mock_bridge.deploy.side_effect = RuntimeError("Worker installation failed")
        mock_bridge_cls.return_value = mock_bridge

        result = runner.invoke(
            app,
            [
                "cluster",
                "up",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "--profiles",
                "default,personal_2",
            ],
        )
        assert result.exit_code == 1
        assert "Cluster deployment failed" in result.stdout
        assert "colab-bridge cluster down" in result.stdout


def test_cli_cluster_down_single():
    with patch("colab_agent_bridge.cli.ClusterManager.teardown_cluster") as mock_td:
        mock_td.return_value = {"success": True, "nodes_released": ["n1", "n2"]}
        result = runner.invoke(app, ["cluster", "down", "--session", "test-cluster"])
        assert result.exit_code == 0
        assert "successfully released" in result.stdout


def test_cli_cluster_down_all():
    with patch("colab_agent_bridge.cli.ClusterManager.teardown_all") as mock_td:
        mock_td.return_value = {"clusters_terminated": 2, "nodes_released": ["n1", "n2", "n3", "n4"]}
        result = runner.invoke(app, ["cluster", "down", "--all"])
        assert result.exit_code == 0
        assert "Terminated 2 cluster(s)" in result.stdout


def test_cli_cluster_list():
    with patch("colab_agent_bridge.cli.ClusterManager.list_clusters") as mock_list:
        node = ClusterNode(profile="default", session_name="c-default", local_rpc_port=50051)
        mock_list.return_value = [
            ClusterInfo(
                session_name="cluster-qwen32b",
                model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
                profiles=["default", "personal_2"],
                gpu="T4",
                nodes=[node, node],
                coordinator_port=9000,
                listen_port=8000,
                endpoint_url="http://localhost:8000/v1",
                total_vram_gb=32,
            )
        ]
        result = runner.invoke(app, ["cluster", "list"], env={"COLUMNS": "160"})
        assert result.exit_code == 0
        assert "Distributed Colab Clusters" in result.stdout
        assert "cluster-qwen32b" in result.stdout
        assert "32 GB" in result.stdout


def test_cli_cluster_status():
    with patch("colab_agent_bridge.cli.ClusterManager.get_cluster_status") as mock_stat:
        node = ClusterNode(
            profile="default",
            session_name="c-default",
            local_rpc_port=50051,
            remote_rpc_port=50052,
            status="active",
        )
        mock_stat.return_value = ClusterInfo(
            session_name="cluster-qwen32b",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            profiles=["default", "personal_2"],
            gpu="T4",
            nodes=[node],
            coordinator_port=9000,
            listen_port=8000,
            endpoint_url="http://localhost:8000/v1",
            total_vram_gb=32,
            usable_vram_gb=29,
            status="active",
        )
        result = runner.invoke(app, ["cluster", "status", "-s", "cluster-qwen32b"])
        assert result.exit_code == 0
        assert "Cluster Status: cluster-qwen32b" in result.stdout
        assert "Total Pooled VRAM" in result.stdout
        assert "Worker Nodes" in result.stdout
        assert "50051" in result.stdout
