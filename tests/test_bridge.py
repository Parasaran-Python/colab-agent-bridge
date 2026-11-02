from unittest.mock import MagicMock, patch

import pytest

from colab_agent_bridge.client import (
    ColabBridge as ExportedBridge,
)
from colab_agent_bridge.client import (
    check_health,
    generate_agent_config,
    keepalive_ping,
)
from colab_agent_bridge.client.bridge import ColabBridge


def test_colab_bridge_init():
    bridge = ColabBridge(session_name="test-bridge", gpu="T4")
    assert bridge.session_name == "test-bridge"
    assert bridge.gpu == "T4"
    assert bridge.backend_name == "vllm"
    assert bridge.tunnel_type == "ssh"
    assert bridge.local_port == 8000
    assert bridge.get_endpoint() is None


def test_colab_bridge_custom_init():
    bridge = ColabBridge(
        session_name="custom-bridge",
        gpu=None,
        tpu="v2-8",
        high_mem=True,
        backend="ollama",
        tunnel_type="cloudflare",
        local_port=9000,
    )
    assert bridge.session_name == "custom-bridge"
    assert bridge.gpu is None
    assert bridge.tpu == "v2-8"
    assert bridge.high_mem is True
    assert bridge.backend_name == "ollama"
    assert bridge.tunnel_type == "cloudflare"
    assert bridge.local_port == 9000


@patch("colab_agent_bridge.client.bridge.start_adapter_server")
@patch("colab_agent_bridge.client.bridge.check_health")
@patch("colab_agent_bridge.client.bridge.SSHTunnel")
def test_colab_bridge_deploy_ssh_success(mock_ssh_cls, mock_health, mock_adapter):
    mock_server = MagicMock()
    mock_adapter.return_value = mock_server
    mock_tunnel = MagicMock()
    mock_tunnel.start.return_value = "http://localhost:8000/v1"
    mock_ssh_cls.return_value = mock_tunnel

    mock_health.return_value = {"ready": True, "elapsed": 1.2, "data": {}}

    bridge = ColabBridge(session_name="test-bridge", gpu="T4", backend="vllm", tunnel_type="ssh")
    bridge.session_mgr.verify_auth = MagicMock(return_value={"authenticated": True})
    bridge.session_mgr.new_session = MagicMock(return_value={"success": True, "stdout": "started"})
    bridge.executor.run_code = MagicMock(return_value={"success": True, "stdout": "ok"})

    result = bridge.deploy(model_name="Qwen/Qwen2.5-Coder-7B-Instruct", wait_for_ready=True)

    bridge.session_mgr.new_session.assert_called_once_with(
        session_name="test-bridge", gpu="T4", tpu=None, high_mem=False
    )
    assert bridge.executor.run_code.call_count == 2
    mock_tunnel.start.assert_called_once()
    mock_health.assert_called_once_with("http://localhost:8000/v1", timeout_seconds=600)

    assert result["session_name"] == "test-bridge"
    assert result["endpoint_url"] == "http://localhost:8000/v1"
    assert result["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert result["health"] == {"ready": True, "elapsed": 1.2, "data": {}}
    assert result["harness"]["endpoint_url"] == "http://localhost:8000/v1"
    assert bridge.get_endpoint() == "http://localhost:8000/v1"

    # Clean up keepalive
    bridge.teardown()


@patch("colab_agent_bridge.client.bridge.CloudflareTunnel")
def test_colab_bridge_deploy_cloudflare_no_wait(mock_cf_cls):
    mock_tunnel = MagicMock()
    mock_tunnel.start.return_value = "https://agent-cf.trycloudflare.com/v1"
    mock_cf_cls.return_value = mock_tunnel

    bridge = ColabBridge(session_name="test-cf", gpu="A100", backend="vllm", tunnel_type="cloudflare")
    bridge.session_mgr.verify_auth = MagicMock(return_value={"authenticated": True})
    bridge.session_mgr.new_session = MagicMock(return_value={"success": True, "stdout": "started"})
    bridge.executor.run_code = MagicMock(return_value={"success": True, "stdout": "ok"})

    result = bridge.deploy(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", wait_for_ready=False)

    assert result["endpoint_url"] == "https://agent-cf.trycloudflare.com/v1"
    assert result["health"] == {"ready": True}
    assert bridge.get_endpoint() == "https://agent-cf.trycloudflare.com/v1"

    bridge.teardown()


def test_colab_bridge_deploy_session_failure():
    bridge = ColabBridge(session_name="fail-bridge", gpu="T4")
    bridge.session_mgr.verify_auth = MagicMock(return_value={"authenticated": True})
    bridge.session_mgr.new_session = MagicMock(
        return_value={"success": False, "stderr": "Quota exceeded", "stdout": ""}
    )

    with pytest.raises(RuntimeError, match="Colab VM allocation failed: Quota exceeded"):
        bridge.deploy(model_name="test-model")


def test_colab_bridge_deploy_install_failure():
    bridge = ColabBridge(session_name="fail-install-bridge", gpu="T4")
    bridge.session_mgr.verify_auth = MagicMock(return_value={"authenticated": True})
    bridge.session_mgr.new_session = MagicMock(return_value={"success": True, "stdout": "VM ready"})
    bridge.executor.run_code = MagicMock(
        return_value={"success": False, "stderr": "pip install failed: out of disk space", "stdout": ""}
    )

    with pytest.raises(
        RuntimeError,
        match="Backend installation failed on Colab VM: pip install failed: out of disk space",
    ):
        bridge.deploy(model_name="test-model")


def test_colab_bridge_deploy_auth_failure():
    bridge = ColabBridge(session_name="auth-fail-bridge", gpu="T4")
    bridge.session_mgr.verify_auth = MagicMock(
        return_value={"authenticated": False, "error": "ADC missing", "remediation": "gcloud auth login"}
    )

    with pytest.raises(RuntimeError, match="Colab CLI authentication failed: ADC missing"):
        bridge.deploy(model_name="test-model")


def test_colab_bridge_teardown():
    bridge = ColabBridge(session_name="teardown-bridge")
    mock_tunnel = MagicMock()
    bridge.tunnel = mock_tunnel
    bridge._keepalive_active = True
    bridge.session_mgr.stop_session = MagicMock(return_value={"success": True})

    res = bridge.teardown()

    assert bridge._keepalive_active is False
    mock_tunnel.stop.assert_called_once()
    bridge.session_mgr.stop_session.assert_called_once_with("teardown-bridge")
    assert res == {"success": True}


def test_client_package_exports():
    assert ExportedBridge is ColabBridge
    assert callable(generate_agent_config)
    assert callable(check_health)
    assert callable(keepalive_ping)
