import subprocess
from unittest.mock import patch, MagicMock
import pytest

from colab_agent_bridge.config import BridgeConfig
from colab_agent_bridge.tunnels import BaseTunnel, SSHTunnel, CloudflareTunnel


def test_base_tunnel_abstract():
    with pytest.raises(TypeError):
        BaseTunnel()  # Can't instantiate abstract class with abstract methods


def test_ssh_tunnel_command_generation():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    tunnel = SSHTunnel(
        session_name="test-session",
        local_port=8000,
        remote_port=8000,
        config=config,
    )

    cmd = tunnel.build_ssh_command()
    assert "/mock/colab" in cmd
    assert "ssh" in cmd
    assert "-s" in cmd
    assert "test-session" in cmd
    assert "--proxy-mode" in cmd
    assert "-L 8000:localhost:8000" in " ".join(cmd)


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
def test_ssh_tunnel_start(mock_sleep, mock_popen):
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process

    tunnel = SSHTunnel(session_name="test-session", local_port=8888, remote_port=8000)
    endpoint = tunnel.start()

    assert endpoint == "http://localhost:8888/v1"
    mock_popen.assert_called_once()
    mock_sleep.assert_called_once_with(2)
    assert tunnel.process is mock_process


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
def test_ssh_tunnel_start_early_exit_failure(mock_sleep, mock_popen):
    mock_process = MagicMock()
    mock_process.poll.return_value = 255
    mock_process.returncode = 255
    mock_process.stderr.read.return_value = b"Host key verification failed\n"
    mock_popen.return_value = mock_process

    tunnel = SSHTunnel(session_name="test-session", local_port=8888, remote_port=8000)
    with pytest.raises(
        RuntimeError,
        match=r"SSH tunnel process exited immediately \(code 255\): Host key verification failed",
    ):
        tunnel.start()


def test_ssh_tunnel_stop_graceful():
    tunnel = SSHTunnel(session_name="test-session")
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # process is running
    tunnel.process = mock_process

    tunnel.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=3)
    mock_process.kill.assert_not_called()


def test_ssh_tunnel_stop_timeout():
    tunnel = SSHTunnel(session_name="test-session")
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # process is running
    mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="colab ssh", timeout=3)
    tunnel.process = mock_process

    tunnel.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=3)
    mock_process.kill.assert_called_once()


def test_ssh_tunnel_stop_already_dead_or_none():
    tunnel = SSHTunnel(session_name="test-session")
    tunnel.stop()  # Should not raise when process is None

    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Process already finished
    tunnel.process = mock_process
    tunnel.stop()
    mock_process.terminate.assert_not_called()


def test_cloudflare_tunnel_script_generation():
    tunnel = CloudflareTunnel(session_name="test-session", remote_port=8000)
    script = tunnel.generate_remote_script()
    assert "cloudflared" in script
    assert "http://localhost:8000" in script


def test_cloudflare_tunnel_start_success():
    tunnel = CloudflareTunnel(session_name="test-session", remote_port=8000)
    mock_executor = MagicMock()
    mock_executor.run_code.return_value = {
        "success": True,
        "stdout": "CLOUDFLARE_URL:https://subdomain-random-test.trycloudflare.com",
        "stderr": "",
        "returncode": 0,
    }
    tunnel.executor = mock_executor

    url = tunnel.start()
    assert url == "https://subdomain-random-test.trycloudflare.com/v1"
    assert tunnel.tunnel_url == "https://subdomain-random-test.trycloudflare.com/v1"
    mock_executor.run_code.assert_called_once()


def test_cloudflare_tunnel_start_failure():
    tunnel = CloudflareTunnel(session_name="test-session", remote_port=8000)
    mock_executor = MagicMock()
    mock_executor.run_code.return_value = {
        "success": False,
        "stdout": "CLOUDFLARE_FAILED",
        "stderr": "curl failed",
        "returncode": 1,
    }
    tunnel.executor = mock_executor

    with pytest.raises(RuntimeError, match="Failed to start Cloudflare Tunnel"):
        tunnel.start()


def test_cloudflare_tunnel_stop():
    tunnel = CloudflareTunnel(session_name="test-session", remote_port=8000)
    mock_executor = MagicMock()
    tunnel.executor = mock_executor

    tunnel.stop()
    mock_executor.run_code.assert_called_once()
    args, _ = mock_executor.run_code.call_args
    assert args[0] == "test-session"
    assert "pkill -f cloudflared" in args[1]
