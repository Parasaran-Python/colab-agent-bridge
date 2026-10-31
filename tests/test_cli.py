from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from colab_agent_bridge.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "up" in result.stdout
    assert "down" in result.stdout
    assert "status" in result.stdout
    assert "recommend" in result.stdout
    assert "harness" in result.stdout
    assert "test" in result.stdout


def test_cli_recommend():
    result = runner.invoke(app, ["recommend", "Qwen/Qwen2.5-Coder-32B-Instruct", "--gpu", "T4"])
    assert result.exit_code == 0
    assert "T4" in result.stdout
    assert "Quantization required" in result.stdout or "True" in result.stdout


def test_cli_status_authenticated():
    with patch("colab_agent_bridge.cli.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.verify_auth.return_value = {"authenticated": True, "output": "ok"}
        mock_mgr.get_session_status.return_value = {"stdout": "Active session agent-bridge"}
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["status", "-s", "agent-bridge"])
        assert result.exit_code == 0
        assert "Google Colab CLI authenticated" in result.stdout
        assert "Active session agent-bridge" in result.stdout


def test_cli_status_unauthenticated():
    with patch("colab_agent_bridge.cli.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.verify_auth.return_value = {
            "authenticated": False,
            "error": "Colab binary not found",
            "remediation": "Ensure google-colab-cli is installed",
        }
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "Authentication Issue" in result.stdout
        assert "Ensure google-colab-cli is installed" in result.stdout


def test_cli_status_no_active_session():
    with patch("colab_agent_bridge.cli.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.verify_auth.return_value = {"authenticated": True}
        mock_mgr.get_session_status.return_value = {"stdout": ""}
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No active session found" in result.stdout


def test_cli_down_success():
    with patch("colab_agent_bridge.cli.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.stop_session.return_value = {"success": True}
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["down", "-s", "my-session"])
        assert result.exit_code == 0
        assert "successfully released" in result.stdout


def test_cli_down_failure():
    with patch("colab_agent_bridge.cli.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.stop_session.return_value = {"success": False, "stderr": "Session not found"}
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["down", "-s", "my-session"])
        assert result.exit_code == 0
        assert "Error stopping session" in result.stdout
        assert "Session not found" in result.stdout


def test_cli_harness():
    result = runner.invoke(app, ["harness", "--endpoint", "http://localhost:8000/v1", "--model", "Qwen/Qwen2.5-Coder-7B-Instruct"])
    assert result.exit_code == 0
    assert "OPENAI_BASE_URL" in result.stdout
    assert "aider" in result.stdout
    assert "Continue Config Block" in result.stdout


def test_cli_up_success():
    with patch("colab_agent_bridge.cli.ColabBridge") as mock_bridge_cls:
        mock_bridge = MagicMock()
        mock_bridge.deploy.return_value = {
            "endpoint_url": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "harness": {"aider_command": "aider --model test"},
        }
        mock_bridge_cls.return_value = mock_bridge

        result = runner.invoke(app, ["up", "Qwen/Qwen2.5-Coder-7B-Instruct", "-s", "agent-bridge", "-g", "T4"])
        assert result.exit_code == 0
        assert "Model is READY and live!" in result.stdout
        assert "http://localhost:8000/v1" in result.stdout
        assert "aider --model test" in result.stdout


def test_cli_up_failure():
    with patch("colab_agent_bridge.cli.ColabBridge") as mock_bridge_cls:
        mock_bridge = MagicMock()
        mock_bridge.deploy.side_effect = RuntimeError("Colab VM allocation failed")
        mock_bridge_cls.return_value = mock_bridge

        result = runner.invoke(app, ["up", "Qwen/Qwen2.5-Coder-7B-Instruct"])
        assert result.exit_code == 1
        assert "Deployment failed" in result.stdout


def test_cli_test_endpoint_success():
    with patch("colab_agent_bridge.cli.check_health") as mock_health:
        mock_health.return_value = {"ready": True, "elapsed": 0.3}
        result = runner.invoke(app, ["test", "--endpoint", "http://localhost:8000/v1"])
        assert result.exit_code == 0
        assert "Endpoint is healthy and responsive" in result.stdout


def test_cli_test_endpoint_failure():
    with patch("colab_agent_bridge.cli.check_health") as mock_health:
        mock_health.return_value = {"ready": False, "error": "Connection refused"}
        result = runner.invoke(app, ["test", "--endpoint", "http://localhost:8000/v1"])
        assert result.exit_code == 1
        assert "Health check failed" in result.stdout
