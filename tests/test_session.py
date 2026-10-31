from unittest.mock import patch, MagicMock
from colab_agent_bridge.config import BridgeConfig
from colab_agent_bridge.core.session import SessionManager


def test_session_manager_command_construction():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Session created: bridge-test\n", stderr="")
        res = mgr.new_session(session_name="bridge-test", gpu="A100", high_mem=True)
        assert res["success"] is True

        args = mock_run.call_args[0][0]
        assert args[0] == "/mock/colab"
        assert "--auth=adc" in args
        assert "new" in args
        assert "-s" in args
        assert "bridge-test" in args
        assert "--gpu" in args
        assert "A100" in args
        assert "--high-mem" in args


def test_session_manager_tpu_construction():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Session created: tpu-session\n", stderr="")
        res = mgr.new_session(session_name="tpu-session", tpu="v5e1")
        assert res["success"] is True

        args = mock_run.call_args[0][0]
        assert "--tpu" in args
        assert "v5e1" in args
        assert "--gpu" not in args
        assert "--high-mem" not in args


def test_session_manager_default_config():
    mgr = SessionManager()
    assert mgr.config is not None
    assert mgr.config.auth_strategy == "adc"


def test_verify_auth_success():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="session-1 (ACTIVE)\n", stderr="")
        res = mgr.verify_auth()
        assert res["authenticated"] is True
        assert res["output"] == "session-1 (ACTIVE)"


def test_verify_auth_failure():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: ADC credentials missing")
        res = mgr.verify_auth()
        assert res["authenticated"] is False
        assert "credentials missing" in res["error"]
        assert "gcloud auth application-default login" in res["remediation"]


def test_verify_auth_file_not_found():
    config = BridgeConfig(colab_cli_path="/nonexistent/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run", side_effect=FileNotFoundError):
        res = mgr.verify_auth()
        assert res["authenticated"] is False
        assert "Colab binary not found" in res["error"]
        assert "remediation" in res


def test_stop_session():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Session stopped\n", stderr="")
        res = mgr.stop_session("test-s")
        assert res["success"] is True
        assert res["session_name"] == "test-s"

        args = mock_run.call_args[0][0]
        assert "stop" in args
        assert "-s" in args
        assert "test-s" in args


def test_list_sessions():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="session-1\nsession-2\n", stderr="")
        res = mgr.list_sessions()
        assert res["success"] is True
        assert "session-1" in res["stdout"]

        args = mock_run.call_args[0][0]
        assert "sessions" in args


def test_get_session_status():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    mgr = SessionManager(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Status: running\n", stderr="")
        res = mgr.get_session_status()
        assert res["success"] is True
        args = mock_run.call_args[0][0]
        assert "status" in args
        assert "-s" not in args

        res_session = mgr.get_session_status(session_name="my-session")
        assert res_session["success"] is True
        args2 = mock_run.call_args[0][0]
        assert "status" in args2
        assert "-s" in args2
        assert "my-session" in args2
