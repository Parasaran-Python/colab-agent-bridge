from unittest.mock import patch, MagicMock
from colab_agent_bridge.config import BridgeConfig
from colab_agent_bridge.core.executor import RemoteExecutor


def test_executor_run_code():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    executor = RemoteExecutor(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="hello from remote\n", stderr="")
        output = executor.run_code(session_name="test-s", code="print('hello from remote')")
        assert output["stdout"] == "hello from remote"
        assert output["success"] is True
        assert output["returncode"] == 0

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        kwargs = mock_run.call_args[1]
        assert args[0] == "/mock/colab"
        assert "--auth=adc" in args
        assert "exec" in args
        assert "-s" in args
        assert "test-s" in args
        assert kwargs["input"] == "print('hello from remote')"


def test_executor_run_code_failure():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    executor = RemoteExecutor(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="SyntaxError: invalid syntax")
        output = executor.run_code(session_name="test-s", code="invalid syntax ((")
        assert output["success"] is False
        assert output["returncode"] == 1
        assert "SyntaxError" in output["stderr"]


def test_executor_run_script():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    executor = RemoteExecutor(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="script output\n", stderr="")
        output = executor.run_script(session_name="test-s", script_path="/tmp/run.py")
        assert output["success"] is True
        assert output["stdout"] == "script output"

        args = mock_run.call_args[0][0]
        assert "exec" in args
        assert "-s" in args
        assert "test-s" in args
        assert "-f" in args
        assert "/tmp/run.py" in args


def test_executor_install_packages():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    executor = RemoteExecutor(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Installed vllm\n", stderr="")
        output = executor.install_packages(session_name="test-s", packages=["vllm", "torch"])
        assert output["success"] is True
        assert output["stdout"] == "Installed vllm"

        args = mock_run.call_args[0][0]
        assert "install" in args
        assert "-s" in args
        assert "test-s" in args
        assert "vllm" in args
        assert "torch" in args


def test_executor_default_config():
    executor = RemoteExecutor()
    assert executor.config is not None
    assert executor.config.auth_strategy in ("adc", "oauth2")
