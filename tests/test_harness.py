import json
from unittest.mock import patch, MagicMock
import requests
from colab_agent_bridge.client.harness import generate_agent_config
from colab_agent_bridge.client.health import check_health, keepalive_ping


def test_generate_agent_config():
    conf = generate_agent_config(
        endpoint_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    assert 'OPENAI_BASE_URL="http://localhost:8000/v1"' in conf["env_snippet"]
    assert 'OPENAI_API_KEY="colab"' in conf["env_snippet"]
    assert conf["openai_client_kwargs"]["base_url"] == "http://localhost:8000/v1"
    assert conf["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert "continue_config" in conf
    assert "aider_command" in conf

    # Validate continue_config parses as valid json with expected keys
    continue_data = json.loads(conf["continue_config"])
    assert continue_data["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert continue_data["apiBase"] == "http://localhost:8000/v1"
    assert continue_data["apiKey"] == "colab"
    assert continue_data["provider"] == "openai"

    # Validate aider_command format
    assert "--openai-api-base http://localhost:8000/v1" in conf["aider_command"]
    assert "--model openai/Qwen/Qwen2.5-Coder-7B-Instruct" in conf["aider_command"]


def test_generate_agent_config_url_normalization():
    conf1 = generate_agent_config("http://localhost:8000", "model-x")
    assert conf1["endpoint_url"] == "http://localhost:8000/v1"

    conf2 = generate_agent_config("http://localhost:8000/", "model-x")
    assert conf2["endpoint_url"] == "http://localhost:8000/v1"

    conf3 = generate_agent_config("http://localhost:8000/v1/", "model-x")
    assert conf3["endpoint_url"] == "http://localhost:8000/v1"


def test_check_health_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"object": "list", "data": [{"id": "test-model"}]}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = check_health("http://localhost:8000/v1", timeout_seconds=5, poll_interval=1)
        assert res["ready"] is True
        assert "elapsed" in res
        assert res["data"] == {"object": "list", "data": [{"id": "test-model"}]}
        mock_get.assert_called_with("http://localhost:8000/v1/models", timeout=3)

        # Verify URL normalization without /v1
        res2 = check_health("http://localhost:8000", timeout_seconds=5, poll_interval=1)
        assert res2["ready"] is True
        mock_get.assert_called_with("http://localhost:8000/v1/models", timeout=3)


def test_check_health_timeout():
    with patch("requests.get", side_effect=requests.RequestException("Connection refused")):
        res = check_health("http://localhost:8000/v1", timeout_seconds=0.1, poll_interval=0.05)
        assert res["ready"] is False
        assert "elapsed" in res
        assert res["error"] == "Health check timed out."


def test_keepalive_ping_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("requests.get", return_value=mock_resp) as mock_get:
        assert keepalive_ping("http://localhost:8000/v1") is True
        mock_get.assert_called_with("http://localhost:8000/v1/models", timeout=5)

        # Verify URL normalization without /v1
        assert keepalive_ping("http://localhost:8000") is True
        mock_get.assert_called_with("http://localhost:8000/v1/models", timeout=5)


def test_keepalive_ping_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    with patch("requests.get", return_value=mock_resp):
        assert keepalive_ping("http://localhost:8000/v1") is False

    with patch("requests.get", side_effect=requests.RequestException("error")):
        assert keepalive_ping("http://localhost:8000/v1") is False
