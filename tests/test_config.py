import os

from colab_agent_bridge.config import BridgeConfig, get_config, resolve_colab_binary


def test_default_config(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("COLAB_"):
            monkeypatch.delenv(key, raising=False)
    # When token.json does not exist, defaults to adc
    monkeypatch.setattr("os.path.exists", lambda path: False)
    config = get_config()
    assert isinstance(config, BridgeConfig)
    assert config.default_port == 8000
    assert config.default_backend == "vllm"
    assert config.default_gpu == "T4"
    assert config.auth_strategy == "adc"


def test_default_config_oauth2_detected(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("COLAB_"):
            monkeypatch.delenv(key, raising=False)
    # When token.json exists, auto-detects oauth2
    monkeypatch.setattr("os.path.exists", lambda path: True if "token.json" in path else False)
    config = get_config()
    assert config.auth_strategy == "oauth2"


def test_env_override_config(monkeypatch):
    monkeypatch.setenv("COLAB_CLI_PATH", "/custom/bin/colab")
    monkeypatch.setenv("COLAB_BRIDGE_PORT", "9000")
    monkeypatch.setenv("COLAB_DEFAULT_BACKEND", "ollama")
    config = get_config()
    assert config.colab_cli_path == "/custom/bin/colab"
    assert config.default_port == 9000
    assert config.default_backend == "ollama"


def test_env_override_all_fields(monkeypatch):
    monkeypatch.setenv("COLAB_CLI_PATH", "/bin/colab-custom")
    monkeypatch.setenv("COLAB_BRIDGE_PORT", "7777")
    monkeypatch.setenv("COLAB_DEFAULT_BACKEND", "llama_cpp")
    monkeypatch.setenv("COLAB_DEFAULT_GPU", "A100")
    monkeypatch.setenv("COLAB_AUTH_STRATEGY", "oauth2")
    config = get_config()
    assert config.colab_cli_path == "/bin/colab-custom"
    assert config.default_port == 7777
    assert config.default_backend == "llama_cpp"
    assert config.default_gpu == "A100"
    assert config.auth_strategy == "oauth2"


def test_resolve_colab_binary_fallbacks(monkeypatch):
    monkeypatch.delenv("COLAB_CLI_PATH", raising=False)

    # Case 1: DEFAULT_SDK_COLAB does not exist, which returns a path
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/colab" if cmd == "colab" else None)
    assert resolve_colab_binary() == "/usr/local/bin/colab"

    # Case 2: which returns None
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    assert resolve_colab_binary() == "colab"
