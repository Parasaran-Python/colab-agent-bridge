import os
import shutil
from dataclasses import dataclass

DEFAULT_SDK_COLAB = "/run/media/parasaran/Dev/SDK/python/install/bin/colab"
DEFAULT_SDK_LLAMA_SERVER = "/run/media/parasaran/Dev/SDK/python/install/bin/llama-server"


@dataclass
class BridgeConfig:
    colab_cli_path: str
    default_port: int = 8000
    default_backend: str = "vllm"
    default_gpu: str = "T4"
    auth_strategy: str = "adc"


def resolve_colab_binary() -> str:
    env_path = os.getenv("COLAB_CLI_PATH")
    if env_path:
        return env_path
    if os.path.exists(DEFAULT_SDK_COLAB):
        return DEFAULT_SDK_COLAB
    which_path = shutil.which("colab")
    if which_path:
        return which_path
    return "colab"


def resolve_llama_server_binary() -> str:
    env_path = os.getenv("LLAMA_SERVER_PATH")
    if env_path:
        return env_path
    if os.path.exists(DEFAULT_SDK_LLAMA_SERVER):
        return DEFAULT_SDK_LLAMA_SERVER
    which_path = shutil.which("llama-server")
    if which_path:
        return which_path
    return "llama-server"


def resolve_auth_strategy() -> str:
    env_strat = os.getenv("COLAB_AUTH_STRATEGY")
    if env_strat:
        return env_strat
    token_path = os.path.expanduser("~/.config/colab-cli/token.json")
    if os.path.exists(token_path):
        return "oauth2"
    return "adc"


def get_config() -> BridgeConfig:
    return BridgeConfig(
        colab_cli_path=resolve_colab_binary(),
        default_port=int(os.getenv("COLAB_BRIDGE_PORT", "8000")),
        default_backend=os.getenv("COLAB_DEFAULT_BACKEND", "vllm"),
        default_gpu=os.getenv("COLAB_DEFAULT_GPU", "T4"),
        auth_strategy=resolve_auth_strategy(),
    )
