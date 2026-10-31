import json
from typing import Dict, Any


def generate_agent_config(endpoint_url: str, model_name: str) -> Dict[str, Any]:
    url = endpoint_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"

    env_snippet = f"""# Colab Agent Bridge Environment
OPENAI_BASE_URL="{url}"
OPENAI_API_KEY="colab"
MODEL_NAME="{model_name}"
"""

    continue_config = {
        "title": f"Colab: {model_name}",
        "model": model_name,
        "apiBase": url,
        "apiKey": "colab",
        "provider": "openai",
    }

    aider_cmd = f"aider --openai-api-base {url} --openai-api-key colab --model openai/{model_name}"

    return {
        "model": model_name,
        "endpoint_url": url,
        "env_snippet": env_snippet,
        "openai_client_kwargs": {
            "base_url": url,
            "api_key": "colab",
        },
        "continue_config": json.dumps(continue_config, indent=2),
        "aider_command": aider_cmd,
    }
