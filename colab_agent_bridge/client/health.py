import time
from typing import Dict, Any
import requests


def check_health(endpoint_url: str, timeout_seconds: int = 600, poll_interval: int = 5) -> Dict[str, Any]:
    """Polls the /models or /health endpoint until the server is ready or timeout expires."""
    url = endpoint_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    models_url = url + "/models"
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            resp = requests.get(models_url, timeout=3)
            if resp.status_code == 200:
                return {"ready": True, "elapsed": round(time.time() - start_time, 1), "data": resp.json()}
        except requests.RequestException:
            pass
        time.sleep(poll_interval)

    return {"ready": False, "elapsed": round(time.time() - start_time, 1), "error": "Health check timed out."}


def keepalive_ping(endpoint_url: str) -> bool:
    """Lightweight ping to prevent idle shutdown."""
    try:
        url = endpoint_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        resp = requests.get(url + "/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
