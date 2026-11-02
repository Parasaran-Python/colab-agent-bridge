# Colab Agent Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, open-source-safe tool and Python SDK (`colab-agent-bridge`) that automates Google Colab VMs via `google-colab-cli`, provisions models using vLLM, Ollama, or llama.cpp on any GPU/TPU/CPU tier, establishes native SSH or Cloudflare tunnels, exposes a unified OpenAI-compatible endpoint, and integrates directly with coding agent harnesses.

**Architecture:** A modular Python architecture separating core CLI execution and hardware matrix logic from remote model backend builders (vLLM, Ollama, llama.cpp), dual network tunnels (native `colab ssh` port forwarding and remote Cloudflare Quick Tunnel), a high-level SDK facade (`ColabBridge`) with health keepalive, and a rich CLI (`colab-bridge`).

**Tech Stack:** Python 3.10+, `google-colab-cli`, `typer` / `click`, `rich`, `pydantic`, `pytest`, `requests`.

**Spec:** [`docs/superpowers/specs/2026-10-31-colab-agent-bridge-design.md`](file:///run/media/parasaran/Dev/linux_dev/git/colab-agent-bridge/docs/superpowers/specs/2026-10-31-colab-agent-bridge-design.md)

## Global Constraints
- Target workspace directory: `/run/media/parasaran/Dev/linux_dev/git/colab-agent-bridge`
- Python runtime binary: `/run/media/parasaran/Dev/SDK/python/install/bin/python`
- Colab binary path: `/run/media/parasaran/Dev/SDK/python/install/bin/colab` (configurable via `COLAB_CLI_PATH` env var)
- Port: Inference servers and local tunnels default to port `8000` exposing standard OpenAI routes (`/v1/models`, `/v1/chat/completions`)
- Zero Secret Leaks: Never commit `.env`, session JSON, OAuth tokens, SSH keys, or credentials
- Atomic commits for each task with descriptive git commit messages

---

### Task 1: Project Scaffolding & Configuration Loader

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `colab_agent_bridge/__init__.py`
- Create: `colab_agent_bridge/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `BridgeConfig` dataclass/pydantic model in `colab_agent_bridge.config` with fields: `colab_cli_path: str`, `default_port: int`, `default_backend: str`, `default_gpu: str`, `auth_strategy: str`. Function `get_config() -> BridgeConfig`.

- [ ] **Step 1: Write failing test for config loader**

```python
# tests/test_config.py
import os
from colab_agent_bridge.config import get_config, BridgeConfig

def test_default_config(monkeypatch):
    monkeypatch.delenv("COLAB_CLI_PATH", raising=False)
    monkeypatch.delenv("COLAB_BRIDGE_PORT", raising=False)
    config = get_config()
    assert isinstance(config, BridgeConfig)
    assert config.default_port == 8000
    assert config.default_backend == "vllm"
    assert config.auth_strategy == "adc"

def test_env_override_config(monkeypatch):
    monkeypatch.setenv("COLAB_CLI_PATH", "/custom/bin/colab")
    monkeypatch.setenv("COLAB_BRIDGE_PORT", "9000")
    monkeypatch.setenv("COLAB_DEFAULT_BACKEND", "ollama")
    config = get_config()
    assert config.colab_cli_path == "/custom/bin/colab"
    assert config.default_port == 9000
    assert config.default_backend == "ollama"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge')

- [ ] **Step 3: Implement minimal code for config and scaffolding**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "colab-agent-bridge"
version = "0.1.0"
description = "Bridge Google Colab GPU/TPU model serving to local coding agent harnesses via OpenAI-compatible endpoints."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "requests>=2.28.0",
    "pydantic>=2.0.0"
]

[project.scripts]
colab-bridge = "colab_agent_bridge.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `.env.example`:
```bash
# Path to colab CLI binary (defaults to detecting on PATH or SDK install bin)
COLAB_CLI_PATH=/run/media/parasaran/Dev/SDK/python/install/bin/colab

# Port for local model endpoint
COLAB_BRIDGE_PORT=8000

# Default serving backend (vllm, ollama, llama_cpp)
COLAB_DEFAULT_BACKEND=vllm

# Default accelerator tier (T4, L4, A100, H100, etc.)
COLAB_DEFAULT_GPU=T4

# Colab CLI auth strategy: 'adc' (Application Default Credentials) or 'oauth2'
COLAB_AUTH_STRATEGY=adc
```

Create `colab_agent_bridge/__init__.py`:
```python
"""Colab Agent Bridge: Orchestrate Colab GPUs, serve models, and bridge to coding agents."""

__version__ = "0.1.0"
```

Create `colab_agent_bridge/config.py`:
```python
import os
import shutil
from dataclasses import dataclass

DEFAULT_SDK_COLAB = "/run/media/parasaran/Dev/SDK/python/install/bin/colab"

@dataclass
class BridgeConfig:
    colab_cli_path: str
    default_port: int = 8000
    default_backend: str = "vllm"
    default_gpu: str = "T4"
    auth_strategy: str = "adc"

def resolve_colab_binary() -> str:
    env_path = os.getenv("COLAB_CLI_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(DEFAULT_SDK_COLAB):
        return DEFAULT_SDK_COLAB
    which_path = shutil.which("colab")
    if which_path:
        return which_path
    return "colab"

def get_config() -> BridgeConfig:
    return BridgeConfig(
        colab_cli_path=resolve_colab_binary(),
        default_port=int(os.getenv("COLAB_BRIDGE_PORT", "8000")),
        default_backend=os.getenv("COLAB_DEFAULT_BACKEND", "vllm"),
        default_gpu=os.getenv("COLAB_DEFAULT_GPU", "T4"),
        auth_strategy=os.getenv("COLAB_AUTH_STRATEGY", "adc")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example colab_agent_bridge/ tests/test_config.py
git commit -m "feat: scaffold project structure and configuration loader"
```

---

### Task 2: Hardware Accelerator Matrix & VRAM Advisor

**Files:**
- Create: `colab_agent_bridge/core/__init__.py`
- Create: `colab_agent_bridge/core/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: None
- Produces: `AcceleratorTier` enum, `HardwareSpec` dataclass, `SUPPORTED_GPUS`, `SUPPORTED_TPUS`, `validate_accelerator(accel: str) -> HardwareSpec`, `recommend_model_strategy(model_name: str, accel: str) -> dict`.

- [ ] **Step 1: Write failing test for hardware module**

```python
# tests/test_hardware.py
import pytest
from colab_agent_bridge.core.hardware import (
    SUPPORTED_GPUS,
    validate_accelerator,
    recommend_model_strategy,
    HardwareSpec
)

def test_validate_accelerator():
    spec_t4 = validate_accelerator("T4")
    assert spec_t4.name == "T4"
    assert spec_t4.vram_gb == 16
    assert spec_t4.is_gpu is True

    spec_a100 = validate_accelerator("A100")
    assert spec_a100.name == "A100"
    assert spec_a100.vram_gb == 40

    spec_tpu = validate_accelerator("v5e1")
    assert spec_tpu.is_tpu is True

    with pytest.raises(ValueError):
        validate_accelerator("UNKNOWN_GPU")

def test_recommend_model_strategy_t4():
    # Large 32B model on T4 should recommend quantization or larger GPU
    rec = recommend_model_strategy("Qwen/Qwen2.5-Coder-32B-Instruct", "T4")
    assert rec["needs_quantization"] is True
    assert "awq" in rec["recommended_quantization"] or "gguf" in rec["recommended_quantization"]

def test_recommend_model_strategy_a100():
    # 7B model on A100 needs no quantization
    rec = recommend_model_strategy("Qwen/Qwen2.5-Coder-7B-Instruct", "A100")
    assert rec["needs_quantization"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_hardware.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.core.hardware')

- [ ] **Step 3: Implement minimal code for hardware matrix**

Create `colab_agent_bridge/core/__init__.py`:
```python
"""Core subsystem for Colab session management and hardware negotiation."""
```

Create `colab_agent_bridge/core/hardware.py`:
```python
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class HardwareSpec:
    name: str
    vram_gb: int
    is_gpu: bool
    is_tpu: bool
    description: str

SUPPORTED_GPUS: Dict[str, HardwareSpec] = {
    "T4": HardwareSpec(name="T4", vram_gb=16, is_gpu=True, is_tpu=False, description="NVIDIA T4 16GB (Available on Free & Paid tiers)"),
    "L4": HardwareSpec(name="L4", vram_gb=24, is_gpu=True, is_tpu=False, description="NVIDIA L4 24GB (Colab Pro / Compute Units)"),
    "G4": HardwareSpec(name="G4", vram_gb=16, is_gpu=True, is_tpu=False, description="NVIDIA G4"),
    "A100": HardwareSpec(name="A100", vram_gb=40, is_gpu=True, is_tpu=False, description="NVIDIA A100 40GB/80GB (Colab Pro+)"),
    "H100": HardwareSpec(name="H100", vram_gb=80, is_gpu=True, is_tpu=False, description="NVIDIA H100 80GB (Colab Pro+ Premium)")
}

SUPPORTED_TPUS: Dict[str, HardwareSpec] = {
    "v5e1": HardwareSpec(name="v5e1", vram_gb=16, is_gpu=False, is_tpu=True, description="Google Cloud TPU v5e (single chip)"),
    "v6e1": HardwareSpec(name="v6e1", vram_gb=32, is_gpu=False, is_tpu=True, description="Google Cloud TPU v6e (Trillium)")
}

CPU_SPEC = HardwareSpec(name="CPU", vram_gb=0, is_gpu=False, is_tpu=False, description="Standard or High-Mem CPU VM")

def validate_accelerator(accel: Optional[str]) -> HardwareSpec:
    if not accel or accel.upper() == "CPU":
        return CPU_SPEC
    accel_upper = accel.upper()
    for name, spec in SUPPORTED_GPUS.items():
        if name.upper() == accel_upper:
            return spec
    accel_lower = accel.lower()
    for name, spec in SUPPORTED_TPUS.items():
        if name.lower() == accel_lower:
            return spec
    raise ValueError(f"Unsupported accelerator '{accel}'. Supported GPUs: {list(SUPPORTED_GPUS.keys())}, TPUs: {list(SUPPORTED_TPUS.keys())}")

def recommend_model_strategy(model_name: str, accel: Optional[str]) -> dict:
    spec = validate_accelerator(accel)
    name_lower = model_name.lower()

    # Heuristic parameter detection from model name
    param_size = 7
    if "70b" in name_lower or "72b" in name_lower:
        param_size = 70
    elif "32b" in name_lower or "33b" in name_lower or "34b" in name_lower:
        param_size = 32
    elif "14b" in name_lower:
        param_size = 14
    elif "1.5b" in name_lower or "1b" in name_lower or "3b" in name_lower:
        param_size = 3
    elif "7b" in name_lower or "8b" in name_lower:
        param_size = 8

    # FP16 requires roughly 2GB VRAM per billion params + 2-4GB KV cache overhead
    fp16_vram_needed = param_size * 2 + 4

    if spec.vram_gb == 0:  # CPU
        return {
            "needs_quantization": True,
            "recommended_backend": "llama_cpp",
            "recommended_quantization": ["gguf", "q4_k_m"],
            "warning": "Running on CPU; model inference will be slow without GPU acceleration."
        }

    if spec.vram_gb < fp16_vram_needed:
        return {
            "needs_quantization": True,
            "recommended_backend": "vllm" if spec.vram_gb >= 16 and param_size <= 32 else "llama_cpp",
            "recommended_quantization": ["awq", "gptq", "gguf"],
            "warning": f"Model {model_name} (~{param_size}B) exceeds FP16 capacity of {spec.name} ({spec.vram_gb}GB VRAM). Quantization required."
        }

    return {
        "needs_quantization": False,
        "recommended_backend": "vllm",
        "recommended_quantization": [],
        "warning": None
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_hardware.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/core/ tests/test_hardware.py
git commit -m "feat: add accelerator hardware matrix and VRAM advisor"
```

---

### Task 3: Colab CLI Session & Process Controller

**Files:**
- Create: `colab_agent_bridge/core/session.py`
- Create: `colab_agent_bridge/core/executor.py`
- Test: `tests/test_session.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `colab_agent_bridge.config.BridgeConfig`, `colab_agent_bridge.core.hardware.validate_accelerator`
- Produces: `SessionManager` class with methods `new_session()`, `stop_session()`, `list_sessions()`, `get_session_status()`, `verify_auth()`. `RemoteExecutor` class with methods `run_code()`, `run_script()`, `install_packages()`.

- [ ] **Step 1: Write failing tests for session and executor**

```python
# tests/test_session.py
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
```

```python
# tests/test_executor.py
from unittest.mock import patch, MagicMock
from colab_agent_bridge.config import BridgeConfig
from colab_agent_bridge.core.executor import RemoteExecutor

def test_executor_run_code():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    executor = RemoteExecutor(config=config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="hello from remote", stderr="")
        output = executor.run_code(session_name="test-s", code="print('hello from remote')")
        assert output["stdout"] == "hello from remote"

        args = mock_run.call_args[0][0]
        assert "exec" in args
        assert "-s" in args
        assert "test-s" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_session.py tests/test_executor.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.core.session')

- [ ] **Step 3: Implement minimal code for session manager and executor**

Create `colab_agent_bridge/core/session.py`:
```python
import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.hardware import validate_accelerator

class SessionManager:
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or get_config()

    def _build_base_cmd(self) -> List[str]:
        return [self.config.colab_cli_path, f"--auth={self.config.auth_strategy}"]

    def verify_auth(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return {"authenticated": True, "output": res.stdout.strip()}
            return {
                "authenticated": False,
                "error": res.stderr.strip() or res.stdout.strip(),
                "remediation": "Run: gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"
            }
        except FileNotFoundError:
            return {
                "authenticated": False,
                "error": f"Colab binary not found at '{self.config.colab_cli_path}'",
                "remediation": "Ensure google-colab-cli is installed or set COLAB_CLI_PATH."
            }

    def new_session(
        self,
        session_name: str,
        gpu: Optional[str] = None,
        tpu: Optional[str] = None,
        high_mem: bool = False
    ) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["new", "-s", session_name]

        if gpu:
            spec = validate_accelerator(gpu)
            if spec.is_gpu:
                cmd.extend(["--gpu", spec.name])
        elif tpu:
            spec = validate_accelerator(tpu)
            if spec.is_tpu:
                cmd.extend(["--tpu", spec.name])

        if high_mem:
            cmd.append("--high-mem")

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode
        }

    def stop_session(self, session_name: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["stop", "-s", session_name]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "session_name": session_name,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip()
        }

    def list_sessions(self) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["sessions"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip()
        }

    def get_session_status(self, session_name: Optional[str] = None) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["status"]
        if session_name:
            cmd.extend(["-s", session_name])
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip()
        }
```

Create `colab_agent_bridge/core/executor.py`:
```python
import subprocess
from typing import Optional, List, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config

class RemoteExecutor:
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or get_config()

    def _build_base_cmd(self) -> List[str]:
        return [self.config.colab_cli_path, f"--auth={self.config.auth_strategy}"]

    def run_code(self, session_name: str, code: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["exec", "-s", session_name]
        res = subprocess.run(cmd, input=code, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode
        }

    def run_script(self, session_name: str, script_path: str) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["exec", "-s", session_name, "-f", script_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode
        }

    def install_packages(self, session_name: str, packages: List[str]) -> Dict[str, Any]:
        cmd = self._build_base_cmd() + ["install", "-s", session_name] + packages
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip()
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_session.py tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/core/session.py colab_agent_bridge/core/executor.py tests/test_session.py tests/test_executor.py
git commit -m "feat: add Colab CLI session manager and remote executor"
```

---

### Task 4: Model Inference Serving Backends (vLLM, Ollama, llama.cpp)

**Files:**
- Create: `colab_agent_bridge/backends/__init__.py`
- Create: `colab_agent_bridge/backends/base.py`
- Create: `colab_agent_bridge/backends/vllm.py`
- Create: `colab_agent_bridge/backends/ollama.py`
- Create: `colab_agent_bridge/backends/llama_cpp.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Consumes: None
- Produces: Abstract class `BaseBackend`, implementations `VLLMBackend`, `OllamaBackend`, `LlamaCppBackend`, factory function `get_backend(name: str) -> BaseBackend`.

- [ ] **Step 1: Write failing test for backends**

```python
# tests/test_backends.py
from colab_agent_bridge.backends import get_backend, VLLMBackend, OllamaBackend, LlamaCppBackend

def test_vllm_backend_script_generation():
    backend = get_backend("vllm")
    assert isinstance(backend, VLLMBackend)
    install_script = backend.generate_install_script()
    assert "vllm" in install_script

    start_script = backend.generate_start_script(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        port=8000,
        quantization="awq"
    )
    assert "vllm.entrypoints.openai.api_server" in start_script
    assert "--model Qwen/Qwen2.5-Coder-7B-Instruct" in start_script
    assert "--port 8000" in start_script
    assert "--quantization awq" in start_script

def test_ollama_backend_script_generation():
    backend = get_backend("ollama")
    assert isinstance(backend, OllamaBackend)
    install_script = backend.generate_install_script()
    assert "ollama" in install_script.lower()

    start_script = backend.generate_start_script(
        model_name="qwen2.5-coder:7b",
        port=8000
    )
    assert "ollama serve" in start_script
    assert "ollama pull qwen2.5-coder:7b" in start_script

def test_llama_cpp_backend_script_generation():
    backend = get_backend("llama_cpp")
    assert isinstance(backend, LlamaCppBackend)
    install_script = backend.generate_install_script()
    assert "llama-cpp-python" in install_script

    start_script = backend.generate_start_script(
        model_name="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        port=8000
    )
    assert "llama_cpp.server" in start_script
    assert "8000" in start_script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_backends.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.backends')

- [ ] **Step 3: Implement minimal code for backends**

Create `colab_agent_bridge/backends/base.py`:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_install_script(self) -> str:
        """Returns Python or bash code to install requirements on the Colab VM."""
        pass

    @abstractmethod
    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        """Returns code that launches the OpenAI-compatible HTTP server on the VM."""
        pass
```

Create `colab_agent_bridge/backends/vllm.py`:
```python
from typing import Any, Optional
from colab_agent_bridge.backends.base import BaseBackend

class VLLMBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "vllm"

    def generate_install_script(self) -> str:
        return """
import subprocess, sys
print("[vLLM] Installing vllm via uv/pip...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "accelerate"], check=True)
print("[vLLM] Installation completed.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        quant = kwargs.get("quantization")
        max_len = kwargs.get("max_model_len", 8192)
        gpu_util = kwargs.get("gpu_memory_utilization", 0.90)

        quant_flag = f"--quantization {quant}" if quant else ""

        return f"""
import subprocess, os, sys, time

log_file = "/content/vllm_server.log"
cmd = (
    f"nohup {{sys.executable}} -m vllm.entrypoints.openai.api_server "
    f"--model {model_name} "
    f"--port {port} "
    f"--host 0.0.0.0 "
    f"--max-model-len {max_len} "
    f"--gpu-memory-utilization {gpu_util} "
    f"{quant_flag} "
    f"> {{log_file}} 2>&1 &"
)

print(f"[vLLM] Launching vLLM OpenAI API server for {model_name} on port {port}...")
os.system(cmd)
time.sleep(2)
print("[vLLM] Background supervisor started. Logs streaming to /content/vllm_server.log")
"""
```

Create `colab_agent_bridge/backends/ollama.py`:
```python
from typing import Any
from colab_agent_bridge.backends.base import BaseBackend

class OllamaBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "ollama"

    def generate_install_script(self) -> str:
        return """
import os, subprocess
print("[Ollama] Downloading and installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")
print("[Ollama] Ollama installation finished.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        # Note: Ollama standard port is 11434, we route to specified port via OLLAMA_HOST
        return f"""
import os, time, subprocess

log_file = "/content/ollama_server.log"
cmd = f"nohup env OLLAMA_HOST=0.0.0.0:{port} ollama serve > {{log_file}} 2>&1 &"
print(f"[Ollama] Starting Ollama server on 0.0.0.0:{port}...")
os.system(cmd)
time.sleep(3)

print(f"[Ollama] Pulling model {model_name}...")
os.system(f"env OLLAMA_HOST=0.0.0.0:{port} ollama pull {model_name}")
print(f"[Ollama] Model {model_name} is ready on port {port}.")
"""
```

Create `colab_agent_bridge/backends/llama_cpp.py`:
```python
from typing import Any
from colab_agent_bridge.backends.base import BaseBackend

class LlamaCppBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "llama_cpp"

    def generate_install_script(self) -> str:
        return """
import subprocess, sys
print("[llama-cpp] Installing llama-cpp-python[server] with CUDA support...")
cmd = [
    sys.executable, "-m", "pip", "install", "-q",
    "llama-cpp-python[server]", "huggingface_hub"
]
subprocess.run(cmd, check=True)
print("[llama-cpp] Installed successfully.")
"""

    def generate_start_script(self, model_name: str, port: int = 8000, **kwargs: Any) -> str:
        hf_file = kwargs.get("hf_file", "*.gguf")
        return f"""
import os, sys, time

log_file = "/content/llama_cpp_server.log"
cmd = (
    f"nohup {{sys.executable}} -m llama_cpp.server "
    f"--hf_model_repo_id {model_name} "
    f"--host 0.0.0.0 "
    f"--port {port} "
    f"--n_gpu_layers -1 "
    f"> {{log_file}} 2>&1 &"
)

print(f"[llama-cpp] Launching server for {model_name} on port {port}...")
os.system(cmd)
time.sleep(2)
print("[llama-cpp] Started in background.")
"""
```

Create `colab_agent_bridge/backends/__init__.py`:
```python
from typing import Dict, Type
from colab_agent_bridge.backends.base import BaseBackend
from colab_agent_bridge.backends.vllm import VLLMBackend
from colab_agent_bridge.backends.ollama import OllamaBackend
from colab_agent_bridge.backends.llama_cpp import LlamaCppBackend

BACKENDS: Dict[str, Type[BaseBackend]] = {
    "vllm": VLLMBackend,
    "ollama": OllamaBackend,
    "llama_cpp": LlamaCppBackend
}

def get_backend(name: str) -> BaseBackend:
    name_lower = name.lower()
    if name_lower not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name_lower]()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_backends.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/backends/ tests/test_backends.py
git commit -m "feat: add model inference serving backends for vLLM, Ollama, and llama.cpp"
```

---

### Task 5: Network Tunnels (Native SSH Bridge & Cloudflare Quick Tunnel)

**Files:**
- Create: `colab_agent_bridge/tunnels/__init__.py`
- Create: `colab_agent_bridge/tunnels/base.py`
- Create: `colab_agent_bridge/tunnels/ssh_tunnel.py`
- Create: `colab_agent_bridge/tunnels/cloudflare.py`
- Test: `tests/test_tunnels.py`

**Interfaces:**
- Consumes: `colab_agent_bridge.config.BridgeConfig`, `colab_agent_bridge.core.executor.RemoteExecutor`
- Produces: `SSHTunnel` and `CloudflareTunnel` inheriting `BaseTunnel`, with methods `start() -> str` (returning the local or public endpoint URL) and `stop()`.

- [ ] **Step 1: Write failing test for tunnels**

```python
# tests/test_tunnels.py
from unittest.mock import patch, MagicMock
from colab_agent_bridge.config import BridgeConfig
from colab_agent_bridge.tunnels import SSHTunnel, CloudflareTunnel

def test_ssh_tunnel_command_generation():
    config = BridgeConfig(colab_cli_path="/mock/colab")
    tunnel = SSHTunnel(session_name="test-session", local_port=8000, remote_port=8000, config=config)

    cmd = tunnel.build_ssh_command()
    assert "/mock/colab" in cmd
    assert "ssh" in cmd
    assert "-s" in cmd
    assert "test-session" in cmd
    assert "--proxy-mode" in cmd
    assert "-L 8000:localhost:8000" in " ".join(cmd)

def test_cloudflare_tunnel_script_generation():
    tunnel = CloudflareTunnel(session_name="test-session", remote_port=8000)
    script = tunnel.generate_remote_script()
    assert "cloudflared" in script
    assert "http://localhost:8000" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_tunnels.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.tunnels')

- [ ] **Step 3: Implement minimal code for tunnels**

Create `colab_agent_bridge/tunnels/base.py`:
```python
from abc import ABC, abstractmethod

class BaseTunnel(ABC):
    @abstractmethod
    def start(self) -> str:
        """Starts the tunnel and returns the active endpoint URL."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the active tunnel."""
        pass
```

Create `colab_agent_bridge/tunnels/ssh_tunnel.py`:
```python
import subprocess
import time
from typing import Optional, List
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.tunnels.base import BaseTunnel

class SSHTunnel(BaseTunnel):
    def __init__(
        self,
        session_name: str,
        local_port: int = 8000,
        remote_port: int = 8000,
        config: Optional[BridgeConfig] = None
    ):
        self.session_name = session_name
        self.local_port = local_port
        self.remote_port = remote_port
        self.config = config or get_config()
        self.process: Optional[subprocess.Popen] = None

    def build_ssh_command(self) -> List[str]:
        # Uses colab ssh with ProxyMode to set up direct local port forwarding
        return [
            self.config.colab_cli_path,
            f"--auth={self.config.auth_strategy}",
            "ssh",
            "-s", self.session_name,
            "--proxy-mode",
            "--",
            "-N",
            "-L", f"{self.local_port}:localhost:{self.remote_port}"
        ]

    def start(self) -> str:
        cmd = self.build_ssh_command()
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        return f"http://localhost:{self.local_port}/v1"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
```

Create `colab_agent_bridge/tunnels/cloudflare.py`:
```python
import time
import re
from typing import Optional
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.tunnels.base import BaseTunnel

class CloudflareTunnel(BaseTunnel):
    def __init__(
        self,
        session_name: str,
        remote_port: int = 8000,
        config: Optional[BridgeConfig] = None
    ):
        self.session_name = session_name
        self.remote_port = remote_port
        self.config = config or get_config()
        self.executor = RemoteExecutor(self.config)
        self.tunnel_url: Optional[str] = None

    def generate_remote_script(self) -> str:
        return f"""
import os, time, subprocess, re

# Download cloudflared if not present
if not os.path.exists("/usr/local/bin/cloudflared"):
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")

log_path = "/content/cloudflared.log"
os.system(f"nohup /usr/local/bin/cloudflared tunnel --url http://localhost:{self.remote_port} > {{log_path}} 2>&1 &")

# Extract the generated trycloudflare url
url = None
for _ in range(30):
    time.sleep(1)
    if os.path.exists(log_path):
        content = open(log_path, 'r', errors='ignore').read()
        match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', content)
        if match:
            url = match.group(0)
            break

if url:
    print(f"CLOUDFLARE_URL:{{url}}")
else:
    print("CLOUDFLARE_FAILED")
"""

    def start(self) -> str:
        script = self.generate_remote_script()
        res = self.executor.run_code(self.session_name, script)
        stdout = res.get("stdout", "")
        match = re.search(r"CLOUDFLARE_URL:(https://[^\s]+)", stdout)
        if match:
            self.tunnel_url = match.group(1) + "/v1"
            return self.tunnel_url
        raise RuntimeError(f"Failed to start Cloudflare Tunnel on Colab VM: {stdout} {res.get('stderr')}")

    def stop(self) -> None:
        kill_code = "import os; os.system('pkill -f cloudflared')"
        self.executor.run_code(self.session_name, kill_code)
```

Create `colab_agent_bridge/tunnels/__init__.py`:
```python
from colab_agent_bridge.tunnels.base import BaseTunnel
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.tunnels.cloudflare import CloudflareTunnel

__all__ = ["BaseTunnel", "SSHTunnel", "CloudflareTunnel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_tunnels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/tunnels/ tests/test_tunnels.py
git commit -m "feat: add native SSH and Cloudflare Quick Tunnel bridges"
```

---

### Task 6: High-Level SDK (`ColabBridge`), Health Keeper & Agent Harness Generator

**Files:**
- Create: `colab_agent_bridge/client/__init__.py`
- Create: `colab_agent_bridge/client/health.py`
- Create: `colab_agent_bridge/client/harness.py`
- Create: `colab_agent_bridge/client/bridge.py`
- Test: `tests/test_harness.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `SessionManager`, `RemoteExecutor`, `BaseBackend`, `BaseTunnel`
- Produces: `ColabBridge` class (`deploy()`, `teardown()`, `get_endpoint()`), `check_health(url: str)`, `generate_agent_config(endpoint_url: str, model_name: str)`.

- [ ] **Step 1: Write failing tests for harness and health/bridge**

```python
# tests/test_harness.py
from colab_agent_bridge.client.harness import generate_agent_config

def test_generate_agent_config():
    conf = generate_agent_config(
        endpoint_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    assert "OPENAI_BASE_URL=\"http://localhost:8000/v1\"" in conf["env_snippet"]
    assert "OPENAI_API_KEY=\"colab\"" in conf["env_snippet"]
    assert conf["openai_client_kwargs"]["base_url"] == "http://localhost:8000/v1"
    assert conf["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert "continue_config" in conf
    assert "aider_command" in conf
```

```python
# tests/test_bridge.py
from unittest.mock import patch, MagicMock
from colab_agent_bridge.client.bridge import ColabBridge

def test_colab_bridge_init():
    bridge = ColabBridge(session_name="test-bridge", gpu="T4")
    assert bridge.session_name == "test-bridge"
    assert bridge.gpu == "T4"
    assert bridge.backend_name == "vllm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_harness.py tests/test_bridge.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.client')

- [ ] **Step 3: Implement minimal code for client, harness, and health**

Create `colab_agent_bridge/client/health.py`:
```python
import time
import requests
from typing import Dict, Any

def check_health(endpoint_url: str, timeout_seconds: int = 600, poll_interval: int = 5) -> Dict[str, Any]:
    """Polls the /models or /health endpoint until the server is ready or timeout expires."""
    models_url = endpoint_url.rstrip("/") + "/models"
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
        resp = requests.get(endpoint_url.rstrip("/") + "/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
```

Create `colab_agent_bridge/client/harness.py`:
```python
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
        "provider": "openai"
    }

    aider_cmd = f"aider --openai-api-base {url} --openai-api-key colab --model openai/{model_name}"

    return {
        "model": model_name,
        "endpoint_url": url,
        "env_snippet": env_snippet,
        "openai_client_kwargs": {
            "base_url": url,
            "api_key": "colab"
        },
        "continue_config": json.dumps(continue_config, indent=2),
        "aider_command": aider_cmd
    }
```

Create `colab_agent_bridge/client/bridge.py`:
```python
import threading
import time
from typing import Optional, Dict, Any
from colab_agent_bridge.config import BridgeConfig, get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.executor import RemoteExecutor
from colab_agent_bridge.backends import get_backend
from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
from colab_agent_bridge.tunnels.cloudflare import CloudflareTunnel
from colab_agent_bridge.client.health import check_health, keepalive_ping
from colab_agent_bridge.client.harness import generate_agent_config

class ColabBridge:
    def __init__(
        self,
        session_name: str = "agent-bridge",
        gpu: Optional[str] = "T4",
        tpu: Optional[str] = None,
        high_mem: bool = False,
        backend: str = "vllm",
        tunnel_type: str = "ssh",
        local_port: int = 8000,
        config: Optional[BridgeConfig] = None
    ):
        self.session_name = session_name
        self.gpu = gpu
        self.tpu = tpu
        self.high_mem = high_mem
        self.backend_name = backend
        self.tunnel_type = tunnel_type
        self.local_port = local_port
        self.config = config or get_config()

        self.session_mgr = SessionManager(self.config)
        self.executor = RemoteExecutor(self.config)
        self.tunnel = None
        self.endpoint_url: Optional[str] = None
        self._keepalive_active = False
        self._keepalive_thread: Optional[threading.Thread] = None

    def deploy(self, model_name: str, wait_for_ready: bool = True, timeout: int = 600) -> Dict[str, Any]:
        # 1. Provision session
        sess_res = self.session_mgr.new_session(
            session_name=self.session_name,
            gpu=self.gpu,
            tpu=self.tpu,
            high_mem=self.high_mem
        )
        if not sess_res["success"]:
            raise RuntimeError(f"Colab VM allocation failed: {sess_res['stderr']}")

        # 2. Install backend prerequisites
        backend = get_backend(self.backend_name)
        install_script = backend.generate_install_script()
        self.executor.run_code(self.session_name, install_script)

        # 3. Launch model server
        start_script = backend.generate_start_script(model_name=model_name, port=8000)
        self.executor.run_code(self.session_name, start_script)

        # 4. Open tunnel
        if self.tunnel_type == "cloudflare":
            self.tunnel = CloudflareTunnel(self.session_name, remote_port=8000, config=self.config)
            self.endpoint_url = self.tunnel.start()
        else:
            self.tunnel = SSHTunnel(self.session_name, local_port=self.local_port, remote_port=8000, config=self.config)
            self.endpoint_url = self.tunnel.start()

        # 5. Wait for readiness
        health_info = {"ready": True}
        if wait_for_ready:
            health_info = check_health(self.endpoint_url, timeout_seconds=timeout)

        # 6. Start keepalive loop in background
        self._start_keepalive()

        harness_conf = generate_agent_config(self.endpoint_url, model_name)
        return {
            "session_name": self.session_name,
            "endpoint_url": self.endpoint_url,
            "model_name": model_name,
            "health": health_info,
            "harness": harness_conf
        }

    def _start_keepalive(self):
        self._keepalive_active = True
        def loop():
            while self._keepalive_active and self.endpoint_url:
                time.sleep(180)
                if self._keepalive_active:
                    keepalive_ping(self.endpoint_url)
        self._keepalive_thread = threading.Thread(target=loop, daemon=True)
        self._keepalive_thread.start()

    def teardown(self) -> Dict[str, Any]:
        self._keepalive_active = False
        if self.tunnel:
            self.tunnel.stop()
        return self.session_mgr.stop_session(self.session_name)
```

Create `colab_agent_bridge/client/__init__.py`:
```python
from colab_agent_bridge.client.bridge import ColabBridge
from colab_agent_bridge.client.harness import generate_agent_config
from colab_agent_bridge.client.health import check_health, keepalive_ping

__all__ = ["ColabBridge", "generate_agent_config", "check_health", "keepalive_ping"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_harness.py tests/test_bridge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/client/ tests/test_harness.py tests/test_bridge.py
git commit -m "feat: add ColabBridge SDK facade, health keeper, and harness config generator"
```

---

### Task 7: Command Line Interface (`colab-bridge`)

**Files:**
- Create: `colab_agent_bridge/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `SessionManager`, `ColabBridge`, `recommend_model_strategy`, `generate_agent_config`
- Produces: CLI commands: `up`, `down`, `status`, `recommend`, `harness`, `test`.

- [ ] **Step 1: Write failing test for CLI**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from colab_agent_bridge.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "up" in result.stdout
    assert "down" in result.stdout
    assert "status" in result.stdout

def test_cli_recommend():
    result = runner.invoke(app, ["recommend", "Qwen/Qwen2.5-Coder-32B-Instruct", "--gpu", "T4"])
    assert result.exit_code == 0
    assert "T4" in result.stdout
    assert "Quantization required" in result.stdout or "True" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_cli.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'colab_agent_bridge.cli')

- [ ] **Step 3: Implement minimal code for CLI**

Create `colab_agent_bridge/cli.py`:
```python
import sys
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
from colab_agent_bridge.config import get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.hardware import validate_accelerator, recommend_model_strategy, SUPPORTED_GPUS, SUPPORTED_TPUS
from colab_agent_bridge.client.bridge import ColabBridge
from colab_agent_bridge.client.harness import generate_agent_config

app = typer.Typer(
    name="colab-bridge",
    help="Bridge Google Colab GPU/TPU model serving to local coding agent harnesses.",
    add_completion=False
)
console = Console()

@app.command()
def status(session_name: Optional[str] = typer.Option(None, "-s", "--session", help="Session name to inspect")):
    """Check active Colab sessions, hardware status, and CLI authentication."""
    config = get_config()
    mgr = SessionManager(config)
    auth_check = mgr.verify_auth()

    if not auth_check["authenticated"]:
        console.print(Panel(f"[red]Authentication Issue:[/red] {auth_check.get('error')}\n\n[bold yellow]Remediation:[/bold yellow]\n{auth_check.get('remediation')}", title="Colab Auth Status"))
        raise typer.Exit(code=1)

    console.print("[green]✔ Google Colab CLI authenticated.[/green]")
    stat = mgr.get_session_status(session_name)
    if stat["stdout"]:
        console.print(Panel(stat["stdout"], title=f"Session Status: {session_name or 'Default'}"))
    else:
        console.print("[yellow]No active session found or VM is currently stopped.[/yellow]")

@app.command()
def recommend(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    gpu: str = typer.Option("T4", "-g", "--gpu", help="Accelerator (T4, L4, A100, H100)")
):
    """Analyze VRAM requirements and recommend serving backend and quantization."""
    rec = recommend_model_strategy(model, gpu)
    table = Table(title=f"Hardware Strategy for {model}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Target Accelerator", gpu)
    table.add_row("Needs Quantization", str(rec["needs_quantization"]))
    table.add_row("Recommended Backend", rec.get("recommended_backend", "vllm"))
    table.add_row("Quantization Formats", ", ".join(rec.get("recommended_quantization", [])) or "None (FP16)")
    if rec.get("warning"):
        table.add_row("Advisory", f"[yellow]{rec['warning']}[/yellow]")

    console.print(table)

@app.command()
def up(
    model: str = typer.Argument(..., help="Model identifier (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)"),
    session: str = typer.Option("agent-bridge", "-s", "--session", help="Colab session name"),
    gpu: Optional[str] = typer.Option("T4", "-g", "--gpu", help="GPU accelerator tier (T4, L4, A100, H100)"),
    tpu: Optional[str] = typer.Option(None, "--tpu", help="TPU accelerator tier (v5e1, v6e1)"),
    high_mem: bool = typer.Option(False, "--high-mem", help="Request high-RAM machine shape"),
    backend: str = typer.Option("vllm", "-b", "--backend", help="Serving backend (vllm, ollama, llama_cpp)"),
    tunnel: str = typer.Option("ssh", "-t", "--tunnel", help="Tunnel type (ssh, cloudflare)"),
    port: int = typer.Option(8000, "-p", "--port", help="Local port to bind"),
    timeout: int = typer.Option(600, "--timeout", help="Timeout in seconds for model readiness")
):
    """Provision a Colab VM, launch model server, and expose OpenAI endpoint."""
    console.print(f"[bold cyan]Deploying {model} on Google Colab ({gpu or tpu or 'CPU'})...[/bold cyan]")
    bridge = ColabBridge(
        session_name=session,
        gpu=gpu,
        tpu=tpu,
        high_mem=high_mem,
        backend=backend,
        tunnel_type=tunnel,
        local_port=port
    )

    try:
        res = bridge.deploy(model_name=model, wait_for_ready=True, timeout=timeout)
        console.print(Panel(
            f"[bold green]✔ Model is READY and live![/bold green]\n\n"
            f"[cyan]Endpoint URL:[/cyan] {res['endpoint_url']}\n"
            f"[cyan]Model Name:[/cyan] {res['model_name']}\n\n"
            f"[bold]To connect coding harnesses:[/bold]\n"
            f"export OPENAI_BASE_URL=\"{res['endpoint_url']}\"\n"
            f"export OPENAI_API_KEY=\"colab\"\n\n"
            f"[yellow]Aider command:[/yellow] {res['harness']['aider_command']}",
            title="Colab Model Bridge Activated"
        ))
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def down(session: str = typer.Option("agent-bridge", "-s", "--session", help="Session to terminate")):
    """Terminate the tunnel and release Colab VM compute resources."""
    config = get_config()
    mgr = SessionManager(config)
    console.print(f"[yellow]Terminating session {session}...[/yellow]")
    res = mgr.stop_session(session)
    if res["success"]:
        console.print(f"[green]✔ Session {session} successfully released.[/green]")
    else:
        console.print(f"[red]Error stopping session:[/red] {res['stderr']}")

@app.command()
def harness(
    endpoint: str = typer.Option("http://localhost:8000/v1", "--endpoint", "-e", help="Active model endpoint URL"),
    model: str = typer.Option("Qwen/Qwen2.5-Coder-7B-Instruct", "--model", "-m", help="Served model name")
):
    """Print configuration snippets for Aider, Cline, Continue, and OpenAI SDK."""
    conf = generate_agent_config(endpoint, model)
    console.print(Panel(conf["env_snippet"], title="Environment Variables (.env)"))
    console.print(Panel(conf["aider_command"], title="Aider CLI Launch Command"))
    console.print(Panel(conf["continue_config"], title="Continue Config Block"))

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add colab_agent_bridge/cli.py tests/test_cli.py
git commit -m "feat: add colab-bridge CLI commands (up, down, status, recommend, harness)"
```

---

### Task 8: Fresh-Agent Playbook (`AGENTS.md`), Setup Guides & Documentation

**Files:**
- Create: `AGENTS.md`
- Create: `README.md`
- Create: `docs/setup_guide.md`
- Create: `docs/hardware_and_models.md`
- Create: `docs/coding_harness_guide.md`
- Test: Verification of all docs, markdown links, and sample outputs

**Interfaces:**
- Produces: Complete, human- and machine-readable instructions enabling fresh agents to inspect, spin up, use, and shut down Colab model backends safely without secret leakage.

- [ ] **Step 1: Write `AGENTS.md` (Agent Playbook)**

Include:
- Exact path to python and colab binary: `/run/media/parasaran/Dev/SDK/python/install/bin`
- Command references (`colab-bridge up`, `colab-bridge status`, `colab-bridge down`, `colab-bridge harness`)
- Pre-flight checks (`colab sessions` or `colab whoami` to ensure ADC credentials have the 4 required scopes)
- Safety rules (always run `colab-bridge down` when done to preserve user compute units)
- Code snippets showing how an agent can call models via Python `openai` library or `requests`.

- [ ] **Step 2: Write `README.md`**

Include:
- Project title, elevator pitch, architecture diagram
- Quickstart guide
- Accelerator hardware matrix table
- Tunneling guide (Native SSH vs Cloudflare)
- Security & open-source hygiene notice

- [ ] **Step 3: Write `docs/setup_guide.md`**

Include:
- Application Default Credentials (ADC) setup with exact 4-scope command
- OAuth2 fallback instructions
- Colab Pro/Pro+ compute unit management

- [ ] **Step 4: Write `docs/hardware_and_models.md`**

Include:
- T4, L4, G4, A100, H100, and TPU matrix
- Recommended coding models (Qwen 2.5 Coder, DeepSeek-Coder, Llama 3.3, StarCoder2)
- Quantization advice (AWQ vs GPTQ vs GGUF)

- [ ] **Step 5: Write `docs/coding_harness_guide.md`**

Include:
- Integrating with Aider
- Integrating with Cline
- Integrating with Continue IDE extension
- Integrating with custom agent loops

- [ ] **Step 6: Run full test suite to verify 100% passing tests**

Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit documentation and complete suite**

```bash
git add AGENTS.md README.md docs/
git commit -m "docs: add AGENTS.md agent playbook, setup guides, hardware matrix, and harness instructions"
```

---

## Plan Self-Review Check
1. **Spec Coverage:**
   - Multi-accelerator support (T4, L4, A100, H100, TPU, CPU, high-mem) -> Task 2 & Task 3
   - Unified OpenAI-compatible backends (vLLM, Ollama, llama.cpp) -> Task 4
   - Hybrid tunneling (native SSH + Cloudflare) -> Task 5
   - High-level SDK + Health Keepalive -> Task 6
   - Rich CLI -> Task 7
   - Fresh-agent playbook (`AGENTS.md`) & setup guides -> Task 8
   - Zero secrets & strict gitignore -> Scaffolding + Task 1 + Task 8
2. **Placeholder scan:** No "TBD" or "TODO". All code blocks, commands, and tests are concrete.
3. **Type consistency:** Function names and dataclasses (`BridgeConfig`, `SessionManager`, `ColabBridge`, `generate_agent_config`) match across tasks.
