# Design Document: Colab Agent Bridge (`colab-agent-bridge`)

**Date:** 2026-10-31
**Status:** Approved
**Project Path:** `/run/media/parasaran/Dev/linux_dev/git/colab-agent-bridge`

---

## 1. Executive Summary

`colab-agent-bridge` is a developer tool and Python SDK designed to orchestrate Google Colab virtual machines via `google-colab-cli`, deploy OpenAI-compatible model inference servers (vLLM, Ollama, llama.cpp), securely tunnel the remote API to the local environment, and seamlessly expose the running models to local coding agent harnesses (such as Aider, Cline, Continue, Cursor, or custom evaluation loops).

The project is built to support all Colab accelerator tiers (T4, L4, G4, A100, H100, TPU v5e1/v6e1, and CPU with `--high-mem`), handle graceful resource fallbacks, safeguard against credential leaks, and provide machine-readable playbooks (`AGENTS.md`) and onboarding documentation so fresh AI agents and developers can independently operate the infrastructure.

---

## 2. Goals and Non-Goals

### 2.1 Goals
- **Multi-Accelerator Support:** Support any Google Colab accelerator shape (T4, L4, G4, A100, H100, TPU v5e1/v6e1, and CPU). Automatically warn or adapt configuration according to available GPU VRAM.
- **Unified OpenAI-Compatible Endpoints:** Deploy models via vLLM, Ollama, or llama.cpp, all exposing standard `/v1/models`, `/v1/chat/completions`, and `/v1/completions` routes on port `8000`.
- **Hybrid Tunneling:**
  - *Native SSH Tunnel:* Zero 3rd-party dependencies, leveraging `colab ssh` with local port forwarding (`localhost:8000`).
  - *Cloudflare Quick Tunnel:* Ephemeral public HTTPS endpoints via `cloudflared` without requiring tokens or accounts.
- **Dual Interfaces:** Rich terminal CLI (`colab-bridge`) for quick human/script management and a clean Python SDK (`ColabBridge`) for programmatic agent pipelines.
- **Coding Harness Integrations:** One-click generation of configuration and `.env` profiles for Aider, Cline, Continue, and custom OpenAI-compatible agent harnesses.
- **Zero-Secret Open Source Hygiene:** Strict `.gitignore` rules, `.env.example` templates, and credential sanity verification before touching Colab APIs.
- **Self-Documenting Agent Guides:** Provide `AGENTS.md` and detailed documentation in `docs/` so fresh coding agents can discover capabilities, configure auth, run models, and properly release billable compute.

### 2.2 Non-Goals
- Replacing `google-colab-cli` (it is a wrapper and orchestration layer around the official CLI).
- Implementing custom inference engines (we leverage battle-tested vLLM, Ollama, and llama-cpp-python).
- Long-term training job orchestrator (focus is on model serving, inference, and coding agent bridging).

---

## 3. Architecture & Data Flow

### 3.1 Architecture Overview

```
+-----------------------------------------------------------------------------------------+
|                                    LOCAL MACHINE                                        |
|                                                                                         |
|  +--------------------+        +---------------------+        +----------------------+  |
|  |   Coding Agent     |        |     Terminal CLI    |        |      Python SDK      |  |
|  | (Aider/Cline/SDK)  |        |    (`colab-bridge`) |        |    (`ColabBridge`)   |  |
|  +---------+----------+        +----------+----------+        +-----------+----------+  |
|            |                              |                               |             |
|            | (HTTP /v1 calls)             +---------------+---------------+             |
|            |                                              |                             |
|            v                                              v                             |
|  +--------------------+                      +---------------------------------------+  |
|  | Local Port Forward |                      |          colab_agent_bridge           |  |
|  | (localhost:8000)   |<-----------+         |  - Core (Session, Hardware, Exec)     |  |
|  +---------+----------+            |         |  - Backends (vLLM, Ollama, llama.cpp) |  |
|            |                       |         |  - Tunnels (Native SSH / Cloudflare)  |  |
|            | (Colab SSH / WS)      |         |  - Client & Health Keeper             |  |
|            |                       |         +-------------------+-------------------+  |
+------------|-----------------------|-----------------------------|----------------------+
             |                       |                             | (colab CLI subprocess)
             |                       |                             v
+------------|-----------------------|----------------------------------------------------+
|            v                       |                                                    |
|  +--------------------+            |                                                    |
|  |    Colab VM SSH    |            |                                                    |
|  |     (Port 8000)    |            |                                                    |
|  +---------+----------+            |                                                    |
|            |                       |                                                    |
|            v                       |                                                    |
|  +--------------------+   (Public HTTPS URL)                                            |
|  | Inference Backend  |            |                                                    |
|  | (vLLM/Ollama/GGUF) |            |                                                    |
|  +---------+----------+            |                                                    |
|            |                       |                                                    |
|            v                       |                                                    |
|  +--------------------+            |                                                    |
|  |  Cloudflare Tunnel |------------+                                                    |
|  |   (Optional)       |                                                                 |
|  +--------------------+                                                                 |
|                                     GOOGLE COLAB VM                                     |
+-----------------------------------------------------------------------------------------+
```

### 3.2 End-to-End Workflow
1. **Validation & Discovery:** Verify Google Colab CLI binary path and validate Application Default Credentials (ADC) scopes (`colaboratory`, `userinfo.email`, `cloud-platform`, `openid`).
2. **Session Provisioning:** Request VM shape (`colab new -s <session_name> [--gpu <tier>] [--tpu <tier>] [--high-mem]`).
3. **Environment Setup & Serving Launch:**
   - Execute backend bootstrap script via `colab exec`.
   - Install dependencies (using high-speed `uv pip` on Colab).
   - Launch model inference server detached in background on port `8000`.
4. **Tunnel Establishment:**
   - *Default (SSH):* Launch `colab ssh --proxy-mode` or background SSH port forward (`-N -L 8000:localhost:8000`).
   - *Optional (Cloudflare):* Run remote `cloudflared` daemon on Colab VM, capture generated URL, and emit it.
5. **Readiness Probe:** Local client polls `/v1/models` until the model is loaded and reporting 200 OK.
6. **Agent Engagement:** Client exports `.env` or configurations for Aider, Cline, Continue, Cursor, or agent loops.
7. **Keepalive Daemon:** Runs an unobtrusive background ping every 3-5 minutes to prevent idle kernel timeout.
8. **Teardown:** `colab-bridge down` kills local tunnels and calls `colab stop -s <session_name>` to immediately stop billing compute units.

---

## 4. Component Specifications

### 4.1 Core Subsystem (`colab_agent_bridge/core`)
- `session.py`: Manages lifecycle (`new`, `sessions`, `status`, `stop`, `restart-kernel`, `url`). Formulates safe subprocess calls to the specified `colab` CLI binary.
- `executor.py`: Handles code execution (`exec`, `upload`, `download`, `install`). Manages file staging and script uploads.
- `hardware.py`:
  - Maps accelerator types:
    - GPU: `T4` (16GB), `L4` (24GB), `G4`, `A100` (40/80GB), `H100` (80GB).
    - TPU: `v5e1`, `v6e1`.
    - CPU with `--high-mem` option.
  - VRAM-aware model heuristics: Checks model parameters vs. target GPU VRAM and warns or suggests quantization (AWQ, GPTQ, GGUF).

### 4.2 Backend Subsystem (`colab_agent_bridge/backends`)
- `base.py`: Abstract Base Class `BaseBackend` defining:
  - `generate_install_script()` -> Shell/Python code to install prerequisites on Colab.
  - `generate_start_script(model, port, **kwargs)` -> Execution script to run server on Colab.
  - `health_endpoint` -> Standard `/v1/models`.
- `vllm.py`: vLLM engine bootstrap. Supports `--max-model-len`, `--gpu-memory-utilization`, `--quantization` (`awq`, `gptq`, `squeezellm`), and tensor parallelism if multiple GPUs exist.
- `ollama.py`: Downloads latest Linux AMD64 Ollama binary, starts `ollama serve`, and pulls specified model (e.g., `qwen2.5-coder:7b`, `deepseek-coder:6.7b`).
- `llama_cpp.py`: Configures `llama-cpp-python` with CUDA acceleration, downloads GGUF models from Hugging Face, and exposes the OpenAI-compatible HTTP server.

### 4.3 Tunnel Subsystem (`colab_agent_bridge/tunnels`)
- `base.py`: Abstract Base Class `BaseTunnel` managing tunnel process lifecycles.
- `ssh_tunnel.py`: Executes OpenSSH with Colab's proxy mode command to map remote port `8000` to local `8000`. Monitors process health and restarts on connection drops.
- `cloudflare.py`: Provisions remote `cloudflared` binary on the VM, runs tunnel pointed at `http://localhost:8000`, extracts `https://*.trycloudflare.com`, and yields the URL.

### 4.4 Client & Harness Subsystem (`colab_agent_bridge/client`)
- `bridge.py`: High-level Python facade `ColabBridge` combining session, backend, tunnel, and health checks.
- `health.py`: HTTP health checking against `/v1/models` with exponential backoff and timeout thresholds.
- `harness.py`: Helper utilities to export environment variables and configuration files:
  - `export_aider_env()`: Generates `OPENAI_API_BASE` and `OPENAI_API_KEY`.
  - `export_cline_config()`: Generates Cline extension JSON configuration.
  - `export_continue_config()`: Generates Continue `config.json` model block.

### 4.5 Command Line Interface (`colab_agent_bridge/cli.py`)
CLI powered by Click/Typer:
- `colab-bridge up`: Deploy a session, install backend, start model, and establish tunnel.
- `colab-bridge down`: Terminate tunnel and release Colab VM.
- `colab-bridge status`: Display active sessions, hardware status, and tunnel health.
- `colab-bridge tunnel`: Open a standalone SSH or Cloudflare tunnel to an existing session.
- `colab-bridge test`: Send a test completion prompt to verify end-to-end functionality.
- `colab-bridge harness`: Print or export agent config snippets.

---

## 5. Security, Secrets & Open-Source Hygiene

1. **Zero Secret Leaks:**
   - All session JSON files (`~/.config/colab-cli/*`), OAuth credentials (`~/.colab-cli-oauth-config.json`), SSH keys, `.env` files, and local logs are strictly ignored in `.gitignore`.
   - `.env.example` provides sanitized template values only.
2. **ADC Scope Verification:**
   - Detects if `gcloud auth application-default login` lacks the 4 required scopes (`openid`, `https://www.googleapis.com/auth/cloud-platform`, `https://www.googleapis.com/auth/userinfo.email`, `https://www.googleapis.com/auth/colaboratory`) and provides the exact remediation command.
3. **Safe Subprocess Execution:**
   - All subprocess calls invoke commands using explicit argument lists (no unsafe raw shell string interpolation).

---

## 6. Testing Strategy

1. **Unit Testing:**
   - `test_hardware.py`: Verify GPU/TPU matrix mapping, VRAM validation, and shape flag construction.
   - `test_session.py`: Verify CLI command formulation (e.g., arguments passed to `colab new`, `colab exec`, `colab stop`).
   - `test_backends.py`: Verify script generation for vLLM, Ollama, and llama.cpp backends.
   - `test_harness.py`: Verify configuration export formatting for Aider, Cline, and OpenAI SDK.
2. **Integration / Mock Testing:**
   - Mocked subprocess executions simulating Colab CLI responses (session lists, execution outputs, and status).

---

## 7. Documentation & Agent Enablement

1. `README.md`: User quickstart, CLI options, architectural diagrams, and troubleshooting.
2. `AGENTS.md`: Machine-readable, step-by-step instructions specifically formatted for fresh AI agents:
   - Environment variables (`COLAB_CLI_PATH`).
   - Standard operation procedures for provisioning, querying, and deallocating.
   - Recovery steps when sessions disconnect or quotas are exhausted.
3. `docs/setup_guide.md`: Detailed authentication instructions for Google Cloud ADC & OAuth2.
4. `docs/hardware_and_models.md`: GPU tier guide (T4, L4, A100, H100) with recommended model sizes and quantization formats.
5. `docs/coding_harness_guide.md`: Configuration guides for Aider, Continue, Cursor, and custom Python agents.
