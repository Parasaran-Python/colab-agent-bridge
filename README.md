# colab-agent-bridge

> Bridge Google Colab GPU/TPU model serving directly to local coding agent harnesses (Aider, Cline, Continue, Cursor) via OpenAI-compatible endpoints.

[![CI](https://github.com/parasaran/colab-agent-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/parasaran/colab-agent-bridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`colab-agent-bridge` turns your Google Colab compute units into a private, high-throughput, OpenAI-compatible LLM inference server. It automatically orchestrates VM provisioning via `google-colab-cli`, installs and starts modern model engines (vLLM, Ollama, or llama.cpp), establishes encrypted tunnels (Native SSH or Cloudflare Quick Tunnels), and exports ready-to-run configurations for your favorite coding tools.

---

## 🌟 Key Features

- **Multi-Accelerator Flexibility:** Run on any Colab hardware shape: Free & Pro NVIDIA T4 (16GB), L4 (24GB), A100 (40/80GB), H100 (80GB), Google Cloud TPU v5e/v6e, or High-Memory CPU.
- **Multi-Account Clustering & Pooling:** Bypass Google Colab's 1-GPU limit per account. Manage multiple Google accounts with isolated credential profiles (`colab-bridge profile`), deploy concurrent GPUs across accounts, and monitor active compute from a unified dashboard.
- **Unified OpenAI-Compatible Endpoints:** Deploy models via **vLLM**, **Ollama**, or **llama.cpp** exposing standard `/v1/models` and `/v1/chat/completions` endpoints.
- **Hybrid Tunneling:**
  - **Native SSH Tunnel:** Zero external accounts, direct local port forward (`localhost:8000`) over encrypted Colab SSH.
  - **Cloudflare Quick Tunnel:** Instant public HTTPS endpoints (`https://*.trycloudflare.com/v1`) without firewall or port-forwarding constraints.
- **Coding Harness Integrations:** Automatic configuration generation for **Aider**, **Cline**, **Continue**, and the **OpenAI Python SDK**.
- **Automated VRAM Advisor:** Built-in hardware strategy engine warns when models exceed accelerator memory and recommends optimal quantization (AWQ, GPTQ, GGUF).
- **Global Teardown & Quota Protection:** Instant nuclear teardown (`colab-bridge down --all`) stops sessions across all accounts and terminates all proxy adapters to prevent compute unit leaks.
- **Background Keepalive:** Integrated health ping prevents Colab idle timeouts while you code.
- **Zero-Secret Security:** Strict credential hygiene, scoped Application Default Credentials (ADC), isolated per-profile token storage, and zero state leakage.
- **Machine-Actionable Agent Playbook:** Includes [AGENTS.md](AGENTS.md) so autonomous AI agents can operate the bridge without human intervention.

---

## 🏗️ Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    LOCAL ENVIRONMENT                                    |
|                                                                                         |
|  +--------------------+        +---------------------+        +----------------------+  |
|  |   Coding Harness   |        |    CLI Controller   |        |      Python SDK      |  |
|  | (Aider/Cline/Cont) |        |   (`colab-bridge`)  |        |    (`ColabBridge`)   |  |
|  +---------+----------+        +----------+----------+        +-----------+----------+  |
|            |                              |                               |             |
|            | (HTTP /v1 requests)          +---------------+---------------+             |
|            |                                              |                             |
|            v                                              v                             |
|  +--------------------+                      +---------------------------------------+  |
|  | Local Port Forward |                      |          colab_agent_bridge           |  |
|  | (localhost:8000)   |<-----------+         |  - Core (Session, Hardware, Executor) |  |
|  +---------+----------+            |         |  - Backends (vLLM, Ollama, llama.cpp) |  |
|            |                       |         |  - Tunnels (Native SSH / Cloudflare)  |  |
|            | (Encrypted SSH / WS)  |         |  - Client & Keepalive Daemon          |  |
|            |                       |         +-------------------+-------------------+  |
+------------|-----------------------|-----------------------------|----------------------+
             |                       |                             | (colab CLI commands)
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

---

## 🚀 Quickstart

### 1. Prerequisites & Authentication

Ensure Google Cloud Application Default Credentials (ADC) are minted with the 4 required scopes:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Verify authentication:
```bash
colab-bridge status
```

*(See [docs/setup_guide.md](docs/setup_guide.md) for full OAuth2 and ADC configuration details).*

### 2. Check Hardware Fit

Analyze whether your preferred model fits into your Colab accelerator tier:

```bash
colab-bridge recommend Qwen/Qwen2.5-Coder-7B-Instruct --gpu T4
```

### 3. Deploy Model Server

Provision the VM, install vLLM, and start the SSH tunnel:

```bash
colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct --gpu T4
```

Output:
```text
✔ Model is READY and live!
Endpoint URL: http://localhost:8000/v1
Model Name: Qwen/Qwen2.5-Coder-7B-Instruct

To connect coding harnesses:
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="colab"

Aider command: aider --openai-api-base http://localhost:8000/v1 --openai-api-key colab --model openai/Qwen/Qwen2.5-Coder-7B-Instruct
```

### 4. Code with Your Agent

Launch Aider or configure your IDE extension:
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key colab --model openai/Qwen/Qwen2.5-Coder-7B-Instruct
```

### 5. Tear Down & Release Resources

> [!IMPORTANT]
> **Always shut down your session when you are done to conserve Compute Units:**
> ```bash
> colab-bridge down -s agent-bridge
> ```
> Or terminate **all** running instances across all Google accounts in one command:
> ```bash
> colab-bridge down --all
> ```

### 6. Multi-Account GPU Sharding / Clustering

Need multiple concurrent GPUs (e.g., serving multiple models simultaneously or sharding workloads)? Add secondary Google accounts:

```bash
# Add a second account profile
colab-bridge profile add personal_2

# Check status and compute balances across all accounts
colab-bridge status --all
colab-bridge profile list

# Deploy a model on the second account (binds to another local port)
colab-bridge up deepseek-ai/deepseek-coder-6.7b-instruct --profile personal_2 --port 8002

# Nuclear teardown when finished
colab-bridge down --all
```

---

## 💻 Accelerator Hardware Matrix

| Accelerator | Type | Memory (VRAM) | Colab Tier | Recommended Serving Capacity |
|---|---|---|---|---|
| **T4** | GPU | 16 GB | Free / Pro | 7B-8B Quantized (AWQ/GPTQ), 3B-7B GGUF |
| **L4** | GPU | 24 GB | Pro / Pro+ | 7B-8B FP16, 14B Quantized (AWQ) |
| **G4** | GPU | 16 GB | Pro | 7B-8B Quantized |
| **A100** | GPU | 40 GB / 80 GB | Pro+ | 14B-32B FP16, 70B Quantized (AWQ) |
| **H100** | GPU | 80 GB | Pro+ Premium | 70B FP16 / BF16, High-throughput vLLM batches |
| **v5e1** | TPU | 16 GB | Pro / Pro+ | Specialized TPU PyTorch / JAX models |
| **v6e1** | TPU | 32 GB | Pro+ | Next-gen Trillium TPU architectures |
| **CPU** | CPU | 12 GB / 52 GB | Free / Pro | GGUF models via llama.cpp (with `--high-mem`) |

For comprehensive model guidance and quantization rules, see [docs/hardware_and_models.md](docs/hardware_and_models.md).

---

## 🌐 Tunneling Modes

`colab-agent-bridge` offers two tunneling architectures:

| Feature | Native SSH (`--tunnel ssh`) | Cloudflare Quick Tunnel (`--tunnel cloudflare`) |
|---|---|---|
| **Local Endpoint** | `http://localhost:8000/v1` | `https://<unique-subdomain>.trycloudflare.com/v1` |
| **External Accounts** | None required (uses Colab credentials) | None required (uses Cloudflare Quick Tunnels) |
| **Network Reach** | Local machine only | Any machine / container / cloud webhook |
| **Setup Time** | ~2 seconds | ~5-10 seconds |
| **Best For** | Local CLI tools (Aider, Continue, Cline) | Remote development containers, webhooks, or peers |

---

## 🛠️ CLI Command Reference

### Profile Management (`colab-bridge profile`)

Manage multiple isolated Google Colab accounts for concurrent GPU allocations:

- **`colab-bridge profile list`**: Renders a Rich table displaying configured profiles, authenticated Google accounts, real-time CU balance, hourly usage rates, active sessions, and compute burn warnings.
- **`colab-bridge profile add <name>`**: Scaffolds isolated profile directory (`~/.config/colab-bridge/profiles/<name>/`), symlinks host SSH keys, and triggers browser OAuth flow.
- **`colab-bridge profile switch <name>`**: Switches the default active account profile.
- **`colab-bridge profile remove <name>`**: Removes profile credentials and directory.

```bash
colab-bridge profile list
colab-bridge profile add personal_2
colab-bridge profile switch personal_2
colab-bridge profile remove personal_2
```

### `colab-bridge status`
Inspects active Colab sessions, hardware status, and compute units across accounts.
```bash
# Check current profile status
colab-bridge status [-s <session_name>] [-p <profile_name>]

# Check unified dashboard across all registered Google accounts
colab-bridge status --all
```

### `colab-bridge recommend`
Analyzes VRAM requirements against target hardware and recommends backend and quantization formats.
```bash
colab-bridge recommend <model_name> [-g <gpu_tier>]
```

### `colab-bridge up`
Provisions VM, installs backend, starts server, connects tunnel, and waits for `/v1/models` readiness.
```bash
colab-bridge up <model_name> \
  [-s <session_name>] \
  [-p <profile_name>] \
  [-g <gpu_tier>] \
  [--tpu <tpu_tier>] \
  [--high-mem] \
  [-b <vllm|ollama|llama_cpp>] \
  [-t <ssh|cloudflare>] \
  [--port <local_port>] \
  [--timeout <seconds>]
```

### `colab-bridge down`
Stops tunnels and terminates Colab VM compute resources. Supports global nuclear teardown.
```bash
# Terminate a single session on current profile
colab-bridge down [-s <session_name>] [-p <profile_name>]

# Global teardown: terminate ALL running VMs across ALL accounts
colab-bridge down --all
```

### `colab-bridge harness`
Prints environment variables and ready-to-paste configurations for Aider, Cline, and Continue.
```bash
colab-bridge harness [-e <endpoint_url>] [-m <model_name>]
```

### `colab-bridge test`
Performs an end-to-end HTTP health test against a live model endpoint.
```bash
colab-bridge test [-e <endpoint_url>] [-t <timeout_seconds>]
```

---

## 🐍 Python SDK

You can orchestrate sessions programmatically within your own Python scripts or agents, including targeting specific multi-account profiles:

```python
from colab_agent_bridge.client.bridge import ColabBridge
from openai import OpenAI

# Deploy under default or a specific multi-account profile
bridge = ColabBridge(
    session_name="my-worker",
    gpu="T4",
    backend="vllm",
    profile="personal_2",  # Optional: target a specific profile
    local_port=8002,
)

try:
    info = bridge.deploy("Qwen/Qwen2.5-Coder-7B-Instruct")
    print(f"Endpoint: {info['endpoint_url']}")

    client = OpenAI(base_url=info['endpoint_url'], api_key="colab")
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        messages=[{"role": "user", "content": "Write a quicksort in Python."}]
    )
    print(completion.choices[0].message.content)
finally:
    # Teardown releases the VM and stops billing CUs
    bridge.teardown()
```

---

## 🔒 Security & Open-Source Hygiene

1. **Zero Secret Leaks:**
   - Session states (`~/.config/colab-cli/*`), credentials (`~/.colab-cli-oauth-config.json`), SSH keys, and `.env` files are excluded by `.gitignore`.
   - Never commit sensitive tokens or session identifiers.
2. **Strict Scope Validation:**
   - The CLI strictly verifies the four ADC scopes (`openid`, `cloud-platform`, `userinfo.email`, `colaboratory`) before issuing VM requests.
3. **Safe Subprocess Calls:**
   - All external commands use explicit argument lists without unescaped shell string interpolation.

---

## 📚 Documentation Index

- [Autonomous Agent Operating Playbook (`AGENTS.md`)](AGENTS.md)
- [Authentication & Setup Guide (`docs/setup_guide.md`)](docs/setup_guide.md)
- [Hardware & Model Guide (`docs/hardware_and_models.md`)](docs/hardware_and_models.md)
- [Coding Harness Guide (`docs/coding_harness_guide.md`)](docs/coding_harness_guide.md)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
