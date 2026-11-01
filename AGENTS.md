# AGENTS.md — Autonomous Agent Operating Playbook

> **Target Audience:** Autonomous coding agents (Claude, Antigravity, Cursor, Cline, Aider, custom agent loops) executing inside this environment without human intervention.

This playbook gives you complete operational instructions for provisioning, querying, and releasing Google Colab GPU/TPU compute resources to serve open-weight Large Language Models (LLMs) via OpenAI-compatible endpoints.

---

## 1. Quick Environment Reference

Always use the explicit binary paths configured in this environment:

| Binary / Tool | Absolute Filesystem Path | Notes |
|---|---|---|
| **Colab Bridge CLI** | `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge` | Main orchestration CLI |
| **Colab CLI** | `/run/media/parasaran/Dev/SDK/python/install/bin/colab` | Underlying Google Colab CLI |
| **Python Interpreter** | `/run/media/parasaran/Dev/SDK/python/install/bin/python3` | System Python 3.14 SDK |
| **Pytest** | `/run/media/parasaran/Dev/SDK/python/install/bin/pytest` | Test suite runner |

Environment configuration is read from `.env` in the repository root or system environment variables:
- `COLAB_CLI_PATH`: Defaults to `/run/media/parasaran/Dev/SDK/python/install/bin/colab`.
- `COLAB_BRIDGE_PORT`: Defaults to `8000`.
- `COLAB_DEFAULT_BACKEND`: `vllm` (options: `vllm`, `ollama`, `llama_cpp`).
- `COLAB_DEFAULT_GPU`: `T4` (options: `T4`, `L4`, `A100`, `H100`, `v5e1`, `v6e1`, `CPU`).
- `COLAB_AUTH_STRATEGY`: `adc` (options: `adc`, `oauth2`).

---

## 2. Cardinal Rules

### Rule 2.1: Compute Unit Safety (Mandatory Teardown)

> [!CAUTION]
> **COMPUTE UNIT PRESERVATION RULE**
> Google Colab VMs consume billable Compute Units (CUs) for every minute they remain active. Leaving an idle VM active drains the user's quota.
>
> **Every autonomous workflow MUST execute a teardown step upon completion, failure, or cancellation:**
> ```bash
> /run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge down -s <session_name>
> ```
> Or for distributed sharded clusters:
> ```bash
> /run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster down --session <session_name>
> ```
> Or when managing multi-account / cluster sessions, perform a global teardown:
> ```bash
> /run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge down --all
> # or
> /run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster down --all
> ```
> In Python code, ALWAYS wrap bridge operations in a `try...finally` block that invokes `bridge.teardown()`.

### Rule 2.2: Mandatory Documentation Synchronization

> [!IMPORTANT]
> **DOCUMENTATION SYNCHRONIZATION REQUIREMENT**
> Whenever any code, feature, CLI command, parameter/flag, configuration setting, SDK method, or architectural behavior is added, modified, or removed, you MUST update all corresponding documentation files in the same changeset before claiming completion:
> - `README.md`: Features, Quickstart, CLI Reference, Python SDK examples.
> - `AGENTS.md`: Autonomous agent playbook, SOPs, commands, diagnostics, recovery table.
> - `docs/`: `setup_guide.md`, `hardware_and_models.md`, `coding_harness_guide.md`.
>
> **Never consider any feature, fix, refactor, or PR complete until documentation is 100% synchronized and verified.**

---

## 3. Pre-Flight Authentication Checks

Before attempting to spin up any Colab instance, verify your credentials.

### Step 3.1: Execute Auth Verification

Run:
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge status
```
Or view all configured account profiles, balances, and active VMs:
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge status --all
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge profile list
```

- **Exit code `0` (Success):** CLI reports `Google Colab CLI authenticated.` Proceed to Step 4.
- **Exit code `1` (Failure):** Application Default Credentials (ADC) are missing or lack the required OAuth scopes.

### Step 3.2: Remediate Missing ADC Scopes

If authentication fails on the default account, the exact Google Cloud ADC command required is:
```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

> [!IMPORTANT]
> The 4 scopes above (`openid`, `cloud-platform`, `userinfo.email`, `colaboratory`) are strictly mandatory. Missing `https://www.googleapis.com/auth/colaboratory` will cause authorization rejections when initiating VM sessions.

### Step 3.3: Managing Multi-Account Profiles

When running concurrent GPU sessions across distinct Google accounts:
- Add a profile: `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge profile add <profile_name>`
- Switch active profile: `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge profile switch <profile_name>`
- Remove a profile: `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge profile remove <profile_name>`

---

## 4. Standard Operating Procedure (SOP)

Follow this 5-stage workflow when a user or sub-task requests local open-source LLM inference.

```
+------------------+     +-------------------+     +------------------+
| 1. Recommend &   | --> | 2. Provision &    | --> | 3. Health Verify |
|    VRAM Check    |     |    Deploy (Up)    |     |    Endpoint      |
+------------------+     +-------------------+     +------------------+
                                                             |
+------------------+     +-------------------+               |
| 5. Mandatory     | <-- | 4. Consume via    | <-------------+
|    Teardown      |     |    Harness / SDK  |
+------------------+     +-------------------+
```

### Stage 1: Analyze Hardware & Quantization Fit

Check whether the target model fits in the target accelerator's VRAM:

```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge recommend Qwen/Qwen2.5-Coder-7B-Instruct --gpu T4
```

Example output:
- `Needs Quantization: True` -> Model (~8B params) requires ~20GB in FP16; exceeds T4's 16GB VRAM.
- `Recommended Backend: vllm`
- `Quantization Formats: awq, gptq, gguf`

### Stage 2: Provision VM and Deploy Server

Start the Colab instance, install dependencies, launch the server, and create an SSH tunnel (forwarding remote port 8000 to `http://localhost:8000/v1`):

```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge up \
  Qwen/Qwen2.5-Coder-7B-Instruct \
  --session agent-bridge \
  --gpu T4 \
  --backend vllm \
  --tunnel ssh \
  --port 8000 \
  --timeout 600
```

*For public HTTPS endpoints (e.g., when routing to remote containers or webhooks), use `--tunnel cloudflare`.*

### Stage 3: Verify Health Endpoint

Verify the server is responding to HTTP requests:

```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge test --endpoint http://localhost:8000/v1
```

Or query `/v1/models` directly:
```bash
curl -s http://localhost:8000/v1/models
```

### Stage 4: Generate Configuration and Connect Harnesses

Export environment settings for your coding harness:

```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge harness \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --endpoint http://localhost:8000/v1
```

Set environment variables in your subshell:
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="colab"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
```

### Stage 5: Mandatory Teardown

When your reasoning, editing, or evaluation tasks finish:

```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge down --session agent-bridge
```

---

### Section 4.2: Distributed Cluster Weight Sharding SOP (Multi-Node Cloud-Native)

When operating on large 32B or 70B parameter models that exceed single-GPU VRAM limits, follow this distributed sharding workflow across multiple authenticated Google profiles.

> [!TIP]
> **Cloud-Native Coordinator Architecture (Zero Local Resource Footprint):**
> By default (`--remote-coordinator`), Node 1 serves as the distributed coordinator inside Google Cloud, downloading GGUF weights directly from HuggingFace at 200+ MB/s and streaming secondary layer shards to Node 2 via a direct Cloudflare WebSocket P2P bridge.
> - **Local Disk Usage:** **0 GB** (no model files downloaded to host drive).
> - **Local RAM Usage:** **< 30 MB** (no model tensors loaded in host memory).
> - **Local Upload Bandwidth:** **0 MB** (bypasses home internet upload limits).
> - **Startup Time:** Slashed from ~80 minutes down to **~2–3 minutes total**.

```
+------------------+     +-------------------+     +------------------+
| 1. Profile &     | --> | 2. Deploy Sharded | --> | 3. Verify Health |
|    Memory Check  |     |    Cluster (Up)   |     |    & RPC Tunnels |
+------------------+     +-------------------+     +------------------+
                                                             |
+------------------+     +-------------------+               |
| 5. Mandatory     | <-- | 4. Route Harness  | <-------------+
|    Cluster Down  |     |    Traffic        |
+------------------+     +-------------------+
```

#### Step 1: Check Profile Balances and Memory Fit
Verify participating profiles are authenticated and check pooled VRAM:
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge profile list
```
*Note: 2x T4 GPUs provide 32GB pooled VRAM (fits Qwen 2.5 Coder 32B quantized). 4x T4 GPUs provide 64GB pooled VRAM (fits Llama 3.3 70B quantized).*

#### Step 2: Deploy Distributed Cluster
Launch worker RPC daemons across remote profiles and establish cloud-native coordinator:
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster up \
  Qwen/Qwen2.5-Coder-32B-Instruct \
  --profiles default,sitonam \
  --session cluster-bridge \
  --gpu T4 \
  --port 8000 \
  --timeout 600
```
*(Optionally pass `--local-coordinator` if you wish to run the coordinator binary locally on host).*

#### Step 3: Inspect Cluster Status & RPC Tunnels
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster status
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge test --endpoint http://localhost:8000/v1
```

#### Step 4: Route Coding Agent Traffic
Export standard environment variables:
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="colab"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
```

#### Step 5: Mandatory Cluster Teardown
Release all worker VMs and close tunnels across all participating profiles:
```bash
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster down --session cluster-bridge
# Or global nuclear teardown:
/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge cluster down --all
```

---

## 5. Programmatic Python SDK Integration

When building an autonomous Python loop, use `colab_agent_bridge` directly.

### Pattern: Safe Python Deployment & Completion

```python
import sys
from openai import OpenAI
from colab_agent_bridge.client.bridge import ColabBridge

# 1. Initialize the bridge configuration (optionally target a secondary profile)
bridge = ColabBridge(
    session_name="agent-worker-01",
    gpu="T4",
    backend="vllm",
    tunnel_type="ssh",
    local_port=8000,
    profile=None,  # Or set to specific account e.g. "personal_2"
)

model_identifier = "Qwen/Qwen2.5-Coder-7B-Instruct"

try:
    print("[Agent] Provisioning Colab instance and launching model...")
    # deploy() creates the VM, boots the backend, starts the tunnel,
    # verifies /v1/models health, and starts a background keepalive daemon.
    result = bridge.deploy(model_name=model_identifier, wait_for_ready=True, timeout=600)
    endpoint_url = result["endpoint_url"]  # http://localhost:8000/v1
    print(f"[Agent] Serving active at {endpoint_url}")

    # 2. Query via OpenAI Python Client
    client = OpenAI(
        base_url=endpoint_url,
        api_key="colab",  # Colab bridges accept any non-empty string
    )

    response = client.chat.completions.create(
        model=model_identifier,
        messages=[
            {"role": "system", "content": "You are an expert systems programmer."},
            {"role": "user", "content": "Write a robust Python context manager for temporary directories."},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    print("[Agent Output]:")
    print(response.choices[0].message.content)

finally:
    # 3. CRITICAL: Teardown VM and release compute units
    print("[Agent] Tearing down Colab session to preserve compute units...")
    teardown_status = bridge.teardown()
    print(f"[Agent] Teardown status: {teardown_status}")
```

### Pattern: Direct HTTP Requests (Zero 3rd-party LLM dependencies)

```python
import requests

endpoint = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
        {"role": "user", "content": "Explain git rebase in one sentence."}
    ],
    "temperature": 0.0,
}

headers = {
    "Authorization": "Bearer colab",
    "Content-Type": "application/json",
}

resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
data = resp.json()
print(data["choices"][0]["message"]["content"])
```

### Pattern: Programmatic Cluster Weight Sharding

```python
from colab_agent_bridge.cluster import ClusterBridge
from openai import OpenAI

cluster = ClusterBridge(
    session_name="agent-sharded-32b",
    profiles=["default", "personal_2"],
    gpu="T4",
    local_port=8000,
)

try:
    print("[Agent] Deploying sharded cluster across profiles...")
    result = cluster.deploy("Qwen/Qwen2.5-Coder-32B-Instruct", wait_for_ready=True)
    print(f"[Agent] Serving 32B model at {result['endpoint_url']} ({result['total_vram_gb']}GB pooled VRAM)")

    client = OpenAI(base_url=result["endpoint_url"], api_key="colab")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": "Analyze and refactor the repository architecture."}],
        temperature=0.1,
    )
    print("[Agent Output]:")
    print(response.choices[0].message.content)

finally:
    print("[Agent] Tearing down cluster to preserve compute units...")
    cluster.teardown()
```

---

## 6. Diagnostic & Error Recovery Playbook

| Issue Encountered | Root Cause | Agent Action / Remediation |
|---|---|---|
| `DefaultCredentialsError: Your default credentials were not found.` | ADC missing or expired. | Run `gcloud auth application-default login --scopes=...` with the 4 scopes listed in Section 3.2. |
| `HTTP 412: TooManyAssignmentsError` | Free tier limit of 1 active GPU reached on current account. | Target a secondary account profile: `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge up <model> --profile <profile_name>`. Or switch active profile: `colab-bridge profile switch <profile_name>`. |
| `Colab VM allocation failed: GPU quota exceeded` | The selected GPU tier is unavailable or out of compute units. | Fallback to a lower GPU (e.g. from `A100` to `L4` or `T4`), or specify standard CPU with `--high-mem`. |
| `⚠️ Active Compute Alert: Compute is currently active` | Untracked remote VM or idle session burning compute units. | Run `/run/media/parasaran/Dev/SDK/python/install/bin/colab-bridge down --all` to execute a global nuclear cleanup across all accounts. |
| `Health check timed out.` | Backend took longer than 600s to download weights, or crashed due to CUDA OOM. | Check Colab VM logs: run `/run/media/parasaran/Dev/SDK/python/install/bin/colab exec -s <session> -- "cat /content/vllm_server.log"`. Use quantized model (AWQ) or `backend="llama_cpp"`. |
| `Connection refused: localhost:8000` | SSH tunnel dropped or failed to bind. | Rerun `colab-bridge status -s <session>`. If VM is up, re-establish tunnel or switch to `--tunnel cloudflare`. |
| `Kernel disconnected / timeout` | Colab idle timeout triggered after prolonged inactivity. | `ColabBridge` runs an automatic keepalive thread every 180s. Ensure `keepalive_ping` is not blocked. |
| `Distributed cluster weight sharding requires at least two profiles` | Single profile supplied to `cluster up`. | Specify at least 2 distinct authenticated profiles (e.g. `--profiles default,personal_2`). |
| `Cluster coordinator binary 'llama-server' not found in PATH` | Local `llama-server` binary missing. | Install llama.cpp with RPC support or set `--coordinator-bin` or `LLAMA_SERVER_PATH`. |
| `Cluster node unreachable / RPC connection refused` | Remote RPC daemon crashed or tunnel port blocked. | Inspect remote log: `colab exec -s <session>-<profile> -- "cat /content/cluster_rpc_server.log"`. |

---

## 7. Model Recommendations by Task

For autonomous agent code generation and repository editing:

- **Fast / Lightweight & Reasoning (T4 16GB, L4 24GB):**
  - `cyankiwi/Ornith-1.5-9B-AWQ-INT4` / `ornith:9b` (Chain-of-Thought reasoning + tool calls on T4)
  - `Qwen/Qwen2.5-Coder-7B-Instruct` (Available in AWQ / GPTQ)
  - `deepseek-ai/deepseek-coder-6.7b-instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
- **Mid-Tier / Complex Code Reasoning (L4 24GB, A100 40GB):**
  - `Qwen/Qwen2.5-Coder-14B-Instruct`
  - `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- **Distributed Sharded Clusters (Multi-Account Free / Pro T4 & L4):**
  - `Qwen/Qwen2.5-Coder-32B-Instruct` (2x T4 GPUs = 32GB pooled VRAM; full 32B coding reasoning on free-tier accounts)
  - `meta-llama/Llama-3.3-70B-Instruct-AWQ` (4x T4 GPUs = 64GB pooled VRAM; 70B parameter frontier reasoning)
- **Heavyweight / Architectural Planning (A100 80GB, H100 80GB):**
  - `Qwen/Qwen2.5-Coder-32B-Instruct` (Unquantized FP16)
  - `meta-llama/Llama-3.3-70B-Instruct-AWQ`
  - `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`

For comprehensive hardware specifications and quantization trade-offs, read [docs/hardware_and_models.md](docs/hardware_and_models.md).
