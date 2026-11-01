# Coding Harness Integration Guide

This guide covers how to connect local coding agent harnesses—such as **Aider**, **Cline**, **Continue**, and custom Python OpenAI loops—to models hosted on Google Colab via `colab-agent-bridge`.

---

## 1. Quick Config Generator (`colab-bridge harness`)

The CLI provides an automated config generator that outputs ready-to-use commands and JSON snippets for your running endpoint:

```bash
colab-bridge harness \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct
```

This outputs `.env` variable blocks, Aider launch commands, and Continue JSON configurations matching your exact model name and port.

---

## 2. Integrating with Aider

[Aider](https://aider.chat/) is a terminal-based AI pair programming tool that works with local git repositories.

### Option A: Direct Command-Line Flags
Pass the endpoint and API key directly on invocation:

```bash
aider \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key colab \
  --model openai/Qwen/Qwen2.5-Coder-7B-Instruct
```

> [!NOTE]
> Prefix the model name with `openai/` so that Aider treats it as an OpenAI-compatible endpoint.

### Option B: Environment Variables
Create a `.env` file or export the following in your shell:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="colab"
export AIDER_MODEL="openai/Qwen/Qwen2.5-Coder-7B-Instruct"

# Launch directly
aider
```

---

## 3. Integrating with OpenCode (`opencode.ai`)

[OpenCode](https://opencode.ai/) is an autonomous AI coding agent and developer tool. You can configure OpenCode to connect to `colab-agent-bridge` via its `@ai-sdk/openai-compatible` provider in `~/.config/opencode/opencode.json`.

### Configuration Block (`~/.config/opencode/opencode.json`)

Add the `"colab"` provider under `"provider"`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "colab": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Google Colab (Bridge)",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "colab",
        "timeout": 600000
      },
      "models": {
        "default": {
          "name": "Colab Active Model",
          "tool_call": true,
          "reasoning": false,
          "attachment": false,
          "limit": { "context": 32768, "output": 8192 }
        },
        "qwen2.5-coder-7b": {
          "name": "Qwen 2.5 Coder 7B (Colab)",
          "tool_call": true,
          "reasoning": false,
          "attachment": false,
          "limit": { "context": 32768, "output": 8192 }
        },
        "qwen2.5-coder-32b": {
          "name": "Qwen 2.5 Coder 32B (Colab)",
          "tool_call": true,
          "reasoning": false,
          "attachment": false,
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

Then reload OpenCode with `opencode reload` or select `colab/default` directly from your OpenCode models picker.

---

## 4. Integrating with Cline (VS Code Extension)

[Cline](https://github.com/cline/cline) is an autonomous coding agent for Visual Studio Code capable of editing files, executing shell commands, and managing git branches.

### Setup Instructions

1. Open VS Code and open the Cline extension settings (gear icon in the Cline panel).
2. Set **API Provider** to `OpenAI Compatible`.
3. Configure the following fields:
   - **Base URL:** `http://localhost:8000/v1` *(or your Cloudflare tunnel URL)*
   - **API Key:** `colab` *(any non-empty string)*
   - **Model ID:** `Qwen/Qwen2.5-Coder-7B-Instruct`
4. Expand **Advanced Options**:
   - Enable **Streaming** (supported out of the box by vLLM, Ollama, and llama.cpp).
   - Set **Context Window** according to your model (e.g. `32768` for Qwen 2.5 Coder).
5. Click **Save Changes** and begin prompting Cline.

---

## 4. Integrating with Continue IDE Extension

[Continue](https://continue.dev/) is an open-source AI code assistant for VS Code and JetBrains IDEs.

### Configuration (`config.json`)

Add the model entry to `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Colab: Qwen 2.5 Coder 7B",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "colab",
      "contextLength": 32768
    }
  ],
  "tabAutocompleteModel": {
    "title": "Colab Autocomplete",
    "provider": "openai",
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "colab"
  }
}
```

Reload the Continue window (`Ctrl+Shift+P` -> `Developer: Reload Window`). Select "Colab: Qwen 2.5 Coder 7B" from the model dropdown.

---

## 5. Integrating with Custom OpenAI Agent Loops

If you are writing a custom autonomous agent, test harness, or benchmark runner using the official `openai` Python SDK (v1.x+):

### Streaming Chat Completions

```python
from openai import OpenAI

# Connect to the local forwarded bridge endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="colab",  # Colab bridge endpoints accept any placeholder key
)

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

stream = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a senior software engineer."},
        {"role": "user", "content": "Write a Rust function to safely parse semver strings."}
    ],
    temperature=0.1,
    stream=True,
)

for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)
print()
```

### JSON Structured Output & Tool Calling

Models such as `Qwen/Qwen2.5-Coder-7B-Instruct` and `Qwen/Qwen2.5-Coder-32B-Instruct` have strong native support for JSON schema enforcement and OpenAI function calling:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"}
                },
                "required": ["path"]
            }
        }
    }
]

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": "Please read the file /workspace/config.json"}
    ],
    tools=tools,
    tool_choice="auto"
)

tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    print(f"Tool to invoke: {tool_calls[0].function.name}")
    print(f"Arguments: {tool_calls[0].function.arguments}")
```

---

## 6. Tunnel Selection: Native SSH vs Cloudflare

`colab-agent-bridge` supports two tunneling architectures depending on your harness topology:

### 1. Native SSH Tunnel (`--tunnel ssh`)
- **Default and recommended** for local harnesses (Aider, VS Code extensions running on host).
- Maps `localhost:8000` directly to Colab's port `8000` through an encrypted SSH channel.
- Requires no external third-party accounts or software.

```bash
colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct --tunnel ssh --port 8000
```

### 2. Cloudflare Quick Tunnel (`--tunnel cloudflare`)
- Generates an ephemeral public HTTPS endpoint: `https://<random-id>.trycloudflare.com/v1`.
- Ideal for:
  - Running harnesses inside Docker containers where `localhost:8000` is isolated.
  - Team sharing or evaluating models from external development machines.
  - Webhook callbacks or remote CI/CD runners.

```bash
colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct --tunnel cloudflare
```

---

## 7. Preventing Colab Idle Timeouts (Keepalive)

Google Colab automatically shuts down runtimes if they remain idle without active execution for 30–90 minutes.

### Automatic Keepalive Daemon
When using `ColabBridge` or `colab-bridge up`, a background health keeper thread automatically issues lightweight `GET /v1/models` requests every 180 seconds. This maintains network traffic through the tunnel and prevents idle suspension while you read or inspect agent edits.

### Manual Endpoint Health Ping
To manually test connection health and latency:
```bash
colab-bridge test --endpoint http://localhost:8000/v1 --timeout 5
```
Or directly via curl:
```bash
curl -i http://localhost:8000/v1/models
```

---

## 8. Multi-Model Harness Routing across Accounts

When leveraging multi-account GPU pooling, you can route different coding tools to distinct models running simultaneously on separate Google Colab accounts:

```bash
# 1. Start primary coding model on Profile 1 (port 8000)
colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct --profile personal_1 --port 8000

# 2. Start deep reasoning model on Profile 2 (port 8002)
colab-bridge up deepseek-ai/deepseek-coder-6.7b-instruct --profile personal_2 --port 8002
```

Configure your environment to route tools:
```bash
# Point Aider to the 7B fast coder
export OPENAI_BASE_URL="http://localhost:8000/v1"
aider

# In Continue / OpenCode, set base URL to port 8002 for architectural reviews
"baseURL": "http://localhost:8002/v1"
```

Terminate all multi-account sessions when done:
```bash
colab-bridge down --all
```

---

## 9. Connecting to Distributed Sharded Clusters (Qwen 32B & Llama 70B)

When running frontier-scale coding models like **Qwen 2.5 Coder 32B** or **Meta Llama 3.3 70B**, `colab-agent-bridge` pools VRAM across 2–4 Google accounts and exposes a single unified endpoint on `http://localhost:8000/v1`.

### 1. Launching the Distributed Cluster
```bash
# Pool 2x T4 GPUs (32GB VRAM) across 'default' and 'personal_2' accounts
colab-bridge cluster up Qwen/Qwen2.5-Coder-32B-Instruct --profiles default,personal_2 --gpu T4
```

### 2. Connecting Aider to the 32B Sharded Cluster
Launch Aider pointing to the unified cluster endpoint:
```bash
aider \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key colab \
  --model openai/Qwen/Qwen2.5-Coder-32B-Instruct
```
Or via environment variables:
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="colab"
export AIDER_MODEL="openai/Qwen/Qwen2.5-Coder-32B-Instruct"
aider
```

### 3. Connecting OpenCode (`opencode.json`)
Configure the `colab` provider in `~/.config/opencode/opencode.json`:
```json
{
  "provider": {
    "colab": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "colab",
        "timeout": 600000
      },
      "models": {
        "qwen2.5-coder-32b": {
          "name": "Qwen 2.5 Coder 32B (Distributed Cluster)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```
OpenCode will route tool execution and autonomous agent steps through `colab_agent_bridge.adapter`, which normalizes tool calls and coordinates layer offloading across the remote Colab GPUs.

### 4. Cluster Teardown
```bash
colab-bridge cluster down --session cluster-bridge
```
