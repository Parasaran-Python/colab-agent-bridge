# Accelerator Hardware Matrix & Model Selection Guide

This guide details the hardware specifications of Google Colab accelerator tiers, memory sizing heuristics, recommended open-weight coding models, and quantization strategies for optimal agent performance.

---

## 1. Google Colab Accelerator Matrix

| Accelerator | Architecture | VRAM / Memory | Colab Tier | Typical CU Cost | Best Fit Serving Backend |
|---|---|---|---|---|---|
| **NVIDIA T4** | Turing | 16 GB GDDR6 | Free & Paid | ~1.0 CU / hr | vLLM (Quantized), llama.cpp |
| **NVIDIA L4** | Ada Lovelace | 24 GB GDDR6 | Pro & Pro+ | ~1.5 - 2.0 CU / hr | vLLM (FP16 / AWQ) |
| **NVIDIA G4** | Turing | 16 GB GDDR6 | Pro | ~1.2 CU / hr | vLLM (Quantized), Ollama |
| **NVIDIA A100** | Ampere | 40 GB / 80 GB HBM2e | Pro+ | ~2.5 - 4.5 CU / hr | vLLM (FP16, 70B AWQ) |
| **NVIDIA H100** | Hopper | 80 GB HBM3 | Pro+ Premium | ~5.0 - 8.0 CU / hr | vLLM (FP16, High-Throughput Batches) |
| **TPU v5e1** | Cloud TPU | 16 GB HBM | Pro / Pro+ | ~1.5 CU / hr | PyTorch / JAX XLA Serving |
| **TPU v6e1** | Trillium | 32 GB HBM | Pro+ | ~3.0 CU / hr | Next-Gen TPU Architectures |
| **CPU VM** | x86_64 | 12 GB (52 GB w/ `--high-mem`) | Free & Pro | ~0.1 - 0.5 CU / hr | llama.cpp (GGUF Q4_K_M) |

---

## 2. VRAM Sizing & Memory Heuristics

When deploying an LLM, VRAM must accommodate two major components:
1. **Model Weights:** The static memory consumed by the model parameters.
2. **KV Cache & Activation Overhead:** Dynamic memory needed for context window length (e.g. 8k to 32k tokens) and parallel requests.

### Rule-of-Thumb Calculations

| Precision / Quantization | Parameter Memory | KV Cache & Overhead | Example: 7B Model | Example: 32B Model | Example: 70B Model |
|---|---|---|---|---|---|
| **FP16 / BF16 (16-bit)** | ~2.0 GB per 1B params | + 2 to 4 GB | ~18 GB VRAM | ~68 GB VRAM | ~144 GB VRAM |
| **8-bit (FP8 / INT8)** | ~1.0 GB per 1B params | + 2 to 3 GB | ~10 GB VRAM | ~35 GB VRAM | ~75 GB VRAM |
| **4-bit (AWQ / GPTQ / GGUF)** | ~0.75 GB per 1B params | + 2 to 3 GB | ~7.5 GB VRAM | ~26 GB VRAM | ~56 GB VRAM |

---

## 2.1 Pooled Multi-GPU VRAM Economics & Distributed Weight Sharding

A fundamental bottleneck of Google Colab is the restriction of **1 active GPU per Google account**. A single free NVIDIA T4 provides 16GB VRAM, which comfortably hosts 7B-8B parameter models, but cannot allocate 32B or 70B parameter models without Out-Of-Memory (OOM) errors.

By utilizing `colab-agent-bridge`'s distributed cluster weight sharding (`colab-bridge cluster up`), multiple Google accounts can be combined into a single logical compute pool.

### Multi-Node Pooled VRAM Economics Matrix

| Cluster Shape | Raw VRAM | Usable Pooled VRAM | Supported Large Models | Multi-Account Cost |
|---|---|---|---|---|
| **2x NVIDIA T4 (16GB)** | 32 GB | ~29 - 30 GB | **Qwen 2.5 Coder 32B** (Q4_K_M / AWQ), DeepSeek 33B, Command R 35B | 2x Free tier accounts (0 CU) or ~2.0 CU/hr |
| **3x NVIDIA T4 (16GB)** | 48 GB | ~43 - 45 GB | **Qwen 2.5 Coder 32B** (FP16/BF16 unquantized), Llama 70B (IQ3/Q3_K_M) | 3x Free tier accounts (0 CU) or ~3.0 CU/hr |
| **4x NVIDIA T4 (16GB)** | 64 GB | ~58 - 60 GB | **Meta Llama 3.3 70B Instruct** (Q4_K_M / AWQ), Qwen 2.5 72B (AWQ) | 4x Free tier accounts (0 CU) or ~4.0 CU/hr |
| **2x NVIDIA L4 (24GB)** | 48 GB | ~45 GB | **Qwen 2.5 Coder 32B** (FP16 / BF16 unquantized) | 2x Colab Pro accounts (~3.5 CU/hr) |
| **2x NVIDIA A100 (40GB)** | 80 GB | ~76 GB | **Meta Llama 3.3 70B Instruct** (FP16 unquantized) | 2x Colab Pro+ accounts (~5.0 - 9.0 CU/hr) |

### How Distributed Weight Sharding Works
1. **Cloud-Native Coordinator Architecture:** Node 1 serves as the distributed coordinator inside Google Cloud, downloading GGUF weights directly from HuggingFace at 200+ MB/s and streaming secondary layer shards to Node 2 via a Cloudflare WebSocket P2P bridge.
2. **Zero Local Resource Consumption:** 0 GB of local disk space, <30 MB of local RAM, and 0 MB of local internet upload bandwidth.
3. **Layer Splitting via RPC:** Each remote Google Colab VM runs `llama.cpp`'s `rpc-server` daemon directly interfacing with the remote NVIDIA GPU (`CUDA0: Tesla T4`).
4. **Balanced Layer Allocation:** Model layers and tensors are split across remote nodes via `--tensor-split 1,1` (or customized ratios), enabling true multi-GPU pipeline parallelism without complex PyTorch cluster setups.

---

## 3. Recommended Coding Models

The following open-weight coding models have been validated for high performance and tool-calling compatibility:

### 1. Qwen 2.5 Coder Series (Alibaba)
Outstanding code generation, reasoning, multi-language syntax, and inline code completion.
- `Qwen/Qwen2.5-Coder-7B-Instruct`
  - **Best on:** T4 (AWQ/GPTQ) or L4 (Unquantized)
  - **Command:** `colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct --gpu L4`
- `Qwen/Qwen2.5-Coder-14B-Instruct`
  - **Best on:** L4 (AWQ) or A100 (Unquantized)
- `Qwen/Qwen2.5-Coder-32B-Instruct`
  - **Best on:** A100 (40GB AWQ or 80GB FP16)
  - Top-tier open-weight coding benchmark performance, rivaling proprietary commercial APIs.

### 2. DeepSeek Coder Series (DeepSeek-AI)
Known for deep architectural analysis, refactoring, and concise diff generation.
- `deepseek-ai/deepseek-coder-6.7b-instruct`
  - **Best on:** T4 or L4
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (16B MoE, 2.4B active)
  - **Best on:** T4 or L4 (Extremely fast token generation via Mixture-of-Experts)

### 3. Meta Llama 3.3 / 3.1 Series
High-capacity general reasoning with strong coding instruction adherence.
- `meta-llama/Llama-3.1-8B-Instruct`
  - **Best on:** T4 (AWQ) or L4 (FP16)
- `meta-llama/Llama-3.3-70B-Instruct-AWQ`
  - **Best on:** A100 80GB or H100 80GB

### 4. Ornith 1.5 9B (Reasoning & Agent Scaffolding)
Deep reasoning model fine-tuned for step-by-step thinking (`<think>`) and robust tool calling in autonomous agent loops.
- `cyankiwi/Ornith-1.5-9B-AWQ-INT4` / `ornith:9b`
  - **Best on:** T4 (16GB) or L4 (24GB)
  - **Memory:** ~6.5 GB VRAM in 4-bit AWQ / Ollama Q4_K_M
  - **Command:** `colab-bridge up ornith:9b --backend ollama --gpu T4`

### 5. StarCoder2 (BigCode / Hugging Face)
Pre-trained on 600+ programming languages, well suited for specialized languages and completion tasks.
- `bigcode/starcoder2-7b`
  - **Best on:** T4 or L4
- `bigcode/starcoder2-15b`
  - **Best on:** L4 (AWQ) or A100

---

## 4. Quantization Guide: AWQ vs GPTQ vs GGUF

Choosing the right quantization format is critical to maximizing token throughput without degrading code generation accuracy:

### Activation-Aware Weight Quantization (AWQ) — Recommended for vLLM
- **Mechanism:** Preserves the most salient weights based on activation distributions.
- **Speed:** Native CUDA acceleration with vLLM; highest tokens/second on NVIDIA GPUs.
- **Compatibility:** Native support in vLLM with `--quantization awq`.
- **Hugging Face Pattern:** Look for repos ending in `-AWQ` (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`).

### Generalized Post-Training Quantization (GPTQ)
- **Mechanism:** Second-order error minimization per layer.
- **Speed:** High inference throughput, slightly slower compilation time than AWQ.
- **Compatibility:** Supported in vLLM with `--quantization gptq`.
- **Hugging Face Pattern:** Look for repos ending in `-GPTQ`.

### GGUF (llama.cpp)
- **Mechanism:** Flexible k-quantization (e.g., `Q4_K_M`, `Q5_K_M`, `Q8_0`).
- **Speed:** Optimized for CPU inference and partial GPU offloading.
- **Compatibility:** Standard format for `llama.cpp` and `ollama`.
- **Best Use Case:** When running on Colab CPU instances with `--high-mem` or when GPUs have insufficient VRAM for vLLM.

---

## 5. Automated Recommendation Engine

`colab-agent-bridge` includes a built-in hardware advisor to determine the optimal configuration before spinning up resources.

### Example: Checking a 7B Model on T4
```bash
colab-bridge recommend Qwen/Qwen2.5-Coder-7B-Instruct --gpu T4
```
**Output:**
```text
Target Accelerator:   T4 (16GB VRAM)
Needs Quantization:   True
Recommended Backend:  vllm
Quantization Formats: awq, gptq, gguf
Advisory:             Model Qwen/Qwen2.5-Coder-7B-Instruct (~8B) exceeds FP16 capacity of T4 (16GB VRAM). Quantization required.
```

### Example: Checking a 32B Model on A100
```bash
colab-bridge recommend Qwen/Qwen2.5-Coder-32B-Instruct --gpu A100
```
**Output:**
```text
Target Accelerator:   A100 (40GB VRAM)
Needs Quantization:   True
Recommended Backend:  vllm
Quantization Formats: awq, gptq, gguf
Advisory:             Model Qwen/Qwen2.5-Coder-32B-Instruct (~32B) exceeds FP16 capacity of A100 (40GB VRAM). Quantization required.
```

### Example: Checking on CPU
```bash
colab-bridge recommend Qwen/Qwen2.5-Coder-7B-Instruct --gpu CPU
```
**Output:**
```text
Target Accelerator:   CPU
Needs Quantization:   True
Recommended Backend:  llama_cpp
Quantization Formats: gguf, q4_k_m
Advisory:             Running on CPU; model inference will be slow without GPU acceleration.
```

---

## 6. Multi-Account Hardware Pooling & Sharding

Because Google Colab limits free accounts to 1 concurrent GPU VM, multi-agent or multi-model coding workflows (e.g., pairing a fast code autocomplete model with an architectural reasoning model) previously required paid tiers or third-party cloud GPUs.

Using `colab-agent-bridge` multi-account profiles, you can pool hardware across separate Google accounts:

| Role | Account / Profile | Hardware Tier | Recommended Model | Port |
|---|---|---|---|---|
| **Fast Completion / Infill** | Profile `personal_1` | T4 (16GB) | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | `8000` |
| **Deep Reasoning & Review** | Profile `personal_2` | T4 or L4 (24GB) | `deepseek-ai/deepseek-coder-6.7b-instruct` | `8002` |

Deploy concurrently:
```bash
# Terminal 1: Fast completion engine
colab-bridge up Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --profile personal_1 --port 8000

# Terminal 2: Deep reasoning engine
colab-bridge up deepseek-ai/deepseek-coder-6.7b-instruct --profile personal_2 --port 8002
```

View active compute across accounts at any time:
```bash
colab-bridge status --all
```
