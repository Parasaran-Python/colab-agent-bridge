# Cloud-Native Distributed Cluster Coordinator & P2P Weight Sharding Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shift the distributed cluster coordinator (`llama-server`) from the local user machine to Colab Node 1, connecting to Colab Node 2 via a high-speed Cloudflare WebSocket P2P bridge, eliminating all local disk, RAM, and upload bottlenecks so 32B models deploy in ~3 minutes.

**Architecture:** 
- Node 1 (Coordinator on primary profile, e.g. `default`) runs `llama-server` with CUDA support on Colab, downloading weights directly from HuggingFace at 200+ MB/s.
- Node 2 (Worker on secondary profile, e.g. `sitonam`) runs `ggml-rpc-server` + `websocat` + `cloudflared` to expose a high-speed Cloudflare tunnel endpoint.
- Node 1 runs a `websocat` client bridging `127.0.0.1:50052` to Node 2's Cloudflare tunnel endpoint.
- Local host machine runs only a lightweight SSH tunnel forwarding `localhost:8000` to Node 1's port 8000 (0 GB local disk, <30 MB local RAM, 0 GB local upload bandwidth).

**Tech Stack:** Python 3.10+, Google Colab CLI, `llama.cpp` (`llama-server`, `ggml-rpc-server`), `websocat` (musl static), `cloudflared`, pytest.

**Spec:** PR #2 on `feature/cluster-sharding`

## Global Constraints
- **Zero Local Disk Footprint**: Do NOT download GGUF model files to the host machine.
- **Zero Local RAM Footprint**: Host machine must not run local `llama-server`.
- **Zero Local Upload Footprint**: Model weights must never flow through host internet upload.
- **Teardown Rule**: Always release both Colab sessions upon completion, failure, or cancellation.
- **Documentation Rule**: Keep `README.md`, `AGENTS.md`, and `docs/` 100% synchronized.
- **Git Branch Rule**: All commits on `feature/cluster-sharding`; do not push to `main`.
- **Device Rule**: Audible TTS notification via `spd-say` on task completion.

---

### Task 1: Remote Coordinator & P2P Tunnel Script Generation in `worker.py`

**Files:**
- Modify: `colab_agent_bridge/cluster/worker.py`
- Modify: `tests/test_cluster.py`

**Interfaces:**
- Consumes: `ClusterNode` from `colab_agent_bridge/cluster/models.py`
- Produces: 
  - `generate_worker_start_script(port, enable_tunnel=True)`
  - `generate_coordinator_start_script(model_name, rpc_endpoints, port=8000, tensor_split="1,1", ngl=999)`

- [ ] **Step 1: Write unit tests for `generate_coordinator_start_script` and P2P tunneling**
  - Verify script downloads `websocat` if missing.
  - Verify script starts `llama-server` on port 8000 with CUDA library path and `--rpc` targeting the bridge.
  - Verify worker script starts `websocat` + `cloudflared` when `enable_tunnel=True`.

- [ ] **Step 2: Implement `generate_coordinator_start_script` in `worker.py`**
  - Add helper to download static `websocat` binary to `/content/websocat`.
  - Add helper to bridge remote Cloudflare WebSocket to local TCP port on coordinator node.
  - Generate start command for `llama-server` with `--hf-repo`, `--port 8000`, `--rpc 127.0.0.1:50052`, `--tensor-split 1,1`, `-ngl 999`.

- [ ] **Step 3: Run pytest on `tests/test_cluster.py` and ensure 100% pass rate**
  - Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/test_cluster.py -v`

---

### Task 2: Refactor `ClusterCoordinator` to Manage Cloud-Native Remote Coordinator

**Files:**
- Modify: `colab_agent_bridge/cluster/coordinator.py`
- Modify: `tests/test_cluster.py`

**Interfaces:**
- Consumes: Remote Node 1 SSH tunnel / execution client
- Produces: `ClusterCoordinator.start_remote(node1, node2_tunnel_url, model_name)` and `ClusterCoordinator.stop_remote()`

- [ ] **Step 1: Write unit tests for remote coordinator lifecycle in `tests/test_cluster.py`**
  - Test remote startup invocation without spawning local `llama-server` subprocess.
  - Test health check polling against forwarded port 8000.

- [ ] **Step 2: Update `ClusterCoordinator` in `coordinator.py`**
  - Remove requirement for local `llama-server` binary on host machine when running in cloud-coordinator mode.
  - Implement remote script execution via `colab exec -s <node1_session>`.
  - Establish local SSH tunnel forwarding `localhost:8000` to Node 1's port 8000.

- [ ] **Step 3: Run pytest and verify all tests pass**
  - Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/ -v`

---

### Task 3: Update `ClusterManager` Orchestration Flow

**Files:**
- Modify: `colab_agent_bridge/cluster/manager.py`
- Modify: `tests/test_cluster.py`

**Interfaces:**
- Consumes: `ClusterNode`, `ClusterCoordinator`, `ColabBridge`
- Produces: `ClusterManager.deploy()` with zero host disk/RAM usage

- [ ] **Step 1: Write unit tests for full cloud-native cluster deployment flow**
  - Mock Node 2 starting `cloudflared` and returning `https://*.trycloudflare.com`.
  - Mock Node 1 starting `websocat` and `llama-server`.
  - Verify local port 8000 forwards directly to Node 1.

- [ ] **Step 2: Update `ClusterManager.deploy()` in `manager.py`**
  - Provision Node 1 (coordinator) and Node 2 (worker) in parallel.
  - Set up Node 2 worker and extract its Cloudflare tunnel URL.
  - Connect Node 1 coordinator to Node 2 tunnel URL.
  - Launch remote `llama-server` on Node 1.
  - Forward port 8000 to local machine and verify OpenAI endpoint `/v1/models`.

- [ ] **Step 3: Run test suite**
  - Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/ -v`

---

### Task 4: Documentation Synchronization & PR Commit

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/hardware_and_models.md`

- [ ] **Step 1: Update documentation files**
  - Explain the Cloud-Native Coordinator + P2P Cloudflare Tunnel architecture.
  - Highlight zero local disk, zero local RAM, and ~3-minute boot times.
  - Update CLI examples and architecture diagrams.

- [ ] **Step 2: Run full regression test suite**
  - Run: `/run/media/parasaran/Dev/SDK/python/install/bin/pytest tests/ -v`

- [ ] **Step 3: Commit to `feature/cluster-sharding` and push to PR #2**
  - Ensure zero commits directly to `main`.

- [ ] **Step 4: Trigger TTS notification**
  - Execute: `spd-say "Cloud-native distributed coordinator implementation ready"`
