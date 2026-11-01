"""Colab Agent Bridge: Orchestrate Colab GPUs, serve models, and bridge to coding agents."""

from colab_agent_bridge.client.bridge import ColabBridge
from colab_agent_bridge.cluster.manager import ClusterBridge, ClusterManager

__version__ = "0.1.0"
__all__ = ["ColabBridge", "ClusterBridge", "ClusterManager", "__version__"]
