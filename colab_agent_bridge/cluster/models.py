import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClusterNode:
    profile: str
    session_name: str
    local_rpc_port: int
    remote_rpc_port: int = 50052
    gpu: Optional[str] = "T4"
    status: str = "initialized"
    role: str = "worker"
    public_tunnel_url: Optional[str] = None
    tunnel: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "session_name": self.session_name,
            "local_rpc_port": self.local_rpc_port,
            "remote_rpc_port": self.remote_rpc_port,
            "gpu": self.gpu,
            "status": self.status,
            "role": self.role,
            "public_tunnel_url": self.public_tunnel_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterNode":
        return cls(
            profile=data["profile"],
            session_name=data["session_name"],
            local_rpc_port=data["local_rpc_port"],
            remote_rpc_port=data.get("remote_rpc_port", 50052),
            gpu=data.get("gpu", "T4"),
            status=data.get("status", "unknown"),
            role=data.get("role", "worker"),
            public_tunnel_url=data.get("public_tunnel_url"),
        )


@dataclass
class ClusterInfo:
    session_name: str
    model_name: str
    profiles: List[str]
    gpu: str
    nodes: List[ClusterNode]
    coordinator_port: int
    listen_port: int
    endpoint_url: str
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    total_vram_gb: int = 0
    usable_vram_gb: int = 0
    tensor_split: Optional[str] = None
    coordinator_pid: Optional[int] = None
    adapter_pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_name": self.session_name,
            "model_name": self.model_name,
            "profiles": self.profiles,
            "gpu": self.gpu,
            "nodes": [n.to_dict() for n in self.nodes],
            "coordinator_port": self.coordinator_port,
            "listen_port": self.listen_port,
            "endpoint_url": self.endpoint_url,
            "status": self.status,
            "created_at": self.created_at,
            "total_vram_gb": self.total_vram_gb,
            "usable_vram_gb": self.usable_vram_gb,
            "tensor_split": self.tensor_split,
            "coordinator_pid": self.coordinator_pid,
            "adapter_pid": self.adapter_pid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterInfo":
        nodes = [ClusterNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            session_name=data["session_name"],
            model_name=data["model_name"],
            profiles=data.get("profiles", []),
            gpu=data.get("gpu", "T4"),
            nodes=nodes,
            coordinator_port=data["coordinator_port"],
            listen_port=data["listen_port"],
            endpoint_url=data["endpoint_url"],
            status=data.get("status", "unknown"),
            created_at=data.get("created_at", time.time()),
            total_vram_gb=data.get("total_vram_gb", 0),
            usable_vram_gb=data.get("usable_vram_gb", 0),
            tensor_split=data.get("tensor_split"),
            coordinator_pid=data.get("coordinator_pid"),
            adapter_pid=data.get("adapter_pid"),
        )
