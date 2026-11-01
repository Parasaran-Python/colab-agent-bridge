import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from colab_agent_bridge.cluster.models import ClusterInfo

DEFAULT_STATE_FILE = Path.home() / ".config" / "colab-bridge" / "clusters.json"


class ClusterStateStore:
    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or DEFAULT_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            content = self.state_file.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            return json.loads(content)
        except Exception:
            return {}

    def _save_raw(self, data: Dict[str, Any]) -> None:
        try:
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get(self, session_name: str) -> Optional[ClusterInfo]:
        data = self._load_raw()
        if session_name in data:
            try:
                return ClusterInfo.from_dict(data[session_name])
            except Exception:
                return None
        return None

    def save(self, cluster: ClusterInfo) -> None:
        data = self._load_raw()
        data[cluster.session_name] = cluster.to_dict()
        self._save_raw(data)

    def list_all(self) -> List[ClusterInfo]:
        data = self._load_raw()
        clusters = []
        for s_name, item in data.items():
            try:
                clusters.append(ClusterInfo.from_dict(item))
            except Exception:
                continue
        return clusters

    def delete(self, session_name: str) -> bool:
        data = self._load_raw()
        if session_name in data:
            del data[session_name]
            self._save_raw(data)
            return True
        return False

    def clear(self) -> None:
        self._save_raw({})
