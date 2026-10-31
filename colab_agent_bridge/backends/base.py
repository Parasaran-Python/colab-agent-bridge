from abc import ABC, abstractmethod
from typing import Any


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
