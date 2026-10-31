from abc import ABC, abstractmethod


class BaseTunnel(ABC):
    @abstractmethod
    def start(self) -> str:
        """Starts the tunnel and returns the active endpoint URL."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the active tunnel."""
        pass
