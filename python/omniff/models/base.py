from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OmniModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]: ...

    def generate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.infer(inputs)
