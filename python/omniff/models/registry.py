from __future__ import annotations

from omniff.models.base import OmniModel


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, OmniModel] = {}

    def register(self, name: str, model: OmniModel) -> None:
        self._models[name] = model

    def get(self, name: str) -> OmniModel:
        if name not in self._models:
            raise KeyError(f"Model not registered: {name}")
        return self._models[name]

    def load(self, name: str) -> None:
        self.get(name).load()

    def unload(self, name: str) -> None:
        self.get(name).unload()

    def list(self) -> list[str]:
        return list(self._models.keys())

    def has(self, name: str) -> bool:
        return name in self._models
