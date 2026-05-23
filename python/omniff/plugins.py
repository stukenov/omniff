from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class ModelPlugin:
    """Plugin interface for registering custom model implementations."""

    def __init__(
        self,
        name: str,
        model_cls: type[OmniModel],
        route_class: str,
        default_config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model_cls = model_cls
        self.route_class = route_class
        self.default_config = default_config or {}


class PluginRegistry:
    """Registry for model plugins. Allows third-party model registration."""

    def __init__(self) -> None:
        self._plugins: dict[str, ModelPlugin] = {}

    def register(self, plugin: ModelPlugin) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> ModelPlugin:
        if name not in self._plugins:
            raise KeyError(f"Plugin not registered: {name}")
        return self._plugins[name]

    def list(self) -> list[str]:
        return list(self._plugins.keys())

    def has(self, name: str) -> bool:
        return name in self._plugins

    def create_model(self, name: str, **overrides: Any) -> OmniModel:
        plugin = self.get(name)
        config = {**plugin.default_config, **overrides}
        return plugin.model_cls(**config)
