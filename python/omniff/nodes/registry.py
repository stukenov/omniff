from __future__ import annotations

from collections.abc import Callable
from typing import Any

NodeFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}

    def register(self, name: str, fn: NodeFn) -> None:
        self._nodes[name] = fn

    def get(self, name: str) -> NodeFn:
        if name not in self._nodes:
            raise KeyError(f"Node type not registered: {name}")
        return self._nodes[name]

    def has(self, name: str) -> bool:
        return name in self._nodes
