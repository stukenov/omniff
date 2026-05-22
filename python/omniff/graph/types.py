from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    source: str
    target: str


@dataclass
class OmniNode:
    id: str
    node_type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniGraph:
    id: str
    nodes: list[OmniNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    side_data: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: OmniNode) -> None:
        self.nodes.append(node)

    def add_edge(self, source: str, target: str) -> None:
        self.edges.append(Edge(source=source, target=target))
