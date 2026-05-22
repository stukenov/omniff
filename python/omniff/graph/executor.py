from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from omniff.graph.types import OmniGraph
from omniff.nodes.registry import NodeRegistry


class GraphExecutor:
    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry

    def execute(self, graph: OmniGraph, initial_inputs: dict[str, Any]) -> dict[str, Any]:
        order = self._topological_sort(graph)
        results: dict[str, Any] = {}

        for node in order:
            fn = self.registry.get(node.node_type)
            node_inputs = {}
            for edge in graph.edges:
                if edge.target == node.id and edge.source in results:
                    node_inputs[edge.source] = results[edge.source]
            if not node_inputs and node.id in initial_inputs:
                node_inputs = initial_inputs[node.id]
            results[node.id] = fn(node_inputs, node.config)

        return results

    def _topological_sort(self, graph: OmniGraph) -> list:
        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)
        node_map = {n.id: n for n in graph.nodes}

        for n in graph.nodes:
            in_degree.setdefault(n.id, 0)
        for e in graph.edges:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result = []
        while queue:
            nid = queue.popleft()
            result.append(node_map[nid])
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(graph.nodes):
            raise ValueError("Graph has a cycle")
        return result
