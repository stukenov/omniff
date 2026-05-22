from __future__ import annotations

from pathlib import Path

import yaml

from omniff.graph.types import OmniGraph, OmniNode, Edge


def load_graph_template(path: Path) -> OmniGraph:
    with open(path) as f:
        raw = yaml.safe_load(f)

    g_raw = raw["graph"]
    graph = OmniGraph(id=g_raw["id"])

    for n in g_raw.get("nodes", []):
        node_type = n["node_type"]
        if isinstance(node_type, dict):
            for key, val in node_type.items():
                node_type = key
                break
        graph.add_node(OmniNode(
            id=n["id"],
            node_type=node_type,
            config=n.get("config", {}),
        ))

    for e in g_raw.get("edges", []):
        graph.add_edge(e["from"], e["to"])

    return graph
