from __future__ import annotations

from pathlib import Path

import yaml

from omniff.graph.types import OmniGraph, OmniNode


def load_chain(path: str | Path) -> OmniGraph:
    path = Path(path)
    with open(path) as f:
        spec = yaml.safe_load(f)

    graph_id = spec.get("id", path.stem)
    graph = OmniGraph(id=graph_id)

    for node_spec in spec.get("nodes", []):
        graph.add_node(
            OmniNode(
                id=node_spec["id"],
                node_type=node_spec["type"],
                config=node_spec.get("config", {}),
            )
        )

    for edge_spec in spec.get("edges", []):
        graph.add_edge(edge_spec["from"], edge_spec["to"])

    return graph


def list_chains(directory: str | Path) -> list[dict[str, str]]:
    directory = Path(directory)
    if not directory.exists():
        return []
    chains = []
    for f in sorted(directory.glob("*.yaml")):
        try:
            with open(f) as fh:
                spec = yaml.safe_load(fh)
            chains.append(
                {
                    "id": spec.get("id", f.stem),
                    "name": spec.get("name", f.stem),
                    "description": spec.get("description", ""),
                    "path": str(f),
                }
            )
        except Exception:
            pass
    return chains
