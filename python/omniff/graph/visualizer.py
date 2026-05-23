from __future__ import annotations

from omniff.graph.types import OmniGraph


def render_ascii(graph: OmniGraph) -> str:
    if not graph.nodes:
        return "(empty graph)"

    lines = []
    for i, node in enumerate(graph.nodes):
        box = f"┌{'─' * (len(node.id) + 4)}┐\n│  {node.id}  │\n└{'─' * (len(node.id) + 4)}┘"
        if i > 0:
            lines.append("  │")
            lines.append("  ▼")
        lines.append(box)

    return "\n".join(lines)


def render_dot(graph: OmniGraph) -> str:
    lines = ["digraph omniff {", "  rankdir=LR;"]
    for node in graph.nodes:
        shape = "box"
        if "demux" in node.node_type:
            shape = "parallelogram"
        elif "mux" in node.node_type:
            shape = "parallelogram"
        elif "validate" in node.node_type:
            shape = "diamond"
        lines.append(f'  "{node.id}" [shape={shape}, label="{node.id}\\n({node.node_type})"];')

    for edge in graph.edges:
        lines.append(f'  "{edge.source}" -> "{edge.target}";')

    lines.append("}")
    return "\n".join(lines)
