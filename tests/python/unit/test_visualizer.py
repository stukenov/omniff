from omniff.graph.planner import GraphPlanner
from omniff.graph.visualizer import render_ascii, render_dot


def test_render_ascii():
    planner = GraphPlanner()
    graph = planner.plan("TEXT_SIMPLE")
    output = render_ascii(graph)
    assert "demux" in output
    assert "llm" in output
    assert "mux" in output


def test_render_dot():
    planner = GraphPlanner()
    graph = planner.plan("IMAGE_CAPTION")
    output = render_dot(graph)
    assert "digraph" in output
    assert "vlm" in output
    assert "->" in output


def test_render_ascii_empty():
    from omniff.graph.types import OmniGraph
    graph = OmniGraph(id="empty")
    assert render_ascii(graph) == "(empty graph)"


def test_render_dot_with_chain(tmp_path):
    from omniff.graph.chain import load_chain
    chain_file = tmp_path / "test.yaml"
    chain_file.write_text("""
id: test
nodes:
  - id: a
    type: demuxer
    config: {}
  - id: b
    type: llm_infer
    config: {}
edges:
  - from: a
    to: b
""")
    graph = load_chain(chain_file)
    dot = render_dot(graph)
    assert '"a"' in dot
    assert '"b"' in dot
