import pytest

from omniff.graph.executor import GraphExecutor
from omniff.graph.types import OmniGraph, OmniNode
from omniff.nodes.registry import NodeRegistry


def make_test_graph():
    g = OmniGraph(id="test")
    g.add_node(OmniNode(id="step1", node_type="echo", config={"value": "hello"}))
    g.add_node(OmniNode(id="step2", node_type="echo", config={"value": "world"}))
    g.add_node(OmniNode(id="step3", node_type="concat", config={}))
    g.add_edge("step1", "step3")
    g.add_edge("step2", "step3")
    return g


def echo_node(inputs: dict, config: dict) -> dict:
    return {"text": config["value"]}


def concat_node(inputs: dict, config: dict) -> dict:
    texts = [v["text"] for v in inputs.values() if "text" in v]
    return {"text": " ".join(texts)}


def test_executor_topological_order():
    graph = make_test_graph()
    registry = NodeRegistry()
    registry.register("echo", echo_node)
    registry.register("concat", concat_node)
    executor = GraphExecutor(registry)
    result = executor.execute(graph, {})
    assert result["step3"]["text"] in ("hello world", "world hello")


def test_executor_single_node():
    g = OmniGraph(id="single")
    g.add_node(OmniNode(id="only", node_type="echo", config={"value": "test"}))
    registry = NodeRegistry()
    registry.register("echo", echo_node)
    executor = GraphExecutor(registry)
    result = executor.execute(g, {})
    assert result["only"]["text"] == "test"


def test_executor_missing_node_type():
    g = OmniGraph(id="bad")
    g.add_node(OmniNode(id="n1", node_type="nonexistent", config={}))
    registry = NodeRegistry()
    executor = GraphExecutor(registry)
    with pytest.raises(KeyError, match="nonexistent"):
        executor.execute(g, {})
