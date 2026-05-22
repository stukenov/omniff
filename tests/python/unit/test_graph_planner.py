from omniff.graph.planner import GraphPlanner


def test_plan_text_simple():
    planner = GraphPlanner()
    graph = planner.plan("TEXT_SIMPLE")
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    assert graph.nodes[0].node_type == "demuxer"
    assert graph.nodes[-1].node_type == "muxer"


def test_plan_image_caption():
    planner = GraphPlanner()
    graph = planner.plan("IMAGE_CAPTION")
    node_types = [n.node_type for n in graph.nodes]
    assert "vlm_infer" in node_types


def test_plan_audio_transcribe():
    planner = GraphPlanner()
    graph = planner.plan("AUDIO_TRANSCRIBE_ONLY")
    node_types = [n.node_type for n in graph.nodes]
    assert "asr_infer" in node_types


def test_plan_with_controls():
    planner = GraphPlanner()
    graph = planner.plan("TEXT_SIMPLE", controls={"model_id": "test/model"})
    for node in graph.nodes:
        assert "model_id" in node.config


def test_plan_fallback_unknown_route():
    planner = GraphPlanner()
    graph = planner.plan("UNKNOWN_ROUTE")
    assert len(graph.nodes) == 4


def test_plan_text_normal_uses_text_simple_template():
    planner = GraphPlanner()
    graph = planner.plan("TEXT_NORMAL")
    node_types = [n.node_type for n in graph.nodes]
    assert "llm_infer" in node_types


def test_available_routes():
    planner = GraphPlanner()
    routes = planner.available_routes()
    assert "TEXT_SIMPLE" in routes
    assert "IMAGE_CAPTION" in routes
    assert len(routes) >= 7


def test_plan_edges_are_sequential():
    planner = GraphPlanner()
    graph = planner.plan("TEXT_SIMPLE")
    for i, edge in enumerate(graph.edges):
        assert edge.source == graph.nodes[i].id
        assert edge.target == graph.nodes[i + 1].id
