from omniff.integrations.langchain import OmniFFTool


def test_omniff_tool_init():
    tool = OmniFFTool()
    assert tool.name == "omniff"
    assert tool.base_url == "http://localhost:8000"
    assert "multimodal" in tool.description.lower()


def test_omniff_tool_custom_url():
    tool = OmniFFTool(base_url="http://gpu-server:9000")
    assert tool.base_url == "http://gpu-server:9000"
