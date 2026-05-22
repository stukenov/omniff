import pytest

from omniff.models.llm import LLMModel


@pytest.fixture(scope="module")
def llm():
    model = LLMModel(model_id="Qwen/Qwen3-4B", device="auto", max_new_tokens=512)
    model.load()
    yield model
    model.unload()


def test_simple_response(llm):
    result = llm.infer({"prompt": "What is 2+2? Answer with just the number.", "thinking": False})
    assert "4" in result["text"]


def test_kazakh_response(llm):
    result = llm.infer({"prompt": "Қазақстанның астанасы қай қала? Тек қала атын жаз.", "thinking": False})
    assert len(result["text"]) > 0, "Model returned empty response"


def test_long_prompt(llm):
    result = llm.infer({"prompt": "List 3 colors. One word each, comma separated.", "thinking": False})
    assert len(result["text"]) > 3
