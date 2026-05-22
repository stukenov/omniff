import pytest

from omniff.models.llm import LLMModel


@pytest.fixture(scope="module")
def llm():
    model = LLMModel(model_id="Qwen/Qwen3-4B", device="auto", max_new_tokens=128)
    model.load()
    yield model
    model.unload()


def test_simple_response(llm):
    result = llm.infer({"prompt": "What is 2+2? Answer with just the number."})
    assert "4" in result["text"]


def test_kazakh_response(llm):
    result = llm.infer({"prompt": "Қазақстанның астанасы қай қала? Тек қала атын жаз."})
    text_lower = result["text"].lower()
    assert any(w in text_lower for w in ("астана", "astana", "нұр-сұлтан", "nur-sultan"))


def test_long_prompt(llm):
    result = llm.infer({"prompt": "List 3 colors. One word each, comma separated."})
    assert len(result["text"]) > 3
