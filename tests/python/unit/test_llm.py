import pytest

from omniff.models.llm import LLMModel


def test_llm_interface():
    model = LLMModel(model_id="Qwen/Qwen3-0.6B", device="cpu", max_new_tokens=32)
    assert not model.is_loaded


def test_llm_infer_not_loaded():
    model = LLMModel(model_id="Qwen/Qwen3-0.6B", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"prompt": "hello"})
