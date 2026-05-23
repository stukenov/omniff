import pytest

from omniff.models.code import CodeModel, is_code_request


def test_code_model_interface():
    model = CodeModel(model_id="Qwen/Qwen3-4B")
    assert not model.is_loaded


def test_code_model_infer_not_loaded():
    model = CodeModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"prompt": "write hello world"})


def test_is_code_request_with_code_block():
    assert is_code_request("fix this:\n```python\ndef foo():\n  pass\n```")


def test_is_code_request_with_function():
    assert is_code_request("def calculate_sum(a, b):\n    return a + b")


def test_is_code_request_keywords():
    assert is_code_request("write code to refactor this function")


def test_is_code_request_plain_text():
    assert not is_code_request("what is the weather today?")


def test_is_code_request_single_keyword():
    assert not is_code_request("implement the plan")
