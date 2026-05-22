import pytest

from omniff.models.vlm import VLMModel


def test_vlm_interface():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    assert not model.is_loaded


def test_vlm_infer_not_loaded():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"image_path": "test.jpg", "prompt": "describe"})
