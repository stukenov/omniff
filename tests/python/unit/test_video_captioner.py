import pytest

from omniff.models.video_captioner import VideoCaptionerModel


def test_video_captioner_interface():
    model = VideoCaptionerModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    assert not model.is_loaded


def test_video_captioner_infer_not_loaded():
    model = VideoCaptionerModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"video_path": "test.mp4"})
