import pytest

from omniff.models.image_edit import ImageEditModel


def test_image_edit_interface():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    assert not model.is_loaded


def test_image_edit_infer_not_loaded():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"image_path": "test.jpg", "prompt": "make it blue"})
