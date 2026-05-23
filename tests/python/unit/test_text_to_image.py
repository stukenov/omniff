import pytest

from omniff.models.text_to_image import TextToImageModel


def test_text_to_image_interface():
    model = TextToImageModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    assert not model.is_loaded


def test_text_to_image_infer_not_loaded():
    model = TextToImageModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"prompt": "a cat"})


def test_text_to_image_default_z_image():
    model = TextToImageModel()
    assert model.model_id == "Tongyi-MAI/Z-Image-Turbo"


def test_text_to_image_sdxl_legacy():
    model = TextToImageModel(model_id="stabilityai/sdxl-turbo")
    assert model.model_id == "stabilityai/sdxl-turbo"
