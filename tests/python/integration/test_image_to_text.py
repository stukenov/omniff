import pytest
from pathlib import Path

from omniff.models.vlm import VLMModel


@pytest.fixture(scope="module")
def vlm():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="auto")
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    path = tmp_path / "red_square.png"
    img.save(path)
    return str(path)


def test_describe_image(vlm, test_image):
    result = vlm.infer({"image_path": test_image, "prompt": "What color is this image? One word."})
    assert "red" in result["text"].lower()


def test_image_qa(vlm, test_image):
    result = vlm.infer({"image_path": test_image, "prompt": "What shape is shown?"})
    assert len(result["text"]) > 0
