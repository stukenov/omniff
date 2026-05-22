import pytest
from pathlib import Path

from omniff.models.text_to_image import TextToImageModel


@pytest.fixture(scope="module")
def txt2img():
    model = TextToImageModel(model_id="stabilityai/sdxl-turbo", device="auto", num_inference_steps=2)
    model.load()
    yield model
    model.unload()


def test_generate_image(txt2img, tmp_path):
    output_path = str(tmp_path / "generated.png")
    result = txt2img.infer({
        "prompt": "a red circle on a white background",
        "output_path": output_path,
        "seed": 42,
    })
    assert Path(result["image_path"]).exists()
    assert Path(result["image_path"]).stat().st_size > 1000


def test_generate_with_dimensions(txt2img, tmp_path):
    output_path = str(tmp_path / "wide.png")
    result = txt2img.infer({
        "prompt": "a sunset landscape",
        "output_path": output_path,
        "width": 512,
        "height": 512,
        "seed": 123,
    })
    assert Path(result["image_path"]).exists()
