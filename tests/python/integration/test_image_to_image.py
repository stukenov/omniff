from pathlib import Path

import pytest

from omniff.models.image_edit import ImageEditModel


@pytest.fixture(scope="module")
def image_edit():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="auto", num_inference_steps=2)
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (512, 512), color=(100, 100, 200))
    path = tmp_path / "input.png"
    img.save(path)
    return str(path)


def test_image_edit_produces_output(image_edit, test_image, tmp_path):
    output_path = str(tmp_path / "output.png")
    result = image_edit.infer(
        {
            "image_path": test_image,
            "prompt": "a beautiful sunset landscape",
            "strength": 0.6,
            "output_path": output_path,
            "seed": 42,
        }
    )
    assert Path(result["image_path"]).exists()
    assert Path(result["image_path"]).stat().st_size > 1000


def test_image_edit_preserves_with_low_strength(image_edit, test_image, tmp_path):
    output_path = str(tmp_path / "output_low.png")
    result = image_edit.infer(
        {
            "image_path": test_image,
            "prompt": "same image but slightly warmer tones",
            "strength": 0.15,
            "output_path": output_path,
            "seed": 42,
        }
    )
    assert Path(result["image_path"]).exists()
