from omniff.validators.text_validator import TextValidator
from omniff.validators.image_validator import ImageValidator


def test_text_validator_pass():
    v = TextValidator(min_length=1)
    result = v.validate({"text": "Hello world"})
    assert result.passed
    assert result.score > 0.5


def test_text_validator_fail_empty():
    v = TextValidator(min_length=1)
    result = v.validate({"text": ""})
    assert not result.passed


def test_text_validator_fail_none():
    v = TextValidator()
    result = v.validate({})
    assert not result.passed


def test_image_validator_pass(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    v = ImageValidator()
    result = v.validate({"image_path": str(img_path)})
    assert result.passed


def test_image_validator_fail_missing():
    v = ImageValidator()
    result = v.validate({"image_path": "/nonexistent.jpg"})
    assert not result.passed
