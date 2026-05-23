import pytest
from omniff.models.tts import TTSModel


def test_tts_interface():
    model = TTSModel(model_id="suno/bark-small")
    assert not model.is_loaded
    assert model.model_id == "suno/bark-small"


def test_tts_infer_not_loaded():
    model = TTSModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"text": "hello"})
