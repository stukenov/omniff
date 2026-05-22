import pytest

from omniff.models.asr import ASRModel


def test_asr_interface():
    model = ASRModel(model_id="openai/whisper-large-v3", device="cpu")
    assert not model.is_loaded


def test_asr_infer_not_loaded():
    model = ASRModel(model_id="openai/whisper-large-v3", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"audio_path": "test.wav"})
