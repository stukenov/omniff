import pytest
import numpy as np

from omniff.models.asr import ASRModel


@pytest.fixture(scope="module")
def asr():
    model = ASRModel(model_id="openai/whisper-large-v3", device="auto")
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_audio(tmp_path):
    import soundfile as sf
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sine.wav"
    sf.write(str(path), audio, sr)
    return str(path)


def test_transcribe_audio(asr, test_audio):
    result = asr.infer({"audio_path": test_audio})
    assert "text" in result
    assert isinstance(result["text"], str)


def test_transcribe_returns_chunks(asr, test_audio):
    result = asr.infer({"audio_path": test_audio})
    assert "chunks" in result
