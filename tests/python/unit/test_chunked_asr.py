import pytest

from omniff.models.chunked_asr import ChunkedASRModel


def test_chunked_asr_interface():
    model = ChunkedASRModel()
    assert not model.is_loaded
    assert model.chunk_length_s == 30.0


def test_chunked_asr_not_loaded():
    model = ChunkedASRModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"audio_path": "test.wav"})


def test_chunked_asr_stream_not_loaded():
    model = ChunkedASRModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        list(model.infer_stream({"audio_path": "test.wav"}))
