"""Integration tests for multi-pipelines on GPU (KazNU server)."""

import os
import tempfile

import pytest

from omniff.runtime.config import OmniFFConfig, RouterConfig
from omniff.runtime.engine import OmniFFRuntime


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_test_audio():
    """Generate a short WAV with silence (2 seconds) for testing."""
    import numpy as np
    import scipy.io.wavfile

    sr = 16000
    duration = 2
    audio = np.zeros(sr * duration, dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    scipy.io.wavfile.write(tmp.name, sr, audio)
    return tmp.name


@pytest.fixture
def runtime():
    config = OmniFFConfig(
        name="test",
        version="1.0",
        router=RouterConfig(router_type="keyword"),
        experts={},
    )
    return OmniFFRuntime(config)


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestAudioTranslateGPU:
    def test_audio_translate_runs(self, runtime):
        audio_path = _make_test_audio()
        try:
            result = runtime.run(
                input=audio_path,
                prompt="translate to english",
                controls={"target_language": "English"},
            )
            assert result.route == "AUDIO_TRANSLATE"
            assert result.output_text is not None
            assert result.metadata["target_language"] == "English"
        finally:
            os.unlink(audio_path)


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestAudioDubGPU:
    def test_audio_dub_produces_file(self, runtime):
        audio_path = _make_test_audio()
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            result = runtime.run(
                input=audio_path,
                prompt="dub this audio",
                output=output_path,
                controls={"target_language": "English"},
            )
            assert result.route == "AUDIO_DUB"
            assert result.output_path is not None
            assert os.path.exists(result.output_path)
        finally:
            os.unlink(audio_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
