"""Tests for VoiceClonerModel."""

from omniff.models.voice_cloner import VoiceClonerModel


def test_voice_cloner_interface():
    model = VoiceClonerModel()
    assert model.model_id == "k2-fsa/OmniVoice"
    assert not model.is_loaded


def test_voice_cloner_cosyvoice():
    model = VoiceClonerModel(model_id="FunAudioLLM/CosyVoice2-0.5B")
    assert model.model_id == "FunAudioLLM/CosyVoice2-0.5B"


def test_voice_cloner_infer_not_loaded():
    model = VoiceClonerModel()
    try:
        model.infer({"text": "hello"})
        assert False
    except RuntimeError:
        pass


def test_voice_cloner_custom_model_id():
    model = VoiceClonerModel(model_id="custom/model")
    assert model.model_id == "custom/model"
