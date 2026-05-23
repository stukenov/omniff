import pytest
from omniff.router.encoder_router import EncoderRouter, ROUTE_LABELS, ROUTE_DESCRIPTIONS


def test_route_labels_match_descriptions():
    for label in ROUTE_LABELS:
        assert label in ROUTE_DESCRIPTIONS


def test_modality_overrides():
    router = EncoderRouter()
    assert router.route("x", input_modality="video").route_class == "VIDEO_CAPTION"
    assert router.route("x", input_modality="document").route_class == "DOCUMENT_READ"
    assert router.route("", input_modality="audio").route_class == "AUDIO_TRANSCRIBE_ONLY"
    assert router.route("question", input_modality="audio").route_class == "AUDIO_QA"
    assert router.route("x", input_modality="image", output_modality="image").route_class == "IMAGE_EDIT"
    assert router.route("x", input_modality="image").route_class == "IMAGE_CAPTION"


def test_encoder_route_text():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    router = EncoderRouter()
    decision = router.route("hello how are you")
    assert decision.route_class in ROUTE_LABELS
    assert decision.confidence > 0


def test_encoder_route_code_detection():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    router = EncoderRouter()
    decision = router.route("write a Python function that sorts a list")
    assert decision.route_class == "CODE"
