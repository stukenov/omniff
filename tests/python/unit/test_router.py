from omniff.router.keyword_router import KeywordRouter, RouteDecision


def test_route_text_simple():
    router = KeywordRouter()
    decision = router.route("Привет, как дела?")
    assert decision.route_class == "TEXT_SIMPLE"
    assert decision.confidence > 0.0


def test_route_image_caption():
    router = KeywordRouter()
    decision = router.route("describe this image", input_modality="image")
    assert decision.route_class == "IMAGE_CAPTION"


def test_route_audio_transcribe():
    router = KeywordRouter()
    decision = router.route("", input_modality="audio")
    assert decision.route_class == "AUDIO_TRANSCRIBE_ONLY"


def test_route_image_edit():
    router = KeywordRouter()
    decision = router.route(
        "make it matte graphite",
        input_modality="image",
        output_modality="image",
    )
    assert decision.route_class == "IMAGE_EDIT"


def test_route_text_complex():
    router = KeywordRouter()
    decision = router.route(
        "Проанализируй этот контракт на юридические риски и составь подробное резюме "
        "с указанием всех потенциальных проблем"
    )
    assert decision.route_class in ("TEXT_COMPLEX", "TEXT_NORMAL")


def test_route_video_caption():
    router = KeywordRouter()
    decision = router.route("describe this video", input_modality="video")
    assert decision.route_class == "VIDEO_CAPTION"


def test_route_document_read():
    router = KeywordRouter()
    decision = router.route("summarize this document", input_modality="document")
    assert decision.route_class == "DOCUMENT_READ"


def test_route_text_to_image():
    router = KeywordRouter()
    decision = router.route("a beautiful sunset", output_modality="image")
    assert decision.route_class == "TEXT_TO_IMAGE"


def test_route_decision_fields():
    router = KeywordRouter()
    decision = router.route("hello")
    assert isinstance(decision, RouteDecision)
    assert hasattr(decision, "route_class")
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "thinking")
