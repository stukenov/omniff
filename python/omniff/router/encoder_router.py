from __future__ import annotations

from typing import Any

from omniff.router.keyword_router import RouteDecision

ROUTE_LABELS = [
    "TEXT_SIMPLE",
    "TEXT_NORMAL",
    "TEXT_COMPLEX",
    "IMAGE_CAPTION",
    "AUDIO_TRANSCRIBE_ONLY",
    "AUDIO_QA",
    "IMAGE_EDIT",
    "TEXT_TO_IMAGE",
    "VIDEO_CAPTION",
    "DOCUMENT_READ",
    "TEXT_TO_SPEECH",
    "CODE",
    "DOCUMENT_TO_DOCUMENT",
]

ROUTE_DESCRIPTIONS = {
    "TEXT_SIMPLE": "simple text question, short answer, greeting, small talk",
    "TEXT_NORMAL": "moderate text question requiring a paragraph answer",
    "TEXT_COMPLEX": "complex analysis, research, comparison, legal, evaluation",
    "IMAGE_CAPTION": "describe an image, visual question answering",
    "AUDIO_TRANSCRIBE_ONLY": "transcribe audio speech to text",
    "AUDIO_QA": "transcribe audio and answer a question about it",
    "IMAGE_EDIT": "edit or modify an existing image",
    "TEXT_TO_IMAGE": "generate a new image from text description",
    "VIDEO_CAPTION": "describe or caption a video",
    "DOCUMENT_READ": "read, extract, or summarize a document",
    "TEXT_TO_SPEECH": "convert text to spoken audio, read aloud, TTS",
    "CODE": "write code, debug, refactor, fix code, implement function",
    "DOCUMENT_TO_DOCUMENT": "transform a document into another document format",
}


class EncoderRouter:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._label_embeddings = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self._model = SentenceTransformer(self.model_name)
        descriptions = [ROUTE_DESCRIPTIONS[label] for label in ROUTE_LABELS]
        self._label_embeddings = self._model.encode(descriptions, normalize_embeddings=True)

    def route(
        self,
        prompt: str,
        input_modality: str = "text",
        output_modality: str | None = None,
    ) -> RouteDecision:
        if input_modality == "video":
            return RouteDecision("VIDEO_CAPTION", 0.95, "normal")
        if input_modality == "document" and output_modality == "document":
            return RouteDecision("DOCUMENT_TO_DOCUMENT", 0.95, "normal")
        if input_modality == "document":
            return RouteDecision("DOCUMENT_READ", 0.95, "normal")
        if input_modality == "audio":
            if prompt.strip():
                return RouteDecision("AUDIO_QA", 0.9, "normal")
            return RouteDecision("AUDIO_TRANSCRIBE_ONLY", 0.95, "fast")
        if input_modality == "image" and output_modality == "image":
            return RouteDecision("IMAGE_EDIT", 0.95, "normal")
        if input_modality == "image":
            return RouteDecision("IMAGE_CAPTION", 0.95, "fast")

        self._load()
        import numpy as np

        query_emb = self._model.encode([prompt], normalize_embeddings=True)
        scores = np.dot(query_emb, self._label_embeddings.T)[0]

        if output_modality == "image":
            idx = ROUTE_LABELS.index("TEXT_TO_IMAGE")
            scores[idx] += 0.3
        if output_modality == "audio":
            idx = ROUTE_LABELS.index("TEXT_TO_SPEECH")
            scores[idx] += 0.3
        if output_modality == "document":
            idx = ROUTE_LABELS.index("DOCUMENT_TO_DOCUMENT")
            scores[idx] += 0.3

        best_idx = int(np.argmax(scores))
        route_class = ROUTE_LABELS[best_idx]
        confidence = float(scores[best_idx])

        thinking_map = {
            "TEXT_SIMPLE": "fast",
            "TEXT_NORMAL": "normal",
            "TEXT_COMPLEX": "deep",
            "CODE": "normal",
            "TEXT_TO_SPEECH": "off",
            "TEXT_TO_IMAGE": "normal",
        }
        thinking = thinking_map.get(route_class, "normal")

        return RouteDecision(route_class, confidence, thinking)
