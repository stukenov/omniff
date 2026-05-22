from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    route_class: str
    confidence: float
    thinking: str = "normal"


COMPLEX_KEYWORDS = [
    "проанализируй", "analyze", "research", "compare", "evaluate",
    "контракт", "contract", "legal", "юридич",
]


class KeywordRouter:
    def route(
        self,
        prompt: str,
        input_modality: str = "text",
        output_modality: str | None = None,
    ) -> RouteDecision:
        output_modality = output_modality or "text"
        prompt_lower = prompt.lower()

        if input_modality == "video":
            return RouteDecision("VIDEO_CAPTION", 0.85, "normal")

        if input_modality == "document":
            return RouteDecision("DOCUMENT_READ", 0.85, "normal")

        if input_modality == "audio":
            if prompt.strip():
                return RouteDecision("AUDIO_QA", 0.8, "normal")
            return RouteDecision("AUDIO_TRANSCRIBE_ONLY", 0.9, "fast")

        if input_modality == "image" and output_modality == "image":
            return RouteDecision("IMAGE_EDIT", 0.85, "normal")

        if input_modality == "image":
            return RouteDecision("IMAGE_CAPTION", 0.85, "fast")

        if output_modality == "image":
            return RouteDecision("TEXT_TO_IMAGE", 0.8, "normal")

        if any(kw in prompt_lower for kw in COMPLEX_KEYWORDS):
            return RouteDecision("TEXT_COMPLEX", 0.7, "deep")

        if len(prompt.split()) > 50:
            return RouteDecision("TEXT_NORMAL", 0.6, "normal")

        return RouteDecision("TEXT_SIMPLE", 0.8, "fast")
