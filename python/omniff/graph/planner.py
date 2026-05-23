from __future__ import annotations

from typing import Any

from omniff.graph.types import OmniGraph, OmniNode


class GraphPlanner:
    """Builds execution graphs from route decisions and input parameters."""

    PIPELINE_TEMPLATES: dict[str, list[dict[str, Any]]] = {
        "TEXT_SIMPLE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "text"}},
            {"id": "llm", "type": "llm_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "IMAGE_CAPTION": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "image"}},
            {"id": "vlm", "type": "vlm_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "AUDIO_TRANSCRIBE_ONLY": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "audio"}},
            {"id": "asr", "type": "asr_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "IMAGE_EDIT": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "image"}},
            {"id": "edit", "type": "image_edit_infer", "config": {}},
            {"id": "validate", "type": "image_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "image"}},
        ],
        "TEXT_TO_IMAGE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "text"}},
            {"id": "gen", "type": "text_to_image_infer", "config": {}},
            {"id": "validate", "type": "image_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "image"}},
        ],
        "VIDEO_CAPTION": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "video"}},
            {"id": "captioner", "type": "video_caption_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "DOCUMENT_READ": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "document"}},
            {"id": "reader", "type": "document_read_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "TEXT_TO_SPEECH": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "text"}},
            {"id": "tts", "type": "tts_infer", "config": {}},
            {"id": "validate", "type": "audio_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "audio"}},
        ],
        "CODE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "text"}},
            {"id": "code", "type": "code_infer", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "DOCUMENT_TO_DOCUMENT": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "document"}},
            {"id": "reader", "type": "document_read_infer", "config": {}},
            {"id": "pdf_gen", "type": "pdf_generate", "config": {}},
            {"id": "validate", "type": "file_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "document"}},
        ],
        "VOICE_CHAIN": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "audio"}},
            {"id": "asr", "type": "asr_infer", "config": {}},
            {"id": "llm", "type": "llm_infer", "config": {}},
            {"id": "tts", "type": "tts_infer", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "audio"}},
        ],
        "AUDIO_TRANSLATE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "audio"}},
            {"id": "asr", "type": "asr_infer", "config": {}},
            {"id": "translate", "type": "llm_infer", "config": {"task": "translate"}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "AUDIO_DUB": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "audio"}},
            {"id": "asr", "type": "asr_infer", "config": {}},
            {"id": "translate", "type": "llm_infer", "config": {"task": "translate"}},
            {"id": "tts", "type": "tts_infer", "config": {}},
            {"id": "validate", "type": "audio_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "audio"}},
        ],
        "VIDEO_DUB": [
            {"id": "extract", "type": "ffmpeg_extract", "config": {"format": "wav"}},
            {"id": "asr", "type": "asr_infer", "config": {}},
            {"id": "translate", "type": "llm_infer", "config": {"task": "translate"}},
            {"id": "tts", "type": "tts_infer", "config": {}},
            {"id": "mux_av", "type": "ffmpeg_mux", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "video"}},
        ],
        "TRANSLATE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "text"}},
            {"id": "translate", "type": "nllb_translate", "config": {}},
            {"id": "validate", "type": "text_validate", "config": {"min_length": 1}},
            {"id": "mux", "type": "muxer", "config": {"modality": "text"}},
        ],
        "VOICE_CLONE": [
            {"id": "demux", "type": "demuxer", "config": {"modality": "audio"}},
            {"id": "clone", "type": "voice_clone_infer", "config": {}},
            {"id": "validate", "type": "audio_validate", "config": {}},
            {"id": "mux", "type": "muxer", "config": {"modality": "audio"}},
        ],
    }

    def plan(
        self,
        route_class: str,
        controls: dict[str, Any] | None = None,
    ) -> OmniGraph:
        controls = controls or {}
        template = self.PIPELINE_TEMPLATES.get(route_class)
        if not template:
            for key in ("TEXT_NORMAL", "TEXT_COMPLEX"):
                if key == route_class:
                    template = self.PIPELINE_TEMPLATES["TEXT_SIMPLE"]
                    break
            if not template:
                template = self.PIPELINE_TEMPLATES["TEXT_SIMPLE"]

        graph = OmniGraph(id=f"plan_{route_class.lower()}")

        for node_def in template:
            config = {**node_def["config"], **controls}
            graph.add_node(
                OmniNode(
                    id=node_def["id"],
                    node_type=node_def["type"],
                    config=config,
                )
            )

        for i in range(len(template) - 1):
            graph.add_edge(template[i]["id"], template[i + 1]["id"])

        return graph

    def available_routes(self) -> list[str]:
        return list(self.PIPELINE_TEMPLATES.keys())
