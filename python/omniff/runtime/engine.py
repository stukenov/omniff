from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.runtime.config import OmniFFConfig
from omniff.runtime.result import RunResult
from omniff.router.keyword_router import KeywordRouter
from omniff.models.registry import ModelRegistry
from omniff.filters.language import detect_language


def _detect_input_modality(input_path: str) -> str:
    if not Path(input_path).exists():
        return "text"
    suffix = Path(input_path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        return "image"
    if suffix in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        return "audio"
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        return "video"
    if suffix in (".pdf", ".docx", ".doc", ".txt"):
        return "document"
    return "text"


class OmniFFRuntime:
    def __init__(self, config: OmniFFConfig) -> None:
        self.config = config
        self.router = KeywordRouter()
        self.models = ModelRegistry()

    @classmethod
    def from_yaml(cls, path: str | Path) -> OmniFFRuntime:
        config = OmniFFConfig.load(Path(path))
        return cls(config)

    def _ensure_model(self, name: str, model_cls: type, **kwargs: Any) -> Any:
        if not self.models.has(name):
            model = model_cls(**kwargs)
            self.models.register(name, model)
        m = self.models.get(name)
        if not m.is_loaded:
            m.load()
        return m

    def run(
        self,
        input: str,
        prompt: str | None = None,
        output_modality: str | None = None,
        thinking: str = "normal",
        controls: dict[str, Any] | None = None,
        output: str | None = None,
    ) -> RunResult:
        controls = controls or {}
        input_modality = _detect_input_modality(input)

        if input_modality == "text" and not Path(input).exists():
            prompt = prompt or input
            input_modality = "text"

        output_modality = output_modality or "text"

        route = self.router.route(prompt or "", input_modality, output_modality)

        if route.route_class in ("TEXT_SIMPLE", "TEXT_NORMAL", "TEXT_COMPLEX"):
            return self._run_text_to_text(prompt or input, route.route_class, controls)

        if route.route_class == "IMAGE_CAPTION":
            return self._run_image_to_text(input, prompt, controls)

        if route.route_class in ("AUDIO_TRANSCRIBE_ONLY", "AUDIO_QA"):
            return self._run_audio_to_text(input, prompt, controls)

        if route.route_class == "IMAGE_EDIT":
            return self._run_image_to_image(input, prompt or "", controls, output)

        return RunResult(output_text=f"Unsupported route: {route.route_class}", route=route.route_class)

    def _run_text_to_text(self, prompt: str, route: str, controls: dict) -> RunResult:
        from omniff.models.llm import LLMModel

        model_id = controls.get("model_id", "Qwen/Qwen3-4B")
        llm = self._ensure_model("llm", LLMModel, model_id=model_id, device="auto")
        result = llm.infer({"prompt": prompt})
        return RunResult(output_text=result["text"], route=route)

    def _run_image_to_text(self, image_path: str, prompt: str | None, controls: dict) -> RunResult:
        from omniff.models.vlm import VLMModel

        model_id = controls.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        vlm = self._ensure_model("vlm", VLMModel, model_id=model_id, device="auto")
        result = vlm.infer({"image_path": image_path, "prompt": prompt or "Describe this image."})
        return RunResult(output_text=result["text"], route="IMAGE_CAPTION")

    def _run_audio_to_text(self, audio_path: str, prompt: str | None, controls: dict) -> RunResult:
        from omniff.models.asr import ASRModel

        model_id = controls.get("model_id", "openai/whisper-large-v3")
        asr = self._ensure_model("asr", ASRModel, model_id=model_id, device="auto")
        result = asr.infer({"audio_path": audio_path, "language": controls.get("language")})
        text = result["text"]

        if prompt:
            from omniff.models.llm import LLMModel

            llm_id = controls.get("llm_model_id", "Qwen/Qwen3-4B")
            llm = self._ensure_model("llm", LLMModel, model_id=llm_id, device="auto")
            qa_result = llm.infer({"prompt": f"Transcript:\n{text}\n\nQuestion: {prompt}"})
            return RunResult(
                output_text=qa_result["text"], route="AUDIO_QA",
                metadata={"transcript": text},
            )

        return RunResult(output_text=text, route="AUDIO_TRANSCRIBE_ONLY")

    def _run_image_to_image(self, image_path: str, prompt: str, controls: dict, output: str | None) -> RunResult:
        from omniff.models.image_edit import ImageEditModel

        model_id = controls.get("model_id", "stabilityai/sdxl-turbo")
        editor = self._ensure_model("image_edit", ImageEditModel, model_id=model_id, device="auto")
        output_path = output or "output.png"
        result = editor.infer({
            "image_path": image_path,
            "prompt": prompt,
            "negative_prompt": controls.get("negative_prompt", ""),
            "strength": controls.get("strength", 0.5),
            "seed": controls.get("seed"),
            "output_path": output_path,
        })
        return RunResult(output_path=result["image_path"], route="IMAGE_EDIT")
