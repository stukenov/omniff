from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omniff.models.registry import ModelRegistry
from omniff.observability import RequestTrace, get_logger
from omniff.reliability import VRAM_ESTIMATES, ModelMutex, TimeoutError, get_free_vram_gb
from omniff.router.keyword_router import KeywordRouter
from omniff.runtime.config import OmniFFConfig
from omniff.runtime.result import RunResult

_log = get_logger("engine")


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
        self._mutex = ModelMutex()
        self._request_count = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> OmniFFRuntime:
        config = OmniFFConfig.load(Path(path))
        return cls(config)

    def _ensure_model(
        self, name: str, model_cls: type, trace: RequestTrace | None = None, **kwargs: Any
    ) -> Any:
        if not self.models.has(name):
            model = model_cls(**kwargs)
            self.models.register(name, model)
        m = self.models.get(name)
        if not m.is_loaded:
            model_id = kwargs.get("model_id", name)

            est = VRAM_ESTIMATES.get(model_id, 0)
            if est > 0:
                free = get_free_vram_gb()
                if 0 < free < est:
                    _log.warning(
                        "Model %s needs ~%.1f GB VRAM, only %.1f GB free",
                        model_id,
                        est,
                        free,
                    )

            ctx = trace.span("load_model", model=model_id) if trace else _null_span()
            try:
                import sys

                sys.stderr.write(f"Loading {model_id}...")
                sys.stderr.flush()
                with ctx:
                    m.load()
                sys.stderr.write(" done\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(" FAILED\n")
                sys.stderr.flush()
                msg = str(e)
                if "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
                    raise RuntimeError(
                        f"Not enough VRAM to load {name} ({model_id}). "
                        f"Try a smaller model or free GPU memory. Original: {msg}"
                    ) from e
                if "does not appear to have" in msg or "not found" in msg.lower():
                    raise RuntimeError(
                        f"Model not found: {model_id}. "
                        f"Check the model ID or run: omniff models pull --model-id <id>"
                    ) from e
                raise
        return m

    def run(
        self,
        input: str,
        prompt: str | None = None,
        output_modality: str | None = None,
        thinking: str = "normal",
        controls: dict[str, Any] | None = None,
        output: str | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        trace = RequestTrace()
        self._request_count += 1
        _log.info("request %s start (count=%d)", trace.request_id, self._request_count)

        controls = controls or {}

        with trace.span("route"):
            input_modality = _detect_input_modality(input)

            if input_modality == "text" and not Path(input).exists():
                if prompt:
                    prompt = f"{input}\n\n{prompt}"
                else:
                    prompt = input
                input_modality = "text"

            output_modality = output_modality or "text"
            route = self.router.route(prompt or "", input_modality, output_modality)

        dispatch = {
            "TEXT_SIMPLE": lambda: self._run_text_to_text(
                prompt or input, route.route_class, thinking, controls, trace
            ),
            "TEXT_NORMAL": lambda: self._run_text_to_text(
                prompt or input, route.route_class, thinking, controls, trace
            ),
            "TEXT_COMPLEX": lambda: self._run_text_to_text(
                prompt or input, route.route_class, thinking, controls, trace
            ),
            "IMAGE_CAPTION": lambda: self._run_image_to_text(input, prompt, controls, trace),
            "AUDIO_TRANSCRIBE_ONLY": lambda: self._run_audio_to_text(
                input, prompt, controls, trace
            ),
            "AUDIO_QA": lambda: self._run_audio_to_text(input, prompt, controls, trace),
            "IMAGE_EDIT": lambda: self._run_image_to_image(
                input, prompt or "", controls, output, trace
            ),
            "TEXT_TO_IMAGE": lambda: self._run_text_to_image(
                prompt or input, controls, output, trace
            ),
            "VIDEO_CAPTION": lambda: self._run_video_to_text(input, prompt, controls, trace),
            "DOCUMENT_READ": lambda: self._run_document_to_text(input, prompt, controls, trace),
            "TEXT_TO_SPEECH": lambda: self._run_text_to_speech(
                prompt or input, controls, output, trace
            ),
            "CODE": lambda: self._run_code(prompt or input, controls, trace),
            "DOCUMENT_TO_DOCUMENT": lambda: self._run_document_to_document(
                input, prompt, controls, output, trace
            ),
        }

        handler = dispatch.get(route.route_class)
        if handler:
            if timeout and timeout > 0:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(handler)
                    try:
                        result = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(
                            f"Pipeline {route.route_class} exceeded {timeout}s timeout"
                        )
            else:
                result = handler()
        else:
            result = RunResult(
                output_text=f"Unsupported route: {route.route_class}", route=route.route_class
            )

        result.metadata = result.metadata or {}
        result.metadata["request_id"] = trace.request_id
        result.metadata["timing"] = trace.timing_breakdown()
        result.metadata["total_ms"] = round(trace.total_ms, 2)

        _log.info("request %s done in %.0fms", trace.request_id, trace.total_ms)
        return result

    def run_stream(
        self,
        input: str,
        prompt: str | None = None,
        thinking: str = "normal",
        controls: dict[str, Any] | None = None,
    ) -> Any:
        from omniff.models.llm import LLMModel

        controls = controls or {}
        input_modality = _detect_input_modality(input)

        if input_modality == "text" and not Path(input).exists():
            if prompt:
                prompt = f"{input}\n\n{prompt}"
            else:
                prompt = input

        model_id = controls.get("model_id", "Qwen/Qwen3-4B")
        llm = self._ensure_model("llm", LLMModel, model_id=model_id, device="auto")
        enable_thinking = thinking not in ("off", "fast")
        self._mutex.acquire("llm")
        try:
            yield from llm.infer_stream({"prompt": prompt or input, "thinking": enable_thinking})
        finally:
            self._mutex.release("llm")

    def _run_text_to_text(
        self, prompt: str, route: str, thinking: str, controls: dict, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.llm import LLMModel

        model_id = controls.get("model_id", "Qwen/Qwen3-4B")
        llm = self._ensure_model("llm", LLMModel, trace=trace, model_id=model_id, device="auto")
        enable_thinking = thinking not in ("off", "fast")
        self._mutex.acquire("llm")
        try:
            with trace.span("infer", model=model_id):
                result = llm.infer({"prompt": prompt, "thinking": enable_thinking})
        finally:
            self._mutex.release("llm")
        return RunResult(output_text=result["text"], route=route)

    def _run_image_to_text(
        self, image_path: str, prompt: str | None, controls: dict, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.vlm import VLMModel

        model_id = controls.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        try:
            vlm = self._ensure_model("vlm", VLMModel, trace=trace, model_id=model_id, device="auto")
        except RuntimeError:
            _log.warning("VLM unavailable, falling back to image metadata")
            return self._fallback_image_metadata(image_path)

        self._mutex.acquire("vlm")
        try:
            with trace.span("infer", model=model_id):
                result = vlm.infer(
                    {"image_path": image_path, "prompt": prompt or "Describe this image."}
                )
        finally:
            self._mutex.release("vlm")
        return RunResult(output_text=result["text"], route="IMAGE_CAPTION")

    def _fallback_image_metadata(self, image_path: str) -> RunResult:
        p = Path(image_path)
        info = f"Image: {p.name}, size: {p.stat().st_size} bytes"
        try:
            from PIL import Image

            img = Image.open(image_path)
            info += f", dimensions: {img.size[0]}x{img.size[1]}, mode: {img.mode}"
        except Exception:
            pass
        return RunResult(
            output_text=info,
            route="IMAGE_CAPTION",
            metadata={"fallback": True},
        )

    def _run_audio_to_text(
        self, audio_path: str, prompt: str | None, controls: dict, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.asr import ASRModel

        model_id = controls.get("model_id", "openai/whisper-large-v3")
        asr = self._ensure_model("asr", ASRModel, trace=trace, model_id=model_id, device="auto")
        self._mutex.acquire("asr")
        try:
            with trace.span("infer", model=model_id):
                result = asr.infer({"audio_path": audio_path, "language": controls.get("language")})
        finally:
            self._mutex.release("asr")
        text = result["text"]

        if prompt:
            from omniff.models.llm import LLMModel

            llm_id = controls.get("llm_model_id", "Qwen/Qwen3-4B")
            llm = self._ensure_model("llm", LLMModel, trace=trace, model_id=llm_id, device="auto")
            self._mutex.acquire("llm")
            try:
                with trace.span("infer_qa", model=llm_id):
                    qa_result = llm.infer({"prompt": f"Transcript:\n{text}\n\nQuestion: {prompt}"})
            finally:
                self._mutex.release("llm")
            return RunResult(
                output_text=qa_result["text"],
                route="AUDIO_QA",
                metadata={"transcript": text},
            )

        return RunResult(output_text=text, route="AUDIO_TRANSCRIBE_ONLY")

    def _run_image_to_image(
        self, image_path: str, prompt: str, controls: dict, output: str | None, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.image_edit import ImageEditModel

        model_id = controls.get("model_id", "stabilityai/sdxl-turbo")
        editor = self._ensure_model(
            "image_edit", ImageEditModel, trace=trace, model_id=model_id, device="auto"
        )
        output_path = output or "output.png"
        self._mutex.acquire("image_edit")
        try:
            with trace.span("infer", model=model_id):
                result = editor.infer(
                    {
                        "image_path": image_path,
                        "prompt": prompt,
                        "negative_prompt": controls.get("negative_prompt", ""),
                        "strength": controls.get("strength", 0.5),
                        "seed": controls.get("seed"),
                        "output_path": output_path,
                    }
                )
        finally:
            self._mutex.release("image_edit")
        return RunResult(output_path=result["image_path"], route="IMAGE_EDIT")

    def _run_text_to_image(
        self, prompt: str, controls: dict, output: str | None, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.text_to_image import TextToImageModel

        model_id = controls.get("model_id", "stabilityai/sdxl-turbo")
        gen = self._ensure_model(
            "text_to_image", TextToImageModel, trace=trace, model_id=model_id, device="auto"
        )
        output_path = output or "output.png"
        self._mutex.acquire("text_to_image")
        try:
            with trace.span("infer", model=model_id):
                result = gen.infer(
                    {
                        "prompt": prompt,
                        "negative_prompt": controls.get("negative_prompt", ""),
                        "seed": controls.get("seed"),
                        "output_path": output_path,
                    }
                )
        finally:
            self._mutex.release("text_to_image")
        return RunResult(output_path=result["image_path"], route="TEXT_TO_IMAGE")

    def _run_video_to_text(
        self, video_path: str, prompt: str | None, controls: dict, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.video_captioner import VideoCaptionerModel

        model_id = controls.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        captioner = self._ensure_model(
            "video_captioner", VideoCaptionerModel, trace=trace, model_id=model_id, device="auto"
        )
        self._mutex.acquire("video_captioner")
        try:
            with trace.span("infer", model=model_id):
                result = captioner.infer(
                    {
                        "video_path": video_path,
                        "prompt": prompt or "Describe this video in detail.",
                    }
                )
        finally:
            self._mutex.release("video_captioner")
        return RunResult(
            output_text=result["text"],
            route="VIDEO_CAPTION",
            metadata={"num_frames": result.get("num_frames", 0)},
        )

    def _run_document_to_text(
        self, doc_path: str, prompt: str | None, controls: dict, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.document_reader import DocumentReaderModel

        llm_id = controls.get("model_id", "Qwen/Qwen3-4B")
        reader = self._ensure_model(
            "document_reader", DocumentReaderModel, trace=trace, llm_model_id=llm_id, device="auto"
        )
        self._mutex.acquire("document_reader")
        try:
            with trace.span("infer", model=llm_id):
                result = reader.infer(
                    {
                        "document_path": doc_path,
                        "prompt": prompt,
                    }
                )
        finally:
            self._mutex.release("document_reader")
        return RunResult(
            output_text=result["text"],
            route="DOCUMENT_READ",
            metadata={"source": result.get("source", "extraction")},
        )

    def _run_text_to_speech(
        self, text: str, controls: dict, output: str | None, trace: RequestTrace
    ) -> RunResult:
        from omniff.models.tts import TTSModel

        model_id = controls.get("model_id", "suno/bark-small")
        tts = self._ensure_model("tts", TTSModel, trace=trace, model_id=model_id, device="auto")
        output_path = output or "output.wav"
        self._mutex.acquire("tts")
        try:
            with trace.span("infer", model=model_id):
                result = tts.infer(
                    {
                        "text": text,
                        "output_path": output_path,
                        "voice_preset": controls.get("voice_preset", "v2/en_speaker_6"),
                    }
                )
        finally:
            self._mutex.release("tts")
        return RunResult(
            output_path=result["audio_path"],
            route="TEXT_TO_SPEECH",
            metadata={"duration_s": result.get("duration_s", 0)},
        )

    def _run_code(self, prompt: str, controls: dict, trace: RequestTrace) -> RunResult:
        from omniff.models.code import CodeModel

        model_id = controls.get("model_id", "Qwen/Qwen3-4B")
        code = self._ensure_model("code", CodeModel, trace=trace, model_id=model_id, device="auto")
        self._mutex.acquire("code")
        try:
            with trace.span("infer", model=model_id):
                result = code.infer({"prompt": prompt})
        finally:
            self._mutex.release("code")
        return RunResult(output_text=result["text"], route="CODE")

    def _run_document_to_document(
        self,
        doc_path: str,
        prompt: str | None,
        controls: dict,
        output: str | None,
        trace: RequestTrace,
    ) -> RunResult:
        from omniff.models.document_reader import DocumentReaderModel
        from omniff.models.pdf_generator import PDFGeneratorModel

        llm_id = controls.get("model_id", "Qwen/Qwen3-4B")
        reader = self._ensure_model(
            "document_reader", DocumentReaderModel, trace=trace, llm_model_id=llm_id, device="auto"
        )
        self._mutex.acquire("document_reader")
        try:
            with trace.span("extract", model=llm_id):
                result = reader.infer(
                    {
                        "document_path": doc_path,
                        "prompt": prompt or "Summarize this document.",
                    }
                )
        finally:
            self._mutex.release("document_reader")

        pdf_gen = self._ensure_model("pdf_generator", PDFGeneratorModel, trace=trace)
        output_path = output or "output.pdf"
        with trace.span("generate_pdf"):
            pdf_result = pdf_gen.infer(
                {
                    "text": result["text"],
                    "title": prompt or "Document Summary",
                    "output_path": output_path,
                }
            )

        return RunResult(
            output_path=pdf_result["pdf_path"],
            route="DOCUMENT_TO_DOCUMENT",
            metadata={"source": result.get("source", "extraction")},
        )


@contextmanager
def _null_span():
    yield None
