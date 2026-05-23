"""Built-in tools for OmniFF agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., str]


def _tool_translate(runtime: Any, text: str, target_language: str = "en") -> str:
    result = runtime.run(
        input=f"translate to {target_language}: {text}",
        controls={"source_language": "auto", "target_language": target_language},
    )
    return result.output_text or ""


def _tool_transcribe(runtime: Any, audio_path: str) -> str:
    result = runtime.run(input=audio_path)
    return result.output_text or ""


def _tool_describe_image(runtime: Any, image_path: str, question: str = "") -> str:
    result = runtime.run(input=image_path, prompt=question or "Describe this image in detail.")
    return result.output_text or ""


def _tool_generate_image(runtime: Any, prompt: str, output_path: str = "output.png") -> str:
    result = runtime.run(input=prompt, output_modality="image", output=output_path)
    return f"Image saved to {result.output_path}" if result.output_path else "Failed"


def _tool_summarize(runtime: Any, text: str) -> str:
    result = runtime.run(input=f"Summarize the following text concisely:\n\n{text}")
    return result.output_text or ""


def _tool_read_document(runtime: Any, file_path: str, question: str = "") -> str:
    result = runtime.run(input=file_path, prompt=question or "Extract key information.")
    return result.output_text or ""


def _tool_generate_code(runtime: Any, task: str, language: str = "python") -> str:
    result = runtime.run(input=f"write code in {language}: {task}")
    return result.output_text or ""


def _tool_web_search(runtime: Any, query: str) -> str:
    """Placeholder — returns search prompt for LLM to answer from knowledge."""
    result = runtime.run(input=f"Answer this question using your knowledge: {query}")
    return result.output_text or ""


def get_builtin_tools() -> list[Tool]:
    return [
        Tool(
            name="translate",
            description="Translate text to a target language using NLLB-200 (200 languages)",
            parameters={"text": "Text to translate", "target_language": "Target language code or name"},
            fn=_tool_translate,
        ),
        Tool(
            name="transcribe",
            description="Transcribe audio file to text using Whisper",
            parameters={"audio_path": "Path to audio file"},
            fn=_tool_transcribe,
        ),
        Tool(
            name="describe_image",
            description="Describe or analyze an image using vision model",
            parameters={"image_path": "Path to image", "question": "Optional question about the image"},
            fn=_tool_describe_image,
        ),
        Tool(
            name="generate_image",
            description="Generate an image from text description",
            parameters={"prompt": "Image description", "output_path": "Output file path"},
            fn=_tool_generate_image,
        ),
        Tool(
            name="summarize",
            description="Summarize a long text into key points",
            parameters={"text": "Text to summarize"},
            fn=_tool_summarize,
        ),
        Tool(
            name="read_document",
            description="Read and extract information from PDF/DOCX/TXT files",
            parameters={"file_path": "Path to document", "question": "What to extract"},
            fn=_tool_read_document,
        ),
        Tool(
            name="generate_code",
            description="Generate code for a given task",
            parameters={"task": "Code task description", "language": "Programming language"},
            fn=_tool_generate_code,
        ),
        Tool(
            name="web_search",
            description="Search for information and answer questions",
            parameters={"query": "Search query"},
            fn=_tool_web_search,
        ),
    ]
