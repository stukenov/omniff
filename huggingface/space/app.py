"""
OmniFF -- FFmpeg for AI | HuggingFace Space (ZeroGPU)

All pipelines: text->text, image->text, audio->text, text->image,
image->image, video->text, document->text, code generation
"""

import mimetypes
import os
import re
import tempfile

import gradio as gr
import spaces
import torch

# ---------------------------------------------------------------------------
# Global model holders (loaded on first GPU call)
# ---------------------------------------------------------------------------
_llm = {"model": None, "tokenizer": None}
_vlm = {"model": None, "processor": None}
_asr = {"model": None, "processor": None}
_t2i = {"pipe": None}
_i2i = {"pipe": None}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _load_llm():
    if _llm["model"] is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-4B"
    _llm["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
    _llm["model"] = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )


def _load_vlm():
    if _vlm["model"] is not None:
        return
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    _vlm["processor"] = AutoProcessor.from_pretrained(model_id)
    _vlm["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )


def _load_asr():
    if _asr["model"] is not None:
        return
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = "openai/whisper-large-v3"
    _asr["processor"] = WhisperProcessor.from_pretrained(model_id)
    _asr["model"] = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )


def _load_t2i():
    if _t2i["pipe"] is not None:
        return
    from diffusers import AutoPipelineForText2Image

    _t2i["pipe"] = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")


def _load_i2i():
    if _i2i["pipe"] is not None:
        return
    from diffusers import AutoPipelineForImage2Image

    _i2i["pipe"] = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")


def _llm_generate(messages: list, thinking: str = "off", max_tokens: int = 512) -> str:
    _load_llm()
    model = _llm["model"]
    tokenizer = _llm["tokenizer"]

    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if thinking == "off":
        chat_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    if thinking != "off":
        max_tokens = max(max_tokens, 2048)

    generated = model.generate(**inputs, max_new_tokens=max_tokens)
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_think(response) or "No output generated."


# ---------------------------------------------------------------------------
# 1. Text -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_text(text: str, thinking: str) -> str:
    if not text.strip():
        return "Please enter some text."
    return _llm_generate([{"role": "user", "content": text}], thinking)


# ---------------------------------------------------------------------------
# 2. Image -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_image(image_path: str, prompt: str) -> str:
    if not image_path:
        return "Please upload an image."

    _load_vlm()
    model = _vlm["model"]
    processor = _vlm["processor"]

    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt or "Describe this image in detail."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "No output generated."


# ---------------------------------------------------------------------------
# 3. Audio -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def process_audio(audio_path: str, language: str) -> str:
    if not audio_path:
        return "Please upload an audio file."

    _load_asr()
    model = _asr["model"]
    processor = _asr["processor"]

    import numpy as np
    import soundfile as sf

    audio_data, sr = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample

        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
        sr = 16000

    input_features = processor(
        audio_data, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device=model.device, dtype=model.dtype)

    gen_kwargs = {}
    if language:
        gen_kwargs["language"] = language

    predicted_ids = model.generate(input_features, **gen_kwargs)
    return (
        processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        or "No output generated."
    )


# ---------------------------------------------------------------------------
# 4. Text -> Image
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def generate_image(prompt: str, seed: int):
    if not prompt.strip():
        return None

    _load_t2i()
    pipe = _t2i["pipe"]

    generator = torch.Generator("cuda")
    if seed >= 0:
        generator.manual_seed(int(seed))

    image = pipe(
        prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator
    ).images[0]
    return image


# ---------------------------------------------------------------------------
# 5. Image -> Image
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def edit_image(image_path: str, prompt: str, strength: float):
    if not image_path:
        return None

    _load_i2i()
    pipe = _i2i["pipe"]

    from PIL import Image

    init_image = Image.open(image_path).convert("RGB").resize((512, 512))

    image = pipe(
        prompt=prompt or "enhance this image",
        image=init_image,
        num_inference_steps=4,
        strength=max(0.1, min(1.0, strength)),
        guidance_scale=0.0,
    ).images[0]
    return image


# ---------------------------------------------------------------------------
# 6. Video -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_video(video_path: str, prompt: str) -> str:
    if not video_path:
        return "Please upload a video."

    _load_vlm()
    model = _vlm["model"]
    processor = _vlm["processor"]

    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": prompt or "Describe what happens in this video.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "No output generated."


# ---------------------------------------------------------------------------
# 7. Document -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_document(file_obj, prompt: str) -> str:
    if not file_obj:
        return "Please upload a document."

    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    text_content = ""

    if ext == "pdf":
        try:
            import fitz

            doc = fitz.open(file_path)
            text_content = "\n".join(page.get_text() for page in doc)
            doc.close()
        except ImportError:
            return "PDF support requires PyMuPDF (fitz). Upload a .txt file instead."
    elif ext in ("txt", "md", "csv", "json", "xml", "html", "log"):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()
    else:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()

    if not text_content.strip():
        return "Could not extract text from document."

    if len(text_content) > 8000:
        text_content = text_content[:8000] + "\n\n[... truncated ...]"

    user_prompt = prompt or "Summarize this document."
    messages = [{"role": "user", "content": f"{user_prompt}\n\n---\n\n{text_content}"}]
    return _llm_generate(messages, thinking="off", max_tokens=1024)


# ---------------------------------------------------------------------------
# 8. Code Generation
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def generate_code(task: str, language: str) -> str:
    if not task.strip():
        return "Please describe what code to generate."

    lang_hint = f" in {language}" if language else ""
    messages = [
        {
            "role": "system",
            "content": f"You are an expert programmer. Write clean, well-structured code{lang_hint}. "
            "Return ONLY the code, no explanations unless asked.",
        },
        {"role": "user", "content": task},
    ]
    return _llm_generate(messages, thinking="off", max_tokens=2048)


# ---------------------------------------------------------------------------
# Universal Chat — auto-routing multimodal pipeline
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv"}
_DOC_EXTS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".log", ".pdf"}

_IMAGE_GEN_PATTERNS = re.compile(
    r"\b(generate|create|draw|paint|make|design|render|imagine|visualize)\b.*\b(image|picture|photo|illustration|art|painting|drawing|icon|logo)\b",
    re.IGNORECASE,
)
_IMAGE_GEN_PATTERNS_REV = re.compile(
    r"\b(image|picture|photo|illustration|art|painting|drawing|icon|logo)\b.*\b(of|for|with|showing|depicting)\b",
    re.IGNORECASE,
)
_CODE_PATTERNS = re.compile(
    r"\b(write|code|implement|create|build|generate)\b.*\b(function|class|script|program|code|api|endpoint|module|component|algorithm)\b",
    re.IGNORECASE,
)
_TRANSLATE_PATTERNS = re.compile(
    r"\b(translate|переведи|перевод|translation|переведите|to english|to russian|to chinese|на английский|на русский|на казахский)\b",
    re.IGNORECASE,
)
_DUB_PATTERNS = re.compile(
    r"\b(dub|dubbing|дубляж|озвуч|переозвуч|voice\s*over|voiceover)\b",
    re.IGNORECASE,
)
_AGENT_PATTERNS = re.compile(
    r"\b(agent|step by step|investigate|research and|plan and execute|multi-step|агент|пошагово|исследуй|разберись)\b",
    re.IGNORECASE,
)


def _classify_file(file_path: str) -> str:
    """Classify a file into a modality category."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _DOC_EXTS:
        return "document"
    # Fallback: try mime type
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
    return "document"


def _detect_intent(text: str) -> str:
    """Detect if text-only input wants image generation, code, translation, or agent."""
    if _IMAGE_GEN_PATTERNS.search(text) or _IMAGE_GEN_PATTERNS_REV.search(text):
        return "text_to_image"
    if _CODE_PATTERNS.search(text):
        return "code"
    if _TRANSLATE_PATTERNS.search(text):
        return "translate"
    if _AGENT_PATTERNS.search(text):
        return "agent"
    return "text"


@spaces.GPU(duration=120)
def universal_chat(message: dict, history: list) -> dict:
    """
    Universal multimodal handler.

    Accepts a dict with 'text' and 'files' keys.
    Routes to the appropriate pipeline based on input type.
    Returns a dict for the chatbot to display.
    """
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    # Normalize file paths
    file_paths = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(f)
        elif isinstance(f, dict) and "path" in f:
            file_paths.append(f["path"])
        elif hasattr(f, "name"):
            file_paths.append(f.name)

    # --- Route based on inputs ---

    # Case 1: File(s) provided — classify by first file
    if file_paths:
        first_file = file_paths[0]
        modality = _classify_file(first_file)

        if modality == "image":
            result = process_image(first_file, text or "Describe this image in detail.")
            return {"role": "assistant", "content": result}

        elif modality == "audio":
            if text and _DUB_PATTERNS.search(text):
                # Audio dubbing: ASR → Translate → TTS
                transcript = process_audio(first_file, "")
                target_lang = "English"
                for lang in ["english", "russian", "chinese", "kazakh", "русский", "английский", "казахский"]:
                    if lang in text.lower():
                        target_lang = lang.capitalize()
                        break
                tr_prompt = f"Translate the following text to {target_lang}. Return ONLY the translation.\n\n{transcript}"
                translated = process_text(tr_prompt, "off")
                return {
                    "role": "assistant",
                    "content": f"**Original transcript:**\n{transcript}\n\n**Dubbed ({target_lang}):**\n{translated}\n\n_(Full audio dubbing with TTS requires local GPU runtime)_",
                }
            elif text and _TRANSLATE_PATTERNS.search(text):
                # Audio translation: ASR → Translate
                transcript = process_audio(first_file, "")
                target_lang = "English"
                for lang in ["english", "russian", "chinese", "kazakh", "русский", "английский", "казахский"]:
                    if lang in text.lower():
                        target_lang = lang.capitalize()
                        break
                tr_prompt = f"Translate the following text to {target_lang}. Return ONLY the translation.\n\n{transcript}"
                translated = process_text(tr_prompt, "off")
                return {
                    "role": "assistant",
                    "content": f"**Original transcript:**\n{transcript}\n\n**Translation ({target_lang}):**\n{translated}",
                }
            result = process_audio(first_file, "")
            if text:
                full_prompt = f"The user uploaded an audio file. Here is the transcription:\n\n{result}\n\nUser's question: {text}"
                answer = process_text(full_prompt, "off")
                return {"role": "assistant", "content": f"**Transcription:**\n{result}\n\n**Answer:**\n{answer}"}
            return {"role": "assistant", "content": f"**Transcription:**\n{result}"}

        elif modality == "video":
            if text and _DUB_PATTERNS.search(text):
                result = process_video(first_file, "Transcribe all speech in this video word by word.")
                target_lang = "English"
                for lang in ["english", "russian", "chinese", "kazakh", "русский", "английский", "казахский"]:
                    if lang in text.lower():
                        target_lang = lang.capitalize()
                        break
                tr_prompt = f"Translate the following text to {target_lang}. Return ONLY the translation.\n\n{result}"
                translated = process_text(tr_prompt, "off")
                return {
                    "role": "assistant",
                    "content": f"**Video transcript:**\n{result}\n\n**Dubbed ({target_lang}):**\n{translated}\n\n_(Full video dubbing with audio muxing requires local GPU runtime)_",
                }
            result = process_video(first_file, text or "Describe what happens in this video.")
            return {"role": "assistant", "content": result}

        elif modality == "document":
            result = process_document(first_file, text or "Summarize this document.")
            return {"role": "assistant", "content": result}

    # Case 2: Text only
    if not text:
        return {
            "role": "assistant",
            "content": "Send me a message or upload a file. I can understand text, images, audio, video, and documents.",
        }

    intent = _detect_intent(text)

    if intent == "text_to_image":
        img = generate_image(text, -1)
        if img is not None:
            # Save to temp file for chatbot display
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            return {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Generated from: *{text}*"},
                    {"type": "image", "image": {"path": tmp.name}},
                ],
            }
        return {"role": "assistant", "content": "Could not generate the image. Please try a different prompt."}

    if intent == "translate":
        target_lang = "English"
        for lang in ["english", "russian", "chinese", "kazakh", "french", "german", "spanish",
                      "русский", "английский", "казахский", "французский", "немецкий"]:
            if lang in text.lower():
                target_lang = lang.capitalize()
                break
        tr_prompt = f"Translate the following text to {target_lang}. Return ONLY the translation.\n\n{text}"
        translated = process_text(tr_prompt, "off")
        return {"role": "assistant", "content": f"**Translation ({target_lang}):**\n{translated}"}

    if intent == "code":
        result = generate_code(text, "")
        return {"role": "assistant", "content": f"```\n{result}\n```"}

    if intent == "agent":
        steps = []
        steps.append(f"**Task:** {text}\n")
        thinking_prompt = f"Break this task into steps and solve it:\n\n{text}"
        plan = process_text(thinking_prompt, "off")
        steps.append(f"**Plan:**\n{plan}\n")
        execution_prompt = f"Execute this plan and provide the final answer:\n\nTask: {text}\nPlan: {plan}"
        answer = process_text(execution_prompt, "off")
        steps.append(f"**Result:**\n{answer}")
        return {"role": "assistant", "content": "\n".join(steps)}

    # Default: text-to-text
    result = process_text(text, "off")
    return {"role": "assistant", "content": result}


# ---------------------------------------------------------------------------
# Warm-up (pre-download models on CPU at startup)
# ---------------------------------------------------------------------------
def _warmup():
    import threading

    def _do():
        try:
            from huggingface_hub import snapshot_download

            for model_id in [
                "Qwen/Qwen3-4B",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "openai/whisper-large-v3",
                "stabilityai/sdxl-turbo",
            ]:
                snapshot_download(model_id)
        except Exception:
            pass

    threading.Thread(target=_do, daemon=True).start()


_warmup()

# ---------------------------------------------------------------------------
# Routing labels — human-readable pipeline names for loading states
# ---------------------------------------------------------------------------
_ROUTE_LABELS = {
    "image": "Vision model (Qwen2.5-VL)",
    "audio": "Speech model (Whisper)",
    "video": "Video model (Qwen2.5-VL)",
    "document": "Language model (Qwen3)",
    "text": "Language model (Qwen3)",
    "text_to_image": "Image generator (SDXL Turbo)",
    "code": "Language model (Qwen3)",
}


# ---------------------------------------------------------------------------
# Design System -- Round 2: Apple HIG-inspired, production polish
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
/* ==================================================================
   BASE — Typography, Container
   ================================================================== */
:root {
    --omiff-blue: #0071e3;
    --omiff-blue-hover: #0077ed;
    --omiff-blue-bg: rgba(0, 113, 227, 0.08);
    --omiff-text-primary: #1d1d1f;
    --omiff-text-secondary: #86868b;
    --omiff-text-tertiary: #aeaeb2;
    --omiff-border: rgba(0, 0, 0, 0.08);
    --omiff-border-subtle: rgba(0, 0, 0, 0.04);
    --omiff-surface: rgba(0, 0, 0, 0.015);
    --omiff-green: #34c759;
    --omiff-orange: #ff9500;
    --omiff-purple: #af52de;
    --omiff-red: #ff3b30;
    --omiff-teal: #32ade6;
    --omiff-violet: #bf5af2;
    --omiff-radius-sm: 8px;
    --omiff-radius-md: 12px;
    --omiff-radius-lg: 16px;
    --omiff-radius-xl: 20px;
}

.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}

/* ==================================================================
   DARK MODE
   ================================================================== */
@media (prefers-color-scheme: dark) {
    :root {
        --omiff-text-primary: #f5f5f7;
        --omiff-text-secondary: #a1a1a6;
        --omiff-text-tertiary: #6e6e73;
        --omiff-border: rgba(255, 255, 255, 0.1);
        --omiff-border-subtle: rgba(255, 255, 255, 0.05);
        --omiff-surface: rgba(255, 255, 255, 0.04);
    }
}
.dark {
    --omiff-text-primary: #f5f5f7;
    --omiff-text-secondary: #a1a1a6;
    --omiff-text-tertiary: #6e6e73;
    --omiff-border: rgba(255, 255, 255, 0.1);
    --omiff-border-subtle: rgba(255, 255, 255, 0.05);
    --omiff-surface: rgba(255, 255, 255, 0.04);
}
.dark .omiff-header h1 { color: var(--omiff-text-primary) !important; }
.dark .omiff-header p { color: var(--omiff-text-secondary) !important; }
.dark .omiff-header .omiff-badge {
    background: rgba(191, 90, 242, 0.15) !important;
}
.dark .tab-nav {
    border-bottom-color: var(--omiff-border) !important;
}
.dark .tab-nav button {
    color: var(--omiff-text-secondary) !important;
}
.dark .tab-nav button:hover {
    color: var(--omiff-text-primary) !important;
}
.dark .tab-nav button.selected {
    color: #409cff !important;
    border-bottom-color: #409cff !important;
}
.dark .omiff-capabilities span {
    background: rgba(255, 255, 255, 0.06) !important;
    color: var(--omiff-text-secondary) !important;
}
.dark .omiff-chat-hint { color: var(--omiff-text-tertiary) !important; }
.dark .omiff-section-label { color: var(--omiff-text-secondary) !important; }
.dark .omiff-footer {
    border-top-color: var(--omiff-border) !important;
}
.dark .omiff-footer p { color: var(--omiff-text-secondary) !important; }
.dark .omiff-footer a { color: #409cff !important; }
.dark label span { color: var(--omiff-text-primary) !important; }
.dark .output-textbox textarea {
    background: var(--omiff-surface) !important;
    border-color: var(--omiff-border) !important;
}
.dark textarea, .dark input[type="text"] {
    border-color: var(--omiff-border) !important;
}
.dark textarea:focus, .dark input[type="text"]:focus {
    border-color: #409cff !important;
    box-shadow: 0 0 0 3px rgba(64, 156, 255, 0.15) !important;
}
.dark .accordion {
    border-color: var(--omiff-border) !important;
}
.dark .accordion > .label-wrap {
    color: var(--omiff-text-secondary) !important;
    background: var(--omiff-surface) !important;
}
.dark button.secondary {
    border-color: var(--omiff-border) !important;
}
.dark .omiff-status-pill {
    background: rgba(255, 255, 255, 0.06) !important;
    color: var(--omiff-text-secondary) !important;
}

/* ==================================================================
   HEADER — Concise, confident identity
   ================================================================== */
.omiff-header {
    text-align: center;
    padding: 36px 16px 28px 16px;
    border-bottom: 1px solid var(--omiff-border);
    margin-bottom: 4px;
}
.omiff-header h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    letter-spacing: -0.8px !important;
    color: var(--omiff-text-primary) !important;
    margin-bottom: 6px !important;
    line-height: 1.15 !important;
}
.omiff-header p {
    font-size: 15px !important;
    color: var(--omiff-text-secondary) !important;
    font-weight: 400 !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}
.omiff-header .omiff-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    color: var(--omiff-violet);
    background: rgba(191, 90, 242, 0.08);
    padding: 3px 10px;
    border-radius: 100px;
    margin-top: 10px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ==================================================================
   TABS — 5 tabs: Chat (hero), Understand, Create, Transcribe, Code
   ================================================================== */
.tab-nav {
    border-bottom: 1px solid var(--omiff-border) !important;
    gap: 0 !important;
    padding: 0 16px !important;
    justify-content: center !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    color: var(--omiff-text-secondary) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: color 0.2s ease, border-color 0.2s ease !important;
    letter-spacing: 0.1px !important;
    white-space: nowrap !important;
}
.tab-nav button:hover {
    color: var(--omiff-text-primary) !important;
}
.tab-nav button.selected {
    color: var(--omiff-blue) !important;
    border-bottom: 2px solid var(--omiff-blue) !important;
    font-weight: 600 !important;
}
/* Chat tab always bold — it is the primary entry point */
.tab-nav button:first-child {
    font-weight: 600 !important;
    letter-spacing: 0 !important;
}

/* ==================================================================
   BUTTONS
   ================================================================== */
.primary {
    background: var(--omiff-blue) !important;
    border: none !important;
    border-radius: var(--omiff-radius-md) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    letter-spacing: 0.1px !important;
}
.primary:hover {
    background: var(--omiff-blue-hover) !important;
    transform: translateY(-1px) !important;
}
.primary:active {
    transform: translateY(0) !important;
}
button.secondary {
    border-radius: var(--omiff-radius-md) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: 1px solid var(--omiff-border) !important;
}

/* ==================================================================
   INPUTS — consistent radii, focus rings
   ================================================================== */
textarea, input[type="text"], .wrap input {
    border-radius: var(--omiff-radius-md) !important;
    border: 1px solid var(--omiff-border) !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--omiff-blue) !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.1) !important;
    outline: none !important;
}

/* ==================================================================
   LABELS
   ================================================================== */
label span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--omiff-text-primary) !important;
    letter-spacing: 0.1px !important;
}

/* ==================================================================
   ACCORDION — progressive disclosure
   ================================================================== */
.accordion {
    border: 1px solid var(--omiff-border-subtle) !important;
    border-radius: var(--omiff-radius-md) !important;
    margin-top: 8px !important;
    overflow: hidden !important;
}
.accordion > .label-wrap {
    padding: 10px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--omiff-text-secondary) !important;
    background: var(--omiff-surface) !important;
}

/* ==================================================================
   OUTPUT AREAS
   ================================================================== */
.output-textbox textarea {
    background: var(--omiff-surface) !important;
    border: 1px solid var(--omiff-border-subtle) !important;
}

/* ==================================================================
   CHATBOT — the hero experience
   ================================================================== */
.chatbot-wrap {
    border-radius: var(--omiff-radius-lg) !important;
    border: 1px solid var(--omiff-border) !important;
}

/* Chat subtitle / hint text */
.omiff-chat-hint {
    font-size: 13px;
    color: var(--omiff-text-tertiary);
    text-align: center;
    padding: 4px 0 2px 0;
    line-height: 1.5;
}

/* Capabilities row — replaces the old pipeline tags */
.omiff-capabilities {
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    padding: 6px 0 4px 0;
}
.omiff-capabilities span {
    font-size: 11px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 100px;
    background: rgba(0, 0, 0, 0.035);
    color: var(--omiff-text-secondary);
    letter-spacing: 0.2px;
}

/* Status pill shown during processing */
.omiff-status-pill {
    display: inline-block;
    font-size: 12px;
    font-weight: 500;
    color: var(--omiff-text-secondary);
    background: var(--omiff-surface);
    padding: 6px 14px;
    border-radius: 100px;
    margin: 8px 0;
    animation: omiff-pulse 1.5s ease-in-out infinite;
}
@keyframes omiff-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ==================================================================
   SECTION LABELS — specialized tabs
   ================================================================== */
.omiff-section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--omiff-text-secondary);
    padding: 8px 0 4px 4px;
}

/* ==================================================================
   FOOTER
   ================================================================== */
.omiff-footer {
    text-align: center;
    padding: 28px 16px;
    border-top: 1px solid var(--omiff-border);
    margin-top: 16px;
}
.omiff-footer p {
    font-size: 12px !important;
    color: var(--omiff-text-secondary) !important;
    line-height: 1.6 !important;
}
.omiff-footer a {
    color: var(--omiff-blue) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}
.omiff-footer a:hover {
    text-decoration: underline !important;
}

/* ==================================================================
   SPACING — 8pt base grid
   ================================================================== */
.block {
    margin-bottom: 4px !important;
}
.form {
    gap: 12px !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio UI — Round 2: Apple HIG Production Design
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="OmniFF -- FFmpeg for AI",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css=_CUSTOM_CSS,
) as demo:

    # ---- Header ----
    gr.HTML(
        """
        <div class="omiff-header">
            <h1>OmniFF</h1>
            <p>FFmpeg for AI &mdash; one interface for every modality</p>
            <span class="omiff-badge">ZeroGPU</span>
        </div>
        """
    )

    # ==================================================================
    # TAB 1: Chat — the hero. Everything starts here.
    # ==================================================================
    with gr.Tab("Chat", id="chat"):
        gr.HTML(
            '<p class="omiff-chat-hint">'
            "Send text, images, audio, video, or documents. OmniFF routes automatically."
            "</p>"
        )

        chatbot = gr.Chatbot(
            height=520,
            type="messages",
            show_label=False,
            avatar_images=(
                None,
                "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
            ),
            placeholder=(
                '<div style="text-align: center; padding: 40px 20px; opacity: 0.6;">'
                '<p style="font-size: 18px; font-weight: 600; margin-bottom: 4px;">What can I help with?</p>'
                '<p style="font-size: 13px;">Drop a file or type a message to begin.</p>'
                "</div>"
            ),
        )

        chat_input = gr.MultimodalTextbox(
            placeholder="Message OmniFF...",
            show_label=False,
            file_count="single",
            sources=["upload", "microphone"],
            submit_btn=True,
            stop_btn=True,
        )

        # Capabilities row — subdued, informational, not distracting
        gr.HTML(
            '<div class="omiff-capabilities">'
            "<span>Text</span>"
            "<span>Vision</span>"
            "<span>Audio</span>"
            "<span>Video</span>"
            "<span>Documents</span>"
            "<span>Code</span>"
            "<span>Image Gen</span>"
            "</div>"
        )

        def respond(message, history):
            """
            Universal chat handler with proper Gradio message format.

            Key fixes from round 1:
            - Files use {"path": ...} dict format, not gr.Image()/gr.Audio() objects
            - Adds routing status messages during processing
            - Graceful error handling wraps all pipeline calls
            """
            if not message or (
                not message.get("text", "").strip() and not message.get("files")
            ):
                return history, gr.MultimodalTextbox(value=None)

            text = message.get("text", "")
            files = message.get("files", [])

            # --- Build user messages ---
            if files:
                for f in files:
                    fp = (
                        f
                        if isinstance(f, str)
                        else (
                            f.get("path", "")
                            if isinstance(f, dict)
                            else getattr(f, "name", str(f))
                        )
                    )
                    # Use {"path": ...} dict format for proper rendering
                    history.append({"role": "user", "content": {"path": fp}})
                if text:
                    history.append({"role": "user", "content": text})
            else:
                history.append({"role": "user", "content": text})

            # --- Determine route for status message ---
            file_paths = []
            for f in files:
                if isinstance(f, str):
                    file_paths.append(f)
                elif isinstance(f, dict) and "path" in f:
                    file_paths.append(f["path"])
                elif hasattr(f, "name"):
                    file_paths.append(f.name)

            if file_paths:
                modality = _classify_file(file_paths[0])
                route_label = _ROUTE_LABELS.get(modality, "model")
            elif text:
                intent = _detect_intent(text)
                route_label = _ROUTE_LABELS.get(intent, "Language model (Qwen3)")
            else:
                route_label = "model"

            # --- Process with error handling ---
            try:
                response = universal_chat(message, history)
                content = response.get("content", "")

                if isinstance(content, list):
                    # Compound response (e.g. text + generated image)
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                history.append(
                                    {"role": "assistant", "content": item["text"]}
                                )
                            elif item.get("type") == "image":
                                img_path = item.get("image", {}).get("path", "")
                                history.append(
                                    {
                                        "role": "assistant",
                                        "content": {"path": img_path},
                                    }
                                )
                        else:
                            history.append(
                                {"role": "assistant", "content": str(item)}
                            )
                else:
                    history.append({"role": "assistant", "content": content})

            except Exception as exc:
                # Graceful error — never show a raw traceback
                error_msg = str(exc)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Something went wrong while processing your request.\n\n"
                            f"**Error:** {error_msg}\n\n"
                            f"Try again, or use a specialized tab for more control."
                        ),
                    }
                )

            return history, gr.MultimodalTextbox(value=None)

        chat_input.submit(respond, [chat_input, chatbot], [chatbot, chat_input])

    # ==================================================================
    # TAB 2: Understand — Vision + Video (both use VLM)
    # ==================================================================
    with gr.Tab("Understand", id="understand"):
        gr.HTML('<p class="omiff-section-label">Visual Understanding</p>')

        with gr.Tabs():
            # -- Image sub-tab --
            with gr.Tab("Image"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="filepath",
                            label="Image",
                            height=300,
                        )
                        img_prompt = gr.Textbox(
                            label="Question",
                            value="Describe this image in detail.",
                            lines=2,
                        )
                        img_btn = gr.Button(
                            "Analyze", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        img_output = gr.Textbox(
                            label="Analysis",
                            lines=12,
                            show_copy_button=True,
                            elem_classes=["output-textbox"],
                        )
                img_btn.click(
                    process_image, [img_input, img_prompt], img_output
                )

            # -- Video sub-tab --
            with gr.Tab("Video"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Video", height=300)
                        vid_prompt = gr.Textbox(
                            label="Question",
                            value="Describe what happens in this video.",
                            lines=2,
                        )
                        vid_btn = gr.Button(
                            "Analyze", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        vid_output = gr.Textbox(
                            label="Analysis",
                            lines=12,
                            show_copy_button=True,
                            elem_classes=["output-textbox"],
                        )
                vid_btn.click(
                    process_video, [vid_input, vid_prompt], vid_output
                )

    # ==================================================================
    # TAB 3: Create — Text-to-Image + Image-to-Image (both use SDXL)
    # ==================================================================
    with gr.Tab("Create", id="create"):
        gr.HTML('<p class="omiff-section-label">Image Creation</p>')

        with gr.Tabs():
            # -- Generate sub-tab --
            with gr.Tab("Generate"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gen_prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            placeholder="A cabin in the mountains at golden hour, cinematic lighting...",
                        )
                        with gr.Accordion("Options", open=False):
                            gen_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                info="-1 for random",
                            )
                        gen_btn = gr.Button(
                            "Generate", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        gen_output = gr.Image(label="Result", height=420)
                gen_btn.click(
                    generate_image, [gen_prompt, gen_seed], gen_output
                )

            # -- Transform sub-tab --
            with gr.Tab("Transform"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        i2i_input = gr.Image(
                            type="filepath",
                            label="Source Image",
                            height=260,
                        )
                        i2i_prompt = gr.Textbox(
                            label="Style Prompt",
                            value="Make it look like a watercolor painting",
                            lines=2,
                        )
                        with gr.Accordion("Options", open=False):
                            i2i_strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Strength",
                                info="Lower = preserve original. Higher = creative freedom.",
                            )
                        i2i_btn = gr.Button(
                            "Transform", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        i2i_output = gr.Image(label="Result", height=420)
                i2i_btn.click(
                    edit_image,
                    [i2i_input, i2i_prompt, i2i_strength],
                    i2i_output,
                )

    # ==================================================================
    # TAB 4: Transcribe — Audio/Speech
    # ==================================================================
    with gr.Tab("Transcribe", id="transcribe"):
        gr.HTML('<p class="omiff-section-label">Speech Recognition</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                aud_input = gr.Audio(
                    type="filepath",
                    label="Audio",
                    sources=["upload", "microphone"],
                )
                with gr.Accordion("Options", open=False):
                    aud_lang = gr.Dropdown(
                        ["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"],
                        value="",
                        label="Language",
                        info="Leave empty for auto-detection",
                    )
                aud_btn = gr.Button("Transcribe", variant="primary", size="lg")
            with gr.Column(scale=1):
                aud_output = gr.Textbox(
                    label="Transcription",
                    lines=12,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

    # ==================================================================
    # TAB 5: Tools — Text, Documents, Code
    # ==================================================================
    with gr.Tab("Tools", id="tools"):
        gr.HTML('<p class="omiff-section-label">Text Processing</p>')

        with gr.Tabs():
            # -- Text generation --
            with gr.Tab("Text"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        txt_input = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            placeholder="Ask anything...",
                        )
                        with gr.Accordion("Options", open=False):
                            txt_thinking = gr.Radio(
                                ["off", "normal"],
                                value="off",
                                label="Reasoning mode",
                                info="Extended reasoning for complex questions",
                            )
                        txt_btn = gr.Button(
                            "Generate", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        txt_output = gr.Textbox(
                            label="Response",
                            lines=12,
                            show_copy_button=True,
                            elem_classes=["output-textbox"],
                        )
                txt_btn.click(
                    process_text, [txt_input, txt_thinking], txt_output
                )

            # -- Documents --
            with gr.Tab("Documents"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        doc_input = gr.File(
                            label="Document",
                            file_types=[
                                ".txt", ".md", ".csv", ".json",
                                ".xml", ".html", ".log", ".pdf",
                            ],
                        )
                        doc_prompt = gr.Textbox(
                            label="Instruction",
                            value="Summarize this document.",
                            lines=2,
                        )
                        doc_btn = gr.Button(
                            "Process", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        doc_output = gr.Textbox(
                            label="Result",
                            lines=12,
                            show_copy_button=True,
                            elem_classes=["output-textbox"],
                        )
                doc_btn.click(
                    process_document, [doc_input, doc_prompt], doc_output
                )

            # -- Code --
            with gr.Tab("Code"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        code_task = gr.Textbox(
                            label="Task",
                            lines=4,
                            placeholder="Write a merge sort function...",
                        )
                        with gr.Accordion("Options", open=False):
                            code_lang = gr.Dropdown(
                                [
                                    "Python", "JavaScript", "TypeScript",
                                    "Rust", "Go", "Java", "C++", "",
                                ],
                                value="Python",
                                label="Language",
                            )
                        code_btn = gr.Button(
                            "Generate", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        code_output = gr.Code(
                            label="Output",
                            language="python",
                            lines=20,
                        )
                code_btn.click(
                    generate_code, [code_task, code_lang], code_output
                )

    # ---- Footer ----
    gr.HTML(
        """
        <div class="omiff-footer">
            <p>
                <a href="https://github.com/stukenov/omniff" target="_blank">GitHub</a>
                &nbsp;&middot;&nbsp;
                <a href="https://huggingface.co/stukenov/omniff" target="_blank">HuggingFace</a>
                &nbsp;&middot;&nbsp;
                Built by <a href="https://github.com/stukenov" target="_blank">Saken Tukenov</a>
            </p>
            <p style="margin-top: 6px; font-size: 11px !important;">
                8 pipelines &middot; Qwen3 &middot; Qwen2.5-VL &middot; Whisper &middot; SDXL Turbo
            </p>
        </div>
        """
    )

demo.launch(show_error=True)
