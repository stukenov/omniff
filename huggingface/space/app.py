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
    """Detect if text-only input wants image generation or code."""
    if _IMAGE_GEN_PATTERNS.search(text) or _IMAGE_GEN_PATTERNS_REV.search(text):
        return "text_to_image"
    if _CODE_PATTERNS.search(text):
        return "code"
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
            result = process_audio(first_file, "")
            if text:
                # User asked a question about audio — transcribe then answer
                full_prompt = f"The user uploaded an audio file. Here is the transcription:\n\n{result}\n\nUser's question: {text}"
                answer = process_text(full_prompt, "off")
                return {"role": "assistant", "content": f"**Transcription:**\n{result}\n\n**Answer:**\n{answer}"}
            return {"role": "assistant", "content": f"**Transcription:**\n{result}"}

        elif modality == "video":
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

    if intent == "code":
        result = generate_code(text, "")
        return {"role": "assistant", "content": f"```\n{result}\n```"}

    # Default: text-to-text
    result = process_text(text, "off")
    return {"role": "assistant", "content": result}


def universal_chat_wrapper(message: dict, history: list):
    """Wrapper that formats the response for gr.Chatbot."""
    response = universal_chat(message, history)
    content = response.get("content", "")

    # Handle compound content (text + image)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item["text"])
                elif item.get("type") == "image":
                    img_path = item.get("image", {}).get("path", "")
                    parts.append(gr.Image(img_path))
            else:
                parts.append(str(item))
        # For Gradio chatbot, return image path as file
        text_parts = []
        file_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "image":
                    file_parts.append(item["image"]["path"])
        combined = "\n".join(text_parts)
        if file_parts:
            return gr.ChatMessage(
                role="assistant",
                content=combined,
                metadata={"files": file_parts},
            )
        return combined

    return content


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
# Design System — Apple HIG-inspired custom CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
/* ---- Typography & Base ---- */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
}

/* ---- Header ---- */
.omiff-header {
    text-align: center;
    padding: 32px 16px 24px 16px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    margin-bottom: 8px;
}
.omiff-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    color: #1d1d1f !important;
    margin-bottom: 4px !important;
    line-height: 1.2 !important;
}
.omiff-header p {
    font-size: 15px !important;
    color: #86868b !important;
    font-weight: 400 !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}
.omiff-header .omiff-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: #bf5af2;
    background: rgba(191, 90, 242, 0.08);
    padding: 3px 10px;
    border-radius: 12px;
    margin-top: 8px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ---- Tabs ---- */
.tab-nav {
    border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
    gap: 0 !important;
    padding: 0 8px !important;
    justify-content: center !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    color: #86868b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: color 0.2s ease, border-color 0.2s ease !important;
    letter-spacing: 0.1px !important;
}
.tab-nav button:hover {
    color: #1d1d1f !important;
}
.tab-nav button.selected {
    color: #0071e3 !important;
    border-bottom: 2px solid #0071e3 !important;
    font-weight: 600 !important;
}

/* ---- Universal Chat Tab Emphasis ---- */
.tab-nav button:first-child {
    font-weight: 600 !important;
}

/* ---- Primary Buttons ---- */
.primary {
    background: #0071e3 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    letter-spacing: 0.1px !important;
}
.primary:hover {
    background: #0077ed !important;
    transform: translateY(-1px) !important;
}
.primary:active {
    transform: translateY(0) !important;
}

/* ---- Secondary Buttons ---- */
button.secondary {
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
}

/* ---- Input Fields ---- */
textarea, input[type="text"], .wrap input {
    border-radius: 10px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.12) !important;
    outline: none !important;
}

/* ---- Labels ---- */
label span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
    letter-spacing: 0.1px !important;
}

/* ---- Accordion (Progressive Disclosure) ---- */
.accordion {
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
    overflow: hidden !important;
}
.accordion > .label-wrap {
    padding: 10px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #86868b !important;
    background: rgba(0, 0, 0, 0.02) !important;
}

/* ---- Output Areas ---- */
.output-textbox textarea {
    background: rgba(0, 0, 0, 0.015) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
}

/* ---- Chatbot ---- */
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
}
.message-wrap {
    padding: 12px 16px !important;
}

/* ---- Section Dividers ---- */
.omiff-section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #86868b;
    padding: 12px 0 4px 4px;
}

/* ---- Footer ---- */
.omiff-footer {
    text-align: center;
    padding: 24px 16px;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
    margin-top: 16px;
}
.omiff-footer p {
    font-size: 12px !important;
    color: #86868b !important;
    line-height: 1.6 !important;
}
.omiff-footer a {
    color: #0071e3 !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}
.omiff-footer a:hover {
    text-decoration: underline !important;
}

/* ---- Pipeline Tag ---- */
.omiff-pipeline-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 8px;
    margin-right: 4px;
    margin-bottom: 4px;
}
.tag-text { background: rgba(0, 113, 227, 0.08); color: #0071e3; }
.tag-image { background: rgba(52, 199, 89, 0.08); color: #34c759; }
.tag-audio { background: rgba(255, 149, 0, 0.08); color: #ff9500; }
.tag-video { background: rgba(175, 82, 222, 0.08); color: #af52de; }
.tag-doc { background: rgba(255, 59, 48, 0.08); color: #ff3b30; }
.tag-code { background: rgba(90, 200, 250, 0.08); color: #32ade6; }

/* ---- Responsive spacing ---- */
.block {
    margin-bottom: 4px !important;
}
.form {
    gap: 12px !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio UI — Apple HIG Redesign
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
            <p>FFmpeg for AI &mdash; a universal multimodal runtime</p>
            <span class="omiff-badge">ZeroGPU</span>
        </div>
        """
    )

    # ==================================================================
    # TAB: Universal Chat (default, first)
    # ==================================================================
    with gr.Tab("Chat", id="chat"):
        gr.HTML(
            '<p style="font-size: 13px; color: #86868b; text-align: center; padding: 8px 0 4px 0;">'
            "Send anything &mdash; text, images, audio, video, documents. OmniFF routes to the right model automatically."
            "</p>"
        )

        chatbot = gr.Chatbot(
            label="OmniFF",
            height=480,
            type="messages",
            show_label=False,
            avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"),
            placeholder="Your conversation will appear here...",
        )

        chat_input = gr.MultimodalTextbox(
            placeholder="Message OmniFF... or drop a file",
            show_label=False,
            file_count="single",
            sources=["upload", "microphone"],
            submit_btn=True,
            stop_btn=True,
        )

        gr.HTML(
            '<div style="display: flex; justify-content: center; gap: 4px; flex-wrap: wrap; padding: 4px 0 8px 0;">'
            '<span class="omiff-pipeline-tag tag-text">Text</span>'
            '<span class="omiff-pipeline-tag tag-image">Image</span>'
            '<span class="omiff-pipeline-tag tag-audio">Audio</span>'
            '<span class="omiff-pipeline-tag tag-video">Video</span>'
            '<span class="omiff-pipeline-tag tag-doc">Documents</span>'
            '<span class="omiff-pipeline-tag tag-code">Code</span>'
            "</div>"
        )

        def respond(message, history):
            """Handle universal chat interaction."""
            if not message or (not message.get("text", "").strip() and not message.get("files")):
                return history, gr.MultimodalTextbox(value=None)

            text = message.get("text", "")
            files = message.get("files", [])

            # Add user message to history
            if files:
                # Show file in chat
                for f in files:
                    fp = f if isinstance(f, str) else (f.get("path", "") if isinstance(f, dict) else getattr(f, "name", str(f)))
                    file_mod = _classify_file(fp)
                    if file_mod == "image":
                        history.append({"role": "user", "content": gr.Image(fp)})
                    elif file_mod == "audio":
                        history.append({"role": "user", "content": gr.Audio(fp)})
                    elif file_mod == "video":
                        history.append({"role": "user", "content": gr.Video(fp)})
                    else:
                        fname = os.path.basename(fp)
                        history.append({"role": "user", "content": f"[Uploaded: {fname}]"})
                if text:
                    history.append({"role": "user", "content": text})
            else:
                history.append({"role": "user", "content": text})

            # Process
            response = universal_chat(message, history)
            content = response.get("content", "")

            # Handle multimodal response
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            history.append({"role": "assistant", "content": item["text"]})
                        elif item.get("type") == "image":
                            img_path = item.get("image", {}).get("path", "")
                            history.append({"role": "assistant", "content": gr.Image(img_path)})
                    else:
                        history.append({"role": "assistant", "content": str(item)})
            else:
                history.append({"role": "assistant", "content": content})

            return history, gr.MultimodalTextbox(value=None)

        chat_input.submit(respond, [chat_input, chatbot], [chatbot, chat_input])

    # ==================================================================
    # SPECIALIZED TABS — organized by modality
    # ==================================================================

    # ---- Text -> Text ----
    with gr.Tab("Text", id="text"):
        gr.HTML('<p class="omiff-section-label">Text Generation</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                txt_input = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Ask anything...",
                    show_label=True,
                )
                with gr.Accordion("Options", open=False):
                    txt_thinking = gr.Radio(
                        ["off", "normal"],
                        value="off",
                        label="Reasoning mode",
                        info="Enable extended reasoning for complex questions",
                    )
                txt_btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                txt_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        txt_btn.click(process_text, [txt_input, txt_thinking], txt_output)

    # ---- Image -> Text ----
    with gr.Tab("Vision", id="vision"):
        gr.HTML('<p class="omiff-section-label">Image Understanding</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="filepath",
                    label="Image",
                    height=280,
                )
                img_prompt = gr.Textbox(
                    label="Question",
                    value="Describe this image in detail.",
                    lines=2,
                )
                img_btn = gr.Button("Analyze", variant="primary", size="lg")
            with gr.Column(scale=1):
                img_output = gr.Textbox(
                    label="Analysis",
                    lines=10,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        img_btn.click(process_image, [img_input, img_prompt], img_output)

    # ---- Audio -> Text ----
    with gr.Tab("Audio", id="audio"):
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
                    lines=10,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

    # ---- Text -> Image ----
    with gr.Tab("Generate", id="generate"):
        gr.HTML('<p class="omiff-section-label">Image Generation</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gen_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="A cyberpunk city at night, neon reflections on wet streets...",
                )
                with gr.Accordion("Options", open=False):
                    gen_seed = gr.Number(
                        label="Seed",
                        value=-1,
                        info="Use -1 for random. Set a specific number for reproducibility.",
                    )
                gen_btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                gen_output = gr.Image(label="Result", height=400)
        gen_btn.click(generate_image, [gen_prompt, gen_seed], gen_output)

    # ---- Image -> Image ----
    with gr.Tab("Transform", id="transform"):
        gr.HTML('<p class="omiff-section-label">Image Transformation</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                i2i_input = gr.Image(
                    type="filepath",
                    label="Source Image",
                    height=240,
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
                        label="Transformation Strength",
                        info="Lower preserves more of the original. Higher allows more creative freedom.",
                    )
                i2i_btn = gr.Button("Transform", variant="primary", size="lg")
            with gr.Column(scale=1):
                i2i_output = gr.Image(label="Result", height=400)
        i2i_btn.click(edit_image, [i2i_input, i2i_prompt, i2i_strength], i2i_output)

    # ---- Video -> Text ----
    with gr.Tab("Video", id="video"):
        gr.HTML('<p class="omiff-section-label">Video Understanding</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                vid_input = gr.Video(label="Video", height=280)
                vid_prompt = gr.Textbox(
                    label="Question",
                    value="Describe what happens in this video.",
                    lines=2,
                )
                vid_btn = gr.Button("Analyze", variant="primary", size="lg")
            with gr.Column(scale=1):
                vid_output = gr.Textbox(
                    label="Analysis",
                    lines=10,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        vid_btn.click(process_video, [vid_input, vid_prompt], vid_output)

    # ---- Document -> Text ----
    with gr.Tab("Documents", id="documents"):
        gr.HTML('<p class="omiff-section-label">Document Analysis</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                doc_input = gr.File(
                    label="Document",
                    file_types=[".txt", ".md", ".csv", ".json", ".xml", ".html", ".log", ".pdf"],
                )
                doc_prompt = gr.Textbox(
                    label="Instruction",
                    value="Summarize this document.",
                    lines=2,
                )
                doc_btn = gr.Button("Process", variant="primary", size="lg")
            with gr.Column(scale=1):
                doc_output = gr.Textbox(
                    label="Result",
                    lines=12,
                    show_copy_button=True,
                    elem_classes=["output-textbox"],
                )
        doc_btn.click(process_document, [doc_input, doc_prompt], doc_output)

    # ---- Code Generation ----
    with gr.Tab("Code", id="code"):
        gr.HTML('<p class="omiff-section-label">Code Generation</p>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                code_task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Write a function that sorts a list using merge sort...",
                )
                with gr.Accordion("Options", open=False):
                    code_lang = gr.Dropdown(
                        ["Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++", ""],
                        value="Python",
                        label="Language",
                    )
                code_btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                code_output = gr.Code(
                    label="Output",
                    language="python",
                    lines=20,
                )
        code_btn.click(generate_code, [code_task, code_lang], code_output)

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
            <p style="margin-top: 8px;">
                8 pipelines &middot; Text &middot; Vision &middot; Audio &middot; Generate &middot; Transform &middot; Video &middot; Documents &middot; Code
            </p>
        </div>
        """
    )

demo.launch(show_error=True)
