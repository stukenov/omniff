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

    model_id = "openai/whisper-large-v3-turbo"
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
    return _strip_think(response) or "Ответ не сгенерирован."


# ---------------------------------------------------------------------------
# 1. Text -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_text(text: str, thinking: str) -> str:
    if not text.strip():
        return "Введите текст."
    return _llm_generate([{"role": "user", "content": text}], thinking)


# ---------------------------------------------------------------------------
# 2. Image -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_image(image_path: str, prompt: str) -> str:
    if not image_path:
        return "Загрузите изображение."

    _load_vlm()
    model = _vlm["model"]
    processor = _vlm["processor"]

    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt or "Опишите это изображение подробно."},
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
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "Ответ не сгенерирован."


# ---------------------------------------------------------------------------
# 3. Audio -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def process_audio(audio_path: str, language: str) -> str:
    if not audio_path:
        return "Загрузите аудио файл."

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
        or "Ответ не сгенерирован."
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
        return "Загрузите видео."

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
                    "text": prompt or "Опишите, что происходит в этом видео.",
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
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "Ответ не сгенерирован."


# ---------------------------------------------------------------------------
# 7. Document -> Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_document(file_obj, prompt: str) -> str:
    if not file_obj:
        return "Загрузите документ."

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
            return "Для PDF нужен PyMuPDF (fitz). Загрузите .txt файл."
    elif ext in ("txt", "md", "csv", "json", "xml", "html", "log"):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()
    else:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()

    if not text_content.strip():
        return "Не удалось извлечь текст из документа."

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
        return "Опишите, какой код нужно сгенерировать."

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
# 9. Text Translation (LLM-based)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return "Введите текст для перевода."
    if not target_lang:
        return "Выберите целевой язык."
    src = f" from {source_lang}" if source_lang else ""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional translator. Translate the following text{src} to {target_lang}. "
                "Return ONLY the translation, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]
    return _llm_generate(messages, thinking="off", max_tokens=1024)


# ---------------------------------------------------------------------------
# 10. Audio Translation (ASR -> LLM translate)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def translate_audio(audio_path: str, language: str, target_lang: str) -> str:
    if not audio_path:
        return "Загрузите аудио файл."
    if not target_lang:
        return "Выберите целевой язык."

    transcript = process_audio(audio_path, language)
    if transcript.startswith("Please") or transcript.startswith("No output"):
        return transcript

    messages = [
        {
            "role": "system",
            "content": (
                f"Translate the following text to {target_lang}. "
                "Return ONLY the translation."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    translation = _llm_generate(messages, thinking="off", max_tokens=1024)
    return f"**Оригинальная транскрипция:**\n{transcript}\n\n**Перевод ({target_lang}):**\n{translation}"


# ---------------------------------------------------------------------------
# 11. Audio Dubbing (ASR -> translate -> info, TTS needs local GPU)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def dub_audio(audio_path: str, language: str, target_lang: str) -> str:
    if not audio_path:
        return "Загрузите аудио файл."
    if not target_lang:
        return "Выберите целевой язык."

    transcript = process_audio(audio_path, language)
    if transcript.startswith("Please") or transcript.startswith("No output"):
        return transcript

    messages = [
        {
            "role": "system",
            "content": (
                f"Translate the following text to {target_lang}. "
                "Return ONLY the translation."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    translation = _llm_generate(messages, thinking="off", max_tokens=1024)
    return (
        f"**Оригинальная транскрипция:**\n{transcript}\n\n"
        f"**Дубляж текст ({target_lang}):**\n{translation}\n\n"
        f"_(Полный аудио-дубляж с TTS/клонированием голоса требует локальный GPU с OmniVoice)_"
    )


# ---------------------------------------------------------------------------
# 12. Agent mode (multi-step reasoning)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_agent(task: str, max_steps: int) -> str:
    if not task.strip():
        return "Опишите задачу для агента."

    max_steps = max(1, min(int(max_steps), 10))
    output_parts = [f"**Задача:** {task}\n"]

    plan_msgs = [
        {
            "role": "system",
            "content": (
                "You are an AI agent. Break the user's task into clear numbered steps. "
                "Be specific and actionable."
            ),
        },
        {"role": "user", "content": task},
    ]
    plan = _llm_generate(plan_msgs, thinking="off", max_tokens=1024)
    output_parts.append(f"**План:**\n{plan}\n")

    exec_msgs = [
        {
            "role": "system",
            "content": (
                "You are an AI agent executing a plan step by step. "
                "For each step, show your reasoning (Thought), what you do (Action), "
                "and what you observe (Observation). Then give a final answer."
            ),
        },
        {"role": "user", "content": f"Task: {task}\n\nPlan:\n{plan}\n\nExecute this plan now."},
    ]
    result = _llm_generate(exec_msgs, thinking="off", max_tokens=2048)
    output_parts.append(f"**Выполнение:**\n{result}")

    return "\n".join(output_parts)


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
                    "content": f"**Оригинальная транскрипция:**\n{transcript}\n\n**Дубляж ({target_lang}):**\n{translated}\n\n_(Полный аудио-дубляж с TTS требует локальный GPU рантайм)_",
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
                    "content": f"**Оригинальная транскрипция:**\n{transcript}\n\n**Перевод ({target_lang}):**\n{translated}",
                }
            result = process_audio(first_file, "")
            if text:
                full_prompt = f"The user uploaded an audio file. Here is the transcription:\n\n{result}\n\nUser's question: {text}"
                answer = process_text(full_prompt, "off")
                return {"role": "assistant", "content": f"**Транскрипция:**\n{result}\n\n**Ответ:**\n{answer}"}
            return {"role": "assistant", "content": f"**Транскрипция:**\n{result}"}

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
                    "content": f"**Транскрипция видео:**\n{result}\n\n**Дубляж ({target_lang}):**\n{translated}\n\n_(Полный видео-дубляж с микшированием аудио требует локальный GPU рантайм)_",
                }
            result = process_video(first_file, text or "Describe what happens in this video.")
            return {"role": "assistant", "content": result}

        elif modality == "document":
            result = process_document(first_file, text or "Кратко изложите этот документ.")
            return {"role": "assistant", "content": result}

    # Case 2: Text only
    if not text:
        return {
            "role": "assistant",
            "content": "Отправьте сообщение или загрузите файл. Я понимаю текст, изображения, аудио, видео и документы.",
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
                    {"type": "text", "text": f"Сгенерировано из: *{text}*"},
                    {"type": "image", "image": {"path": tmp.name}},
                ],
            }
        return {"role": "assistant", "content": "Не удалось сгенерировать изображение. Попробуйте другой промпт."}

    if intent == "translate":
        target_lang = "English"
        for lang in ["english", "russian", "chinese", "kazakh", "french", "german", "spanish",
                      "русский", "английский", "казахский", "французский", "немецкий"]:
            if lang in text.lower():
                target_lang = lang.capitalize()
                break
        tr_prompt = f"Translate the following text to {target_lang}. Return ONLY the translation.\n\n{text}"
        translated = process_text(tr_prompt, "off")
        return {"role": "assistant", "content": f"**Перевод ({target_lang}):**\n{translated}"}

    if intent == "code":
        result = generate_code(text, "")
        return {"role": "assistant", "content": f"```\n{result}\n```"}

    if intent == "agent":
        steps = []
        steps.append(f"**Задача:** {text}\n")
        thinking_prompt = f"Break this task into steps and solve it:\n\n{text}"
        plan = process_text(thinking_prompt, "off")
        steps.append(f"**План:**\n{plan}\n")
        execution_prompt = f"Execute this plan and provide the final answer:\n\nTask: {text}\nPlan: {plan}"
        answer = process_text(execution_prompt, "off")
        steps.append(f"**Результат:**\n{answer}")
        return {"role": "assistant", "content": "\n".join(steps)}

    # Default: text-to-text
    result = process_text(text, "off")
    return {"role": "assistant", "content": result}


# ---------------------------------------------------------------------------
# Warm-up (pre-download models on CPU at startup)
# ---------------------------------------------------------------------------
_SPACE_MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "openai/whisper-large-v3-turbo",
    "stabilityai/sdxl-turbo",
]


def _download_all_models():
    """Download all models in background so Gradio starts immediately."""
    import threading

    def _do():
        from huggingface_hub import snapshot_download

        for model_id in _SPACE_MODELS:
            try:
                print(f"[OmniFF] Downloading {model_id}...")
                snapshot_download(model_id)
                print(f"[OmniFF] ✓ {model_id} cached")
            except Exception as e:
                print(f"[OmniFF] ✗ {model_id} failed: {e}")
        print("[OmniFF] All models cached.")

    threading.Thread(target=_do, daemon=True).start()


_download_all_models()

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
    "translate": "Language model (Qwen3)",
    "dub": "Whisper + Qwen3",
    "agent": "Language model (Qwen3)",
}


# ---------------------------------------------------------------------------
# Design System -- Round 2: Apple HIG-inspired, production polish
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
/* ==================================================================
   BASE — Typography, Container, 4pt grid
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
    --omiff-violet: #bf5af2;
    --omiff-radius-sm: 6px;
    --omiff-radius-md: 10px;
    --omiff-radius-lg: 14px;
}

.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 0 8px !important;
}

/* Tighten all block spacing globally */
.block { margin-bottom: 2px !important; }
.form { gap: 6px !important; }
.gap { gap: 4px !important; }
.tabitem { padding: 8px 0 0 0 !important; }
.panel { padding: 0 !important; }

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
.dark .omiff-header { border-bottom-color: var(--omiff-border) !important; }
.dark .omiff-header h1 { color: var(--omiff-text-primary) !important; }
.dark .omiff-header .omiff-sub { color: var(--omiff-text-secondary) !important; }
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
.dark .omiff-chat-hint { color: var(--omiff-text-tertiary) !important; }
.dark .omiff-footer {
    border-top-color: var(--omiff-border) !important;
}
.dark .omiff-footer p,
.dark .omiff-footer span { color: var(--omiff-text-tertiary) !important; }
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
    box-shadow: 0 0 0 2px rgba(64, 156, 255, 0.15) !important;
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

/* ==================================================================
   HEADER — Compact single-line identity
   ================================================================== */
.omiff-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 14px 16px 10px 16px;
    border-bottom: 1px solid var(--omiff-border);
    margin-bottom: 0;
}
.omiff-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    color: var(--omiff-text-primary) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.omiff-header .omiff-sub {
    font-size: 13px;
    color: var(--omiff-text-secondary);
    font-weight: 400;
    margin: 0;
    line-height: 1.2;
}
.omiff-header .omiff-badge {
    font-size: 9px;
    font-weight: 600;
    color: var(--omiff-violet);
    background: rgba(191, 90, 242, 0.08);
    padding: 2px 8px;
    border-radius: 100px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ==================================================================
   TABS — Flat hierarchy, no nesting. 8 top-level tabs.
   ================================================================== */
.tab-nav {
    border-bottom: 1px solid var(--omiff-border) !important;
    gap: 0 !important;
    padding: 0 8px !important;
    justify-content: center !important;
}
.tab-nav button {
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    color: var(--omiff-text-secondary) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
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
.tab-nav button:first-child {
    font-weight: 600 !important;
}

/* ==================================================================
   BUTTONS — compact
   ================================================================== */
.primary {
    background: var(--omiff-blue) !important;
    border: none !important;
    border-radius: var(--omiff-radius-md) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
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
    font-size: 12px !important;
    font-weight: 500 !important;
    border: 1px solid var(--omiff-border) !important;
}

/* ==================================================================
   INPUTS — tighter
   ================================================================== */
textarea, input[type="text"], .wrap input {
    border-radius: var(--omiff-radius-sm) !important;
    border: 1px solid var(--omiff-border) !important;
    font-size: 13px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--omiff-blue) !important;
    box-shadow: 0 0 0 2px rgba(0, 113, 227, 0.1) !important;
    outline: none !important;
}

/* ==================================================================
   LABELS — smaller
   ================================================================== */
label span {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--omiff-text-primary) !important;
}

/* ==================================================================
   ACCORDION — progressive disclosure, compact
   ================================================================== */
.accordion {
    border: 1px solid var(--omiff-border-subtle) !important;
    border-radius: var(--omiff-radius-sm) !important;
    margin-top: 4px !important;
    overflow: hidden !important;
}
.accordion > .label-wrap {
    padding: 6px 12px !important;
    font-size: 12px !important;
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
    font-size: 13px !important;
}

/* ==================================================================
   CHATBOT — compact hero
   ================================================================== */
.chatbot-wrap {
    border-radius: var(--omiff-radius-md) !important;
    border: 1px solid var(--omiff-border) !important;
}
.omiff-chat-hint {
    font-size: 12px;
    color: var(--omiff-text-tertiary);
    text-align: center;
    padding: 2px 0 0 0;
    line-height: 1.4;
    margin: 0;
}

/* Status pill shown during processing */
.omiff-status-pill {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    color: var(--omiff-text-secondary);
    background: var(--omiff-surface);
    padding: 4px 12px;
    border-radius: 100px;
    margin: 4px 0;
    animation: omiff-pulse 1.5s ease-in-out infinite;
}
@keyframes omiff-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ==================================================================
   FOOTER — single condensed line
   ================================================================== */
.omiff-footer {
    text-align: center;
    padding: 10px 16px;
    border-top: 1px solid var(--omiff-border);
    margin-top: 4px;
}
.omiff-footer p {
    font-size: 11px !important;
    color: var(--omiff-text-tertiary) !important;
    line-height: 1.5 !important;
    margin: 0 !important;
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
   COMPACT CONTROLS — inline options row
   ================================================================== */
.omiff-inline-opts {
    display: flex;
    align-items: end;
    gap: 8px;
}
"""


# ---------------------------------------------------------------------------
# Gradio UI — Round 3: Compact, dense, flat hierarchy (Apple HIG)
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="OmniFF — FFmpeg для ИИ",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        radius_size=gr.themes.sizes.radius_md,
    ),
    css=_CUSTOM_CSS,
) as demo:

    # ---- Header — single compact line ----
    gr.HTML(
        '<div class="omiff-header">'
        "<h1>OmniFF</h1>"
        '<span class="omiff-sub">FFmpeg для ИИ</span>'
        '<span class="omiff-badge">ZeroGPU</span>'
        "</div>"
    )

    # ==================================================================
    # Flat tab bar — no sub-tabs. Every pipeline is one click away.
    # Chat | Image | Video | Generate | Transform | Transcribe | Text | Documents | Code
    # ==================================================================

    # ------------------------------------------------------------------
    # TAB: Chat — the universal multimodal entry point
    # ------------------------------------------------------------------
    with gr.Tab("Чат", id="chat"):
        gr.HTML(
            '<p class="omiff-chat-hint">'
            "Текст, изображения, аудио, видео, документы — авто-маршрутизация."
            "</p>"
        )

        chatbot = gr.Chatbot(
            height=420,
            type="messages",
            show_label=False,
            avatar_images=(
                None,
                "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
            ),
            placeholder=(
                '<div style="text-align:center; padding:24px 16px; opacity:0.5;">'
                '<p style="font-size:16px; font-weight:600; margin-bottom:2px;">Чем могу помочь?</p>'
                '<p style="font-size:12px;">Загрузите файл или напишите сообщение.</p>'
                "</div>"
            ),
        )

        chat_input = gr.MultimodalTextbox(
            placeholder="Сообщение OmniFF...",
            show_label=False,
            file_count="single",
            sources=["upload", "microphone"],
            submit_btn=True,
            stop_btn=True,
        )

        def respond(message, history):
            if not message or (
                not message.get("text", "").strip() and not message.get("files")
            ):
                return history, gr.MultimodalTextbox(value=None)

            text = message.get("text", "")
            files = message.get("files", [])

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
                    history.append({"role": "user", "content": {"path": fp}})
                if text:
                    history.append({"role": "user", "content": text})
            else:
                history.append({"role": "user", "content": text})

            file_paths = []
            for f in files:
                if isinstance(f, str):
                    file_paths.append(f)
                elif isinstance(f, dict) and "path" in f:
                    file_paths.append(f["path"])
                elif hasattr(f, "name"):
                    file_paths.append(f.name)

            try:
                response = universal_chat(message, history)
                content = response.get("content", "")

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                history.append(
                                    {"role": "assistant", "content": item["text"]}
                                )
                            elif item.get("type") == "image":
                                img_path = item.get("image", {}).get("path", "")
                                history.append(
                                    {"role": "assistant", "content": {"path": img_path}}
                                )
                        else:
                            history.append({"role": "assistant", "content": str(item)})
                else:
                    history.append({"role": "assistant", "content": content})

            except Exception as exc:
                error_msg = str(exc)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Что-то пошло не так.\n\n"
                            f"**Ошибка:** {error_msg}\n\n"
                            f"Попробуйте снова или используйте специализированную вкладку."
                        ),
                    }
                )

            return history, gr.MultimodalTextbox(value=None)

        chat_input.submit(respond, [chat_input, chatbot], [chatbot, chat_input])

    # ------------------------------------------------------------------
    # TAB: Image — analyze an image (VLM)
    # ------------------------------------------------------------------
    with gr.Tab("Изображение", id="image"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_input = gr.Image(type="filepath", label="Изображение", height=240)
                img_prompt = gr.Textbox(
                    label="Вопрос",
                    value="Опишите это изображение подробно.",
                    lines=2,
                )
                img_btn = gr.Button("Анализ", variant="primary")
            with gr.Column(scale=1):
                img_output = gr.Textbox(
                    label="Результат", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        img_btn.click(process_image, [img_input, img_prompt], img_output)

    # ------------------------------------------------------------------
    # TAB: Video — analyze a video (VLM)
    # ------------------------------------------------------------------
    with gr.Tab("Видео", id="video"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                vid_input = gr.Video(label="Видео", height=240)
                vid_prompt = gr.Textbox(
                    label="Вопрос",
                    value="Опишите, что происходит в этом видео.",
                    lines=2,
                )
                vid_btn = gr.Button("Анализ", variant="primary")
            with gr.Column(scale=1):
                vid_output = gr.Textbox(
                    label="Результат", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        vid_btn.click(process_video, [vid_input, vid_prompt], vid_output)

    # ------------------------------------------------------------------
    # TAB: Generate — text to image (SDXL Turbo)
    # ------------------------------------------------------------------
    with gr.Tab("Генерация", id="generate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gen_prompt = gr.Textbox(
                    label="Промпт", lines=2,
                    placeholder="Домик в горах на закате...",
                )
                with gr.Row():
                    gen_seed = gr.Number(label="Сид", value=-1, scale=1)
                    gen_btn = gr.Button("Создать", variant="primary", scale=2)
            with gr.Column(scale=1):
                gen_output = gr.Image(label="Результат", height=360)
        gen_btn.click(generate_image, [gen_prompt, gen_seed], gen_output)

    # ------------------------------------------------------------------
    # TAB: Transform — image to image (SDXL Turbo)
    # ------------------------------------------------------------------
    with gr.Tab("Трансформация", id="transform"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                i2i_input = gr.Image(type="filepath", label="Исходник", height=200)
                i2i_prompt = gr.Textbox(
                    label="Стиль",
                    value="Сделай как акварельную картину",
                    lines=2,
                )
                with gr.Row():
                    i2i_strength = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Сила", scale=2,
                    )
                    i2i_btn = gr.Button("Преобразовать", variant="primary", scale=1)
            with gr.Column(scale=1):
                i2i_output = gr.Image(label="Результат", height=360)
        i2i_btn.click(edit_image, [i2i_input, i2i_prompt, i2i_strength], i2i_output)

    # ------------------------------------------------------------------
    # TAB: Transcribe — audio/speech (Whisper)
    # ------------------------------------------------------------------
    with gr.Tab("Транскрипция", id="transcribe"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                aud_input = gr.Audio(
                    type="filepath", label="Аудио",
                    sources=["upload", "microphone"],
                )
                with gr.Row():
                    aud_lang = gr.Dropdown(
                        ["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"],
                        value="", label="Язык", scale=2,
                        info="Пусто = авто-определение",
                    )
                    aud_btn = gr.Button("Распознать", variant="primary", scale=1)
            with gr.Column(scale=1):
                aud_output = gr.Textbox(
                    label="Транскрипция", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

    # ------------------------------------------------------------------
    # TAB: Text — LLM text generation
    # ------------------------------------------------------------------
    with gr.Tab("Текст", id="text"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                txt_input = gr.Textbox(
                    label="Запрос", lines=3,
                    placeholder="Спросите что угодно...",
                )
                with gr.Row():
                    txt_thinking = gr.Radio(
                        ["off", "normal"], value="off",
                        label="Рассуждение", scale=2,
                    )
                    txt_btn = gr.Button("Генерировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                txt_output = gr.Textbox(
                    label="Ответ", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        txt_btn.click(process_text, [txt_input, txt_thinking], txt_output)

    # ------------------------------------------------------------------
    # TAB: Documents — document analysis (LLM)
    # ------------------------------------------------------------------
    with gr.Tab("Документы", id="documents"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                doc_input = gr.File(
                    label="Документ",
                    file_types=[".txt", ".md", ".csv", ".json",
                                ".xml", ".html", ".log", ".pdf"],
                )
                doc_prompt = gr.Textbox(
                    label="Инструкция",
                    value="Кратко изложите этот документ.",
                    lines=2,
                )
                doc_btn = gr.Button("Обработать", variant="primary")
            with gr.Column(scale=1):
                doc_output = gr.Textbox(
                    label="Результат", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        doc_btn.click(process_document, [doc_input, doc_prompt], doc_output)

    # ------------------------------------------------------------------
    # TAB: Code — code generation (LLM)
    # ------------------------------------------------------------------
    with gr.Tab("Код", id="code"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                code_task = gr.Textbox(
                    label="Задача", lines=3,
                    placeholder="Напиши функцию сортировки слиянием...",
                )
                with gr.Row():
                    code_lang = gr.Dropdown(
                        ["Python", "JavaScript", "TypeScript",
                         "Rust", "Go", "Java", "C++", ""],
                        value="Python", label="Язык", scale=2,
                    )
                    code_btn = gr.Button("Генерировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                code_output = gr.Code(label="Результат", language="python", lines=14)
        code_btn.click(generate_code, [code_task, code_lang], code_output)

    # ------------------------------------------------------------------
    # TAB: Translate — text and audio translation
    # ------------------------------------------------------------------
    with gr.Tab("Перевод", id="translate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                tr_input = gr.Textbox(
                    label="Текст для перевода", lines=4,
                    placeholder="Введите текст или загрузите аудио ниже...",
                )
                tr_audio = gr.Audio(
                    type="filepath", label="Или загрузите аудио",
                    sources=["upload", "microphone"],
                )
                with gr.Row():
                    tr_src_lang = gr.Dropdown(
                        ["", "Английский", "Русский", "Казахский", "Китайский",
                         "Французский", "Немецкий", "Испанский", "Японский", "Корейский"],
                        value="", label="Исходный", scale=1,
                        info="Пусто = авто-определение",
                    )
                    tr_tgt_lang = gr.Dropdown(
                        ["Английский", "Русский", "Казахский", "Китайский",
                         "Французский", "Немецкий", "Испанский", "Японский", "Корейский"],
                        value="Русский", label="Целевой", scale=1,
                    )
                    tr_btn = gr.Button("Перевести", variant="primary", scale=1)
            with gr.Column(scale=1):
                tr_output = gr.Textbox(
                    label="Перевод", lines=10,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )

        def _handle_translate(text, audio, src_lang, tgt_lang):
            if audio:
                lang_code = ""
                lang_map = {"английский": "en", "русский": "ru", "казахский": "kk",
                            "китайский": "zh", "французский": "fr", "немецкий": "de",
                            "испанский": "es", "японский": "ja", "корейский": "ko"}
                if src_lang:
                    lang_code = lang_map.get(src_lang.lower(), "")
                return translate_audio(audio, lang_code, tgt_lang)
            if text and text.strip():
                return translate_text(text, src_lang, tgt_lang)
            return "Введите текст или загрузите аудио."

        tr_btn.click(
            _handle_translate,
            [tr_input, tr_audio, tr_src_lang, tr_tgt_lang],
            tr_output,
        )

    # ------------------------------------------------------------------
    # TAB: Dub — audio/video dubbing pipeline
    # ------------------------------------------------------------------
    with gr.Tab("Дубляж", id="dub"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                dub_audio_input = gr.Audio(
                    type="filepath", label="Аудио файл",
                    sources=["upload", "microphone"],
                )
                dub_video_input = gr.Video(label="Или видео файл", height=200)
                with gr.Row():
                    dub_src_lang = gr.Dropdown(
                        ["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"],
                        value="", label="Исходный язык", scale=1,
                        info="Пусто = авто-определение",
                    )
                    dub_tgt_lang = gr.Dropdown(
                        ["Английский", "Русский", "Казахский", "Китайский",
                         "Французский", "Немецкий", "Испанский", "Японский"],
                        value="Русский", label="Целевой язык", scale=1,
                    )
                    dub_btn = gr.Button("Дублировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                dub_output = gr.Textbox(
                    label="Результат дубляжа", lines=12,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )

        def _handle_dub(audio, video, src_lang, tgt_lang):
            if video:
                transcript = process_video(video, "Transcribe all speech in this video word by word.")
                messages = [
                    {
                        "role": "system",
                        "content": f"Translate the following text to {tgt_lang}. Return ONLY the translation.",
                    },
                    {"role": "user", "content": transcript},
                ]
                translation = _llm_generate(messages, thinking="off", max_tokens=1024)
                return (
                    f"**Транскрипция видео:**\n{transcript}\n\n"
                    f"**Дубляж ({tgt_lang}):**\n{translation}\n\n"
                    f"_(Полный видео-дубляж с микшированием аудио требует локальный GPU рантайм)_"
                )
            if audio:
                return dub_audio(audio, src_lang, tgt_lang)
            return "Загрузите аудио или видео файл."

        dub_btn.click(
            _handle_dub,
            [dub_audio_input, dub_video_input, dub_src_lang, dub_tgt_lang],
            dub_output,
        )

    # ------------------------------------------------------------------
    # TAB: Agent — multi-step reasoning
    # ------------------------------------------------------------------
    with gr.Tab("Агент", id="agent"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                agent_task = gr.Textbox(
                    label="Задача", lines=3,
                    placeholder="Исследуй и объясни применения квантовых вычислений...",
                )
                with gr.Row():
                    agent_steps = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Макс. шагов", scale=2,
                    )
                    agent_btn = gr.Button("Запустить", variant="primary", scale=1)
            with gr.Column(scale=1):
                agent_output = gr.Textbox(
                    label="Результат агента", lines=14,
                    show_copy_button=True, elem_classes=["output-textbox"],
                )
        agent_btn.click(run_agent, [agent_task, agent_steps], agent_output)

    # ---- Footer — condensed single line ----
    gr.HTML(
        '<div class="omiff-footer">'
        "<p>"
        '<a href="https://github.com/stukenov/omniff" target="_blank">GitHub</a>'
        " &middot; "
        '<a href="https://huggingface.co/stukenov/omniff" target="_blank">HuggingFace</a>'
        " &middot; "
        '<a href="https://github.com/stukenov" target="_blank">Saken Tukenov</a>'
        " &middot; "
        "Qwen3 &middot; Qwen2.5-VL &middot; Whisper &middot; SDXL Turbo &middot; Перевод &middot; Дубляж &middot; Агент"
        "</p>"
        "</div>"
    )

demo.launch(show_error=True)
