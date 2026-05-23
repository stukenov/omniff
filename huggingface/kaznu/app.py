"""
OmniFF — FFmpeg для ИИ | KazNU Server (2x A10 22GB)

Full-featured Gradio app with all pipelines.
GPU0: LLM + VLM | GPU1: ASR + Image Gen
"""

import mimetypes
import os
import re
import tempfile
import threading

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Model holders
# ---------------------------------------------------------------------------
_llm = {"model": None, "tokenizer": None}
_vlm = {"model": None, "processor": None}
_asr = {"model": None, "processor": None}
_translator = {"model": None, "tokenizer": None}
_t2i = {"pipe": None}
_i2i = {"pipe": None}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Model loaders — GPU0: LLM+VLM, GPU1: ASR+T2I
# ---------------------------------------------------------------------------
def _load_llm():
    if _llm["model"] is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from transformers import BitsAndBytesConfig

    model_id = "Qwen/Qwen3.6-35B-A3B"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    _llm["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
    _llm["model"] = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map={"": "cuda:0"}
    )


def _load_vlm():
    if _vlm["model"] is not None:
        return
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from transformers import BitsAndBytesConfig

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    _vlm["processor"] = AutoProcessor.from_pretrained(model_id)
    _vlm["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant_config, device_map={"": "cuda:0"}
    )


def _load_asr():
    if _asr["model"] is not None:
        return
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = "openai/whisper-large-v3-turbo"
    _asr["processor"] = WhisperProcessor.from_pretrained(model_id)
    _asr["model"] = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": "cuda:1"}
    )


def _load_translator():
    if _translator["model"] is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "tencent/Hy-MT2-30B-A3B"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    _translator["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
    _translator["model"] = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map={"": "cuda:0"}
    )


def _translate_with_hymt(text: str, src_lang: str, tgt_lang: str) -> str:
    _load_translator()
    model = _translator["model"]
    tokenizer = _translator["tokenizer"]
    prompt = f"Translate the following text from {src_lang} to {tgt_lang}.\n{src_lang}: {text}\n{tgt_lang}:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=1024)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_t2i():
    if _t2i["pipe"] is not None:
        return
    from diffusers import DiffusionPipeline

    _t2i["pipe"] = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.float16,
    ).to("cuda:1")


def _load_i2i():
    if _i2i["pipe"] is not None:
        return
    from diffusers import AutoPipelineForImage2Image

    _i2i["pipe"] = AutoPipelineForImage2Image.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.float16,
    ).to("cuda:1")


# ---------------------------------------------------------------------------
# LLM generate helper
# ---------------------------------------------------------------------------
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
# Pipeline functions
# ---------------------------------------------------------------------------
def process_text(text: str, thinking: str) -> str:
    if not text.strip():
        return "Введите текст."
    return _llm_generate([{"role": "user", "content": text}], thinking)


def process_image(image_path: str, prompt: str) -> str:
    if not image_path:
        return "Загрузите изображение."
    _load_vlm()
    model = _vlm["model"]
    processor = _vlm["processor"]
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt or "Опишите это изображение подробно."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "Ответ не сгенерирован."


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
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip() or "Ответ не сгенерирован."


def generate_image(prompt: str, seed: int):
    if not prompt.strip():
        return None
    _load_t2i()
    pipe = _t2i["pipe"]
    generator = torch.Generator("cuda:1")
    if seed >= 0:
        generator.manual_seed(int(seed))
    image = pipe(prompt=prompt, num_inference_steps=4, generator=generator).images[0]
    return image


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
    ).images[0]
    return image


def process_video(video_path: str, prompt: str) -> str:
    if not video_path:
        return "Загрузите видео."
    _load_vlm()
    model = _vlm["model"]
    processor = _vlm["processor"]
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
        {"type": "text", "text": prompt or "Опишите, что происходит в этом видео."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "Ответ не сгенерирован."


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
            return "Для PDF нужен PyMuPDF (fitz)."
    else:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()
    if not text_content.strip():
        return "Не удалось извлечь текст из документа."
    if len(text_content) > 8000:
        text_content = text_content[:8000] + "\n\n[... обрезано ...]"
    user_prompt = prompt or "Кратко изложите этот документ."
    messages = [{"role": "user", "content": f"{user_prompt}\n\n---\n\n{text_content}"}]
    return _llm_generate(messages, thinking="off", max_tokens=1024)


def generate_code(task: str, language: str) -> str:
    if not task.strip():
        return "Опишите, какой код нужно сгенерировать."
    lang_hint = f" in {language}" if language else ""
    messages = [
        {"role": "system", "content": f"You are an expert programmer. Write clean, well-structured code{lang_hint}. Return ONLY the code, no explanations unless asked."},
        {"role": "user", "content": task},
    ]
    return _llm_generate(messages, thinking="off", max_tokens=2048)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return "Введите текст для перевода."
    if not target_lang:
        return "Выберите целевой язык."
    src = source_lang or "Auto"
    try:
        return _translate_with_hymt(text, src, target_lang)
    except Exception:
        src_hint = f" from {source_lang}" if source_lang else ""
        messages = [
            {"role": "system", "content": f"Translate the following text{src_hint} to {target_lang}. Return ONLY the translation."},
            {"role": "user", "content": text},
        ]
        return _llm_generate(messages, thinking="off", max_tokens=1024)


def translate_audio(audio_path: str, language: str, target_lang: str) -> str:
    if not audio_path:
        return "Загрузите аудио файл."
    if not target_lang:
        return "Выберите целевой язык."
    transcript = process_audio(audio_path, language)
    if transcript.startswith("Загрузите") or transcript.startswith("Ответ не"):
        return transcript
    messages = [
        {"role": "system", "content": f"Translate the following text to {target_lang}. Return ONLY the translation."},
        {"role": "user", "content": transcript},
    ]
    translation = _llm_generate(messages, thinking="off", max_tokens=1024)
    return f"**Оригинальная транскрипция:**\n{transcript}\n\n**Перевод ({target_lang}):**\n{translation}"


def dub_audio(audio_path: str, language: str, target_lang: str) -> str:
    if not audio_path:
        return "Загрузите аудио файл."
    if not target_lang:
        return "Выберите целевой язык."
    transcript = process_audio(audio_path, language)
    if transcript.startswith("Загрузите") or transcript.startswith("Ответ не"):
        return transcript
    messages = [
        {"role": "system", "content": f"Translate the following text to {target_lang}. Return ONLY the translation."},
        {"role": "user", "content": transcript},
    ]
    translation = _llm_generate(messages, thinking="off", max_tokens=1024)
    return (
        f"**Оригинальная транскрипция:**\n{transcript}\n\n"
        f"**Дубляж текст ({target_lang}):**\n{translation}"
    )


def run_agent(task: str, max_steps: int) -> str:
    if not task.strip():
        return "Опишите задачу для агента."
    max_steps = max(1, min(int(max_steps), 10))
    output_parts = [f"**Задача:** {task}\n"]
    plan_msgs = [
        {"role": "system", "content": "You are an AI agent. Break the user's task into clear numbered steps. Be specific and actionable."},
        {"role": "user", "content": task},
    ]
    plan = _llm_generate(plan_msgs, thinking="off", max_tokens=1024)
    output_parts.append(f"**План:**\n{plan}\n")
    exec_msgs = [
        {"role": "system", "content": "You are an AI agent executing a plan step by step. For each step, show your reasoning (Thought), what you do (Action), and what you observe (Observation). Then give a final answer."},
        {"role": "user", "content": f"Task: {task}\n\nPlan:\n{plan}\n\nExecute this plan now."},
    ]
    result = _llm_generate(exec_msgs, thinking="off", max_tokens=2048)
    output_parts.append(f"**Выполнение:**\n{result}")
    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# Universal Chat
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv"}
_DOC_EXTS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".log", ".pdf"}

_IMAGE_GEN_PATTERNS = re.compile(
    r"\b(generate|create|draw|paint|make|design|render|imagine|visualize|нарисуй|создай|сгенерируй)\b.*\b(image|picture|photo|illustration|art|painting|drawing|icon|logo|картинк|изображени|фото|рисун)\b",
    re.IGNORECASE,
)
_CODE_PATTERNS = re.compile(
    r"\b(write|code|implement|create|build|generate|напиши|создай)\b.*\b(function|class|script|program|code|api|endpoint|module|component|algorithm|функци|класс|скрипт|код|программ)\b",
    re.IGNORECASE,
)
_TRANSLATE_PATTERNS = re.compile(
    r"\b(translate|переведи|перевод|translation|переведите|to english|to russian|to chinese|на английский|на русский|на казахский)\b",
    re.IGNORECASE,
)
_AGENT_PATTERNS = re.compile(
    r"\b(agent|step by step|investigate|research and|plan and execute|multi-step|агент|пошагово|исследуй|разберись)\b",
    re.IGNORECASE,
)


def _classify_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _DOC_EXTS:
        return "document"
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
    if _IMAGE_GEN_PATTERNS.search(text):
        return "text_to_image"
    if _CODE_PATTERNS.search(text):
        return "code"
    if _TRANSLATE_PATTERNS.search(text):
        return "translate"
    if _AGENT_PATTERNS.search(text):
        return "agent"
    return "text"


def universal_chat(message: dict, history: list) -> dict:
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    file_paths = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(f)
        elif isinstance(f, dict) and "path" in f:
            file_paths.append(f["path"])
        elif hasattr(f, "name"):
            file_paths.append(f.name)

    if file_paths:
        first_file = file_paths[0]
        modality = _classify_file(first_file)
        if modality == "image":
            result = process_image(first_file, text or "Опишите это изображение подробно.")
            return {"role": "assistant", "content": result}
        elif modality == "audio":
            result = process_audio(first_file, "")
            if text:
                full_prompt = f"The user uploaded an audio file. Here is the transcription:\n\n{result}\n\nUser's question: {text}"
                answer = process_text(full_prompt, "off")
                return {"role": "assistant", "content": f"**Транскрипция:**\n{result}\n\n**Ответ:**\n{answer}"}
            return {"role": "assistant", "content": f"**Транскрипция:**\n{result}"}
        elif modality == "video":
            result = process_video(first_file, text or "Опишите, что происходит в этом видео.")
            return {"role": "assistant", "content": result}
        elif modality == "document":
            result = process_document(first_file, text or "Кратко изложите этот документ.")
            return {"role": "assistant", "content": result}

    if not text:
        return {"role": "assistant", "content": "Отправьте сообщение или загрузите файл. Я понимаю текст, изображения, аудио, видео и документы."}

    intent = _detect_intent(text)
    if intent == "text_to_image":
        img = generate_image(text, -1)
        if img is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            return {"role": "assistant", "content": [
                {"type": "text", "text": f"Сгенерировано из: *{text}*"},
                {"type": "image", "image": {"path": tmp.name}},
            ]}
        return {"role": "assistant", "content": "Не удалось сгенерировать изображение."}
    if intent == "translate":
        tr_prompt = f"Translate the following text to English. Return ONLY the translation.\n\n{text}"
        translated = process_text(tr_prompt, "off")
        return {"role": "assistant", "content": f"**Перевод:**\n{translated}"}
    if intent == "code":
        result = generate_code(text, "")
        return {"role": "assistant", "content": f"```\n{result}\n```"}
    if intent == "agent":
        result = run_agent(text, 5)
        return {"role": "assistant", "content": result}

    result = process_text(text, "off")
    return {"role": "assistant", "content": result}


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
:root {
    --omiff-blue: #0071e3;
    --omiff-blue-hover: #0077ed;
    --omiff-text-primary: #1d1d1f;
    --omiff-text-secondary: #86868b;
    --omiff-text-tertiary: #aeaeb2;
    --omiff-border: rgba(0, 0, 0, 0.08);
    --omiff-surface: rgba(0, 0, 0, 0.015);
    --omiff-violet: #bf5af2;
    --omiff-radius-sm: 6px;
    --omiff-radius-md: 10px;
}
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
}
.block { margin-bottom: 2px !important; }
.form { gap: 6px !important; }
.tabitem { padding: 8px 0 0 0 !important; }
.omiff-header {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    padding: 14px 16px 10px 16px; border-bottom: 1px solid var(--omiff-border); margin-bottom: 0;
}
.omiff-header h1 { font-size: 22px !important; font-weight: 700 !important; margin: 0 !important; }
.omiff-header .omiff-sub { font-size: 13px; color: var(--omiff-text-secondary); }
.omiff-header .omiff-badge {
    font-size: 9px; font-weight: 600; color: var(--omiff-violet);
    background: rgba(191, 90, 242, 0.08); padding: 2px 8px; border-radius: 100px;
    text-transform: uppercase;
}
.tab-nav {
    border-bottom: 1px solid var(--omiff-border) !important;
    justify-content: center !important;
}
.tab-nav button {
    font-size: 12px !important; font-weight: 500 !important; padding: 8px 14px !important;
    color: var(--omiff-text-secondary) !important; border: none !important;
    border-bottom: 2px solid transparent !important; background: none !important;
}
.tab-nav button.selected {
    color: var(--omiff-blue) !important; border-bottom: 2px solid var(--omiff-blue) !important;
    font-weight: 600 !important;
}
.primary {
    background: var(--omiff-blue) !important; border: none !important;
    border-radius: var(--omiff-radius-md) !important; font-size: 13px !important; font-weight: 600 !important;
    padding: 8px 20px !important;
}
.omiff-footer {
    text-align: center; padding: 10px 16px; border-top: 1px solid var(--omiff-border); margin-top: 4px;
}
.omiff-footer p { font-size: 11px !important; color: var(--omiff-text-tertiary) !important; margin: 0 !important; }
.omiff-footer a { color: var(--omiff-blue) !important; text-decoration: none !important; }
.omiff-chat-hint { font-size: 12px; color: var(--omiff-text-tertiary); text-align: center; padding: 2px 0 0 0; }
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="OmniFF — FFmpeg для ИИ") as demo:

    gr.HTML(
        '<div class="omiff-header">'
        "<h1>OmniFF</h1>"
        '<span class="omiff-sub">FFmpeg для ИИ</span>'
        '<span class="omiff-badge">KazNU · 2×A10</span>'
        "</div>"
    )

    # --- Чат ---
    with gr.Tab("Чат", id="chat"):
        gr.HTML('<p class="omiff-chat-hint">Текст, изображения, аудио, видео, документы — авто-маршрутизация.</p>')
        chatbot = gr.Chatbot(
            height=420, show_label=False,
            placeholder='<div style="text-align:center; padding:24px 16px; opacity:0.5;"><p style="font-size:16px; font-weight:600;">Чем могу помочь?</p><p style="font-size:12px;">Загрузите файл или напишите сообщение.</p></div>',
        )
        chat_input = gr.MultimodalTextbox(
            placeholder="Сообщение OmniFF...", show_label=False,
            file_count="single", sources=["upload", "microphone"], submit_btn=True, stop_btn=True,
        )

        def respond(message, history):
            if not message or (not message.get("text", "").strip() and not message.get("files")):
                return history, gr.MultimodalTextbox(value=None)
            text = message.get("text", "")
            files = message.get("files", [])
            if files:
                for f in files:
                    fp = f if isinstance(f, str) else (f.get("path", "") if isinstance(f, dict) else getattr(f, "name", str(f)))
                    history.append({"role": "user", "content": {"path": fp}})
                if text:
                    history.append({"role": "user", "content": text})
            else:
                history.append({"role": "user", "content": text})
            try:
                response = universal_chat(message, history)
                content = response.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                history.append({"role": "assistant", "content": item["text"]})
                            elif item.get("type") == "image":
                                history.append({"role": "assistant", "content": {"path": item.get("image", {}).get("path", "")}})
                        else:
                            history.append({"role": "assistant", "content": str(item)})
                else:
                    history.append({"role": "assistant", "content": content})
            except Exception as exc:
                error_msg = str(exc)[:200]
                history.append({"role": "assistant", "content": f"Что-то пошло не так.\n\n**Ошибка:** {error_msg}\n\nПопробуйте снова."})
            return history, gr.MultimodalTextbox(value=None)

        chat_input.submit(respond, [chat_input, chatbot], [chatbot, chat_input])

    # --- Изображение ---
    with gr.Tab("Изображение", id="image"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_input = gr.Image(type="filepath", label="Изображение", height=240)
                img_prompt = gr.Textbox(label="Вопрос", value="Опишите это изображение подробно.", lines=2)
                img_btn = gr.Button("Анализ", variant="primary")
            with gr.Column(scale=1):
                img_output = gr.Textbox(label="Результат", lines=10)
        img_btn.click(process_image, [img_input, img_prompt], img_output)

    # --- Видео ---
    with gr.Tab("Видео", id="video"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                vid_input = gr.Video(label="Видео", height=240)
                vid_prompt = gr.Textbox(label="Вопрос", value="Опишите, что происходит в этом видео.", lines=2)
                vid_btn = gr.Button("Анализ", variant="primary")
            with gr.Column(scale=1):
                vid_output = gr.Textbox(label="Результат", lines=10)
        vid_btn.click(process_video, [vid_input, vid_prompt], vid_output)

    # --- Генерация ---
    with gr.Tab("Генерация", id="generate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gen_prompt = gr.Textbox(label="Промпт", lines=2, placeholder="Домик в горах на закате...")
                with gr.Row():
                    gen_seed = gr.Number(label="Сид", value=-1, scale=1)
                    gen_btn = gr.Button("Создать", variant="primary", scale=2)
            with gr.Column(scale=1):
                gen_output = gr.Image(label="Результат", height=360)
        gen_btn.click(generate_image, [gen_prompt, gen_seed], gen_output)

    # --- Трансформация ---
    with gr.Tab("Трансформация", id="transform"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                i2i_input = gr.Image(type="filepath", label="Исходник", height=200)
                i2i_prompt = gr.Textbox(label="Стиль", value="Сделай как акварельную картину", lines=2)
                with gr.Row():
                    i2i_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Сила", scale=2)
                    i2i_btn = gr.Button("Преобразовать", variant="primary", scale=1)
            with gr.Column(scale=1):
                i2i_output = gr.Image(label="Результат", height=360)
        i2i_btn.click(edit_image, [i2i_input, i2i_prompt, i2i_strength], i2i_output)

    # --- Транскрипция ---
    with gr.Tab("Транскрипция", id="transcribe"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                aud_input = gr.Audio(type="filepath", label="Аудио", sources=["upload", "microphone"])
                with gr.Row():
                    aud_lang = gr.Dropdown(["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"], value="", label="Язык", scale=2, info="Пусто = авто-определение")
                    aud_btn = gr.Button("Распознать", variant="primary", scale=1)
            with gr.Column(scale=1):
                aud_output = gr.Textbox(label="Транскрипция", lines=10)
        aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

    # --- Текст ---
    with gr.Tab("Текст", id="text"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                txt_input = gr.Textbox(label="Запрос", lines=3, placeholder="Спросите что угодно...")
                with gr.Row():
                    txt_thinking = gr.Radio(["off", "normal"], value="off", label="Рассуждение", scale=2)
                    txt_btn = gr.Button("Генерировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Ответ", lines=10)
        txt_btn.click(process_text, [txt_input, txt_thinking], txt_output)

    # --- Документы ---
    with gr.Tab("Документы", id="documents"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                doc_input = gr.File(label="Документ", file_types=[".txt", ".md", ".csv", ".json", ".xml", ".html", ".log", ".pdf"])
                doc_prompt = gr.Textbox(label="Инструкция", value="Кратко изложите этот документ.", lines=2)
                doc_btn = gr.Button("Обработать", variant="primary")
            with gr.Column(scale=1):
                doc_output = gr.Textbox(label="Результат", lines=10)
        doc_btn.click(process_document, [doc_input, doc_prompt], doc_output)

    # --- Код ---
    with gr.Tab("Код", id="code"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                code_task = gr.Textbox(label="Задача", lines=3, placeholder="Напиши функцию сортировки слиянием...")
                with gr.Row():
                    code_lang = gr.Dropdown(["Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++", ""], value="Python", label="Язык", scale=2)
                    code_btn = gr.Button("Генерировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                code_output = gr.Code(label="Результат", language="python", lines=14)
        code_btn.click(generate_code, [code_task, code_lang], code_output)

    # --- Перевод ---
    with gr.Tab("Перевод", id="translate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                tr_input = gr.Textbox(label="Текст для перевода", lines=4, placeholder="Введите текст или загрузите аудио ниже...")
                tr_audio = gr.Audio(type="filepath", label="Или загрузите аудио", sources=["upload", "microphone"])
                with gr.Row():
                    tr_src_lang = gr.Dropdown(["", "Английский", "Русский", "Казахский", "Китайский", "Французский", "Немецкий", "Испанский", "Японский", "Корейский"],
                                              value="", label="Исходный", scale=1, info="Пусто = авто-определение")
                    tr_tgt_lang = gr.Dropdown(["Английский", "Русский", "Казахский", "Китайский", "Французский", "Немецкий", "Испанский", "Японский", "Корейский"],
                                              value="Русский", label="Целевой", scale=1)
                    tr_btn = gr.Button("Перевести", variant="primary", scale=1)
            with gr.Column(scale=1):
                tr_output = gr.Textbox(label="Перевод", lines=10)

        def _handle_translate(text, audio, src_lang, tgt_lang):
            if audio:
                lang_code = ""
                lang_map = {"английский": "en", "русский": "ru", "казахский": "kk", "китайский": "zh",
                            "французский": "fr", "немецкий": "de", "испанский": "es", "японский": "ja", "корейский": "ko"}
                if src_lang:
                    lang_code = lang_map.get(src_lang.lower(), "")
                return translate_audio(audio, lang_code, tgt_lang)
            if text and text.strip():
                return translate_text(text, src_lang, tgt_lang)
            return "Введите текст или загрузите аудио."

        tr_btn.click(_handle_translate, [tr_input, tr_audio, tr_src_lang, tr_tgt_lang], tr_output)

    # --- Дубляж ---
    with gr.Tab("Дубляж", id="dub"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                dub_audio_input = gr.Audio(type="filepath", label="Аудио файл", sources=["upload", "microphone"])
                dub_video_input = gr.Video(label="Или видео файл", height=200)
                with gr.Row():
                    dub_src_lang = gr.Dropdown(["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"],
                                               value="", label="Исходный язык", scale=1, info="Пусто = авто-определение")
                    dub_tgt_lang = gr.Dropdown(["Английский", "Русский", "Казахский", "Китайский", "Французский", "Немецкий", "Испанский", "Японский"],
                                               value="Русский", label="Целевой язык", scale=1)
                    dub_btn = gr.Button("Дублировать", variant="primary", scale=1)
            with gr.Column(scale=1):
                dub_output = gr.Textbox(label="Результат дубляжа", lines=12)

        def _handle_dub(audio, video, src_lang, tgt_lang):
            if video:
                transcript = process_video(video, "Transcribe all speech in this video word by word.")
                messages = [
                    {"role": "system", "content": f"Translate the following text to {tgt_lang}. Return ONLY the translation."},
                    {"role": "user", "content": transcript},
                ]
                translation = _llm_generate(messages, thinking="off", max_tokens=1024)
                return f"**Транскрипция видео:**\n{transcript}\n\n**Дубляж ({tgt_lang}):**\n{translation}"
            if audio:
                return dub_audio(audio, src_lang, tgt_lang)
            return "Загрузите аудио или видео файл."

        dub_btn.click(_handle_dub, [dub_audio_input, dub_video_input, dub_src_lang, dub_tgt_lang], dub_output)

    # --- Перевод изображений ---
    with gr.Tab("Перевод фото", id="img_translate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                imtr_input = gr.Image(type="filepath", label="Изображение с текстом", height=280)
                with gr.Row():
                    imtr_tgt_lang = gr.Dropdown(
                        ["Русский", "Английский", "Казахский", "Китайский", "Французский", "Немецкий", "Испанский"],
                        value="Русский", label="Перевести на", scale=2,
                    )
                    imtr_btn = gr.Button("Перевести", variant="primary", scale=1)
            with gr.Column(scale=1):
                imtr_output = gr.Textbox(label="Результат перевода", lines=14)

        def _translate_image(image_path, target_lang):
            if not image_path:
                return "Загрузите изображение с текстом."
            ocr_prompt = "Read ALL text visible in this image. Output the text exactly as written, preserving layout. If there are multiple text blocks, separate them with blank lines."
            extracted = process_image(image_path, ocr_prompt)
            if not extracted or extracted == "Ответ не сгенерирован.":
                return "Не удалось распознать текст на изображении."
            messages = [
                {"role": "system", "content": f"Translate the following text to {target_lang}. Preserve the original formatting and line breaks. Return ONLY the translation."},
                {"role": "user", "content": extracted},
            ]
            translation = _llm_generate(messages, thinking="off", max_tokens=1024)
            return f"**Распознанный текст:**\n{extracted}\n\n---\n\n**Перевод ({target_lang}):**\n{translation}"

        imtr_btn.click(_translate_image, [imtr_input, imtr_tgt_lang], imtr_output)

    # --- Агент ---
    with gr.Tab("Агент", id="agent"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                agent_task = gr.Textbox(label="Задача", lines=3, placeholder="Исследуй и объясни применения квантовых вычислений...")
                with gr.Row():
                    agent_steps = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Макс. шагов", scale=2)
                    agent_btn = gr.Button("Запустить", variant="primary", scale=1)
            with gr.Column(scale=1):
                agent_output = gr.Textbox(label="Результат агента", lines=14)
        agent_btn.click(run_agent, [agent_task, agent_steps], agent_output)

    gr.HTML(
        '<div class="omiff-footer"><p>'
        '<a href="https://github.com/stukenov/omniff" target="_blank">GitHub</a> &middot; '
        '<a href="https://huggingface.co/stukenov/omniff" target="_blank">HuggingFace</a> &middot; '
        '<a href="https://github.com/stukenov" target="_blank">Saken Tukenov</a> &middot; '
        'Qwen3 &middot; Qwen2.5-VL &middot; Whisper &middot; Z-Image-Turbo &middot; 2×A10 22GB'
        '</p></div>'
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
)
