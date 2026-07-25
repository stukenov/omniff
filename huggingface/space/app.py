"""
OmniFF — FFmpeg for AI | HuggingFace Space (ZeroGPU)
Nuxt SPA via iframe + API routes via Gradio's FastAPI
"""

import os
import re
import tempfile

import gradio as gr
import spaces
import torch

_llm = {"model": None, "tokenizer": None}
_vlm = {"model": None, "processor": None}
_asr = {"model": None, "processor": None}
_t2i = {"pipe": None}

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
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")


def _llm_generate(messages: list, thinking: str = "off", max_tokens: int = 512) -> str:
    _load_llm()
    model, tokenizer = _llm["model"], _llm["tokenizer"]
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if thinking == "off":
        chat_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    if thinking != "off":
        max_tokens = max(max_tokens, 2048)
    generated = model.generate(**inputs, max_new_tokens=max_tokens)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_think(response) or "Ответ не сгенерирован."


# --- Gradio-wrapped API functions (called by gr.Interface) ---

@spaces.GPU(duration=120)
def gr_generate_text(prompt: str, thinking: str) -> str:
    if not prompt.strip():
        return "Введите текст."
    return _llm_generate([{"role": "user", "content": prompt}], thinking)


@spaces.GPU(duration=120)
def gr_describe_image(image_path: str, prompt: str) -> str:
    if not image_path:
        return "Загрузите изображение."
    _load_vlm()
    from qwen_vl_utils import process_vision_info
    model, processor = _vlm["model"], _vlm["processor"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt or "Опишите это изображение подробно."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "Ответ не сгенерирован."


@spaces.GPU(duration=60)
def gr_transcribe(audio_path: str, language: str) -> str:
    if not audio_path:
        return "Загрузите аудио."
    _load_asr()
    import numpy as np
    import soundfile as sf
    model, processor = _asr["model"], _asr["processor"]
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
        sr = 16000
    input_features = processor(audio_data, sampling_rate=sr, return_tensors="pt"
                               ).input_features.to(device=model.device, dtype=model.dtype)
    gen_kwargs = {}
    if language:
        gen_kwargs["language"] = language
    predicted_ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip() or "Ответ не сгенерирован."


@spaces.GPU(duration=60)
def gr_generate_image(prompt: str, seed: int):
    if not prompt.strip():
        return None
    _load_t2i()
    pipe = _t2i["pipe"]
    generator = torch.Generator("cuda")
    if seed >= 0:
        generator.manual_seed(int(seed))
    return pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator).images[0]


@spaces.GPU(duration=120)
def gr_translate(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return "Введите текст."
    src = f" from {source_lang}" if source_lang else ""
    messages = [
        {"role": "system", "content": f"Translate the following text{src} to {target_lang}. Return ONLY the translation."},
        {"role": "user", "content": text},
    ]
    return _llm_generate(messages, thinking="off", max_tokens=1024)


@spaces.GPU(duration=120)
def gr_generate_code(task: str, language: str) -> str:
    if not task.strip():
        return "Опишите задачу."
    lang_hint = f" in {language}" if language else ""
    messages = [
        {"role": "system", "content": f"You are an expert programmer. Write clean, well-structured code{lang_hint}. Return ONLY the code."},
        {"role": "user", "content": task},
    ]
    return _llm_generate(messages, thinking="off", max_tokens=2048)


@spaces.GPU(duration=120)
def gr_summarize(file_obj, prompt: str) -> str:
    if not file_obj:
        return "Загрузите документ."
    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    with open(file_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > 8000:
        content = content[:8000] + "\n\n[... обрезано ...]"
    user_prompt = prompt or "Кратко изложите этот документ."
    messages = [{"role": "user", "content": f"{user_prompt}\n\n---\n\n{content}"}]
    return _llm_generate(messages, thinking="off", max_tokens=1024)


# ---------------------------------------------------------------------------
# Gradio UI — Nuxt SPA in iframe, API tabs as fallback
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

with gr.Blocks(
    title="OmniFF — FFmpeg для ИИ",
    css=".gradio-container{max-width:100%!important;padding:0!important}"
        " footer{display:none!important}"
        " .main .wrap{padding:0!important}"
        " .tabitem{padding:12px!important}",
) as demo:

    with gr.Tab("Верстак"):
        gr.HTML(
            '<iframe src="/omniff-spa/index.html" '
            'style="width:100%;height:85vh;border:none" '
            'allow="microphone;camera"></iframe>'
        )

    with gr.Tab("Текст"):
        with gr.Row():
            with gr.Column():
                txt_in = gr.Textbox(label="Запрос", lines=3, placeholder="Спросите что угодно...")
                txt_think = gr.Radio(["off", "normal"], value="off", label="Рассуждение")
                txt_btn = gr.Button("Генерировать", variant="primary")
            with gr.Column():
                txt_out = gr.Textbox(label="Ответ", lines=10, show_copy_button=True)
        txt_btn.click(gr_generate_text, [txt_in, txt_think], txt_out)

    with gr.Tab("Изображение"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="filepath", label="Изображение")
                img_prompt = gr.Textbox(label="Вопрос", value="Опишите изображение подробно.")
                img_btn = gr.Button("Анализ", variant="primary")
            with gr.Column():
                img_out = gr.Textbox(label="Результат", lines=10, show_copy_button=True)
        img_btn.click(gr_describe_image, [img_in, img_prompt], img_out)

    with gr.Tab("Аудио"):
        with gr.Row():
            with gr.Column():
                aud_in = gr.Audio(type="filepath", label="Аудио", sources=["upload", "microphone"])
                aud_lang = gr.Dropdown(["", "en", "ru", "kk"], value="", label="Язык")
                aud_btn = gr.Button("Распознать", variant="primary")
            with gr.Column():
                aud_out = gr.Textbox(label="Транскрипция", lines=10, show_copy_button=True)
        aud_btn.click(gr_transcribe, [aud_in, aud_lang], aud_out)

    with gr.Tab("Генерация"):
        with gr.Row():
            with gr.Column():
                gen_prompt = gr.Textbox(label="Промпт", lines=2, placeholder="Опишите изображение...")
                gen_seed = gr.Number(label="Сид", value=-1)
                gen_btn = gr.Button("Создать", variant="primary")
            with gr.Column():
                gen_out = gr.Image(label="Результат")
        gen_btn.click(gr_generate_image, [gen_prompt, gen_seed], gen_out)

    with gr.Tab("Перевод"):
        with gr.Row():
            with gr.Column():
                tr_in = gr.Textbox(label="Текст", lines=4)
                tr_src = gr.Dropdown(["", "Английский", "Русский", "Казахский"], value="", label="Исходный")
                tr_tgt = gr.Dropdown(["Русский", "Английский", "Казахский"], value="Русский", label="Целевой")
                tr_btn = gr.Button("Перевести", variant="primary")
            with gr.Column():
                tr_out = gr.Textbox(label="Перевод", lines=10, show_copy_button=True)
        tr_btn.click(gr_translate, [tr_in, tr_src, tr_tgt], tr_out)

    with gr.Tab("Код"):
        with gr.Row():
            with gr.Column():
                code_task = gr.Textbox(label="Задача", lines=3)
                code_lang = gr.Dropdown(["Python", "JavaScript", "Rust", "Go", ""], value="Python", label="Язык")
                code_btn = gr.Button("Генерировать", variant="primary")
            with gr.Column():
                code_out = gr.Code(label="Результат", language="python", lines=14)
        code_btn.click(gr_generate_code, [code_task, code_lang], code_out)


# Mount SPA static files
if os.path.isdir(STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    demo.app.mount("/omniff-spa", StaticFiles(directory=STATIC_DIR, html=True), name="spa")


# API middleware — intercept /api/* before Gradio's catch-all
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Route
from fastapi import FastAPI as _FA, File as _File, Form as _Form, UploadFile as _UF
from fastapi.responses import JSONResponse, FileResponse as _FR

_api = _FA()


@_api.post("/generate_text")
def _api_gen_text(prompt: str = _Form(...), thinking: str = _Form("off")):
    return JSONResponse({"result": gr_generate_text(prompt, thinking)})


@_api.post("/describe")
def _api_describe(file: _UF = _File(...), prompt: str = _Form("")):
    suffix = os.path.splitext(file.filename or ".jpg")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    try:
        result = gr_describe_image(tmp.name, prompt)
    finally:
        os.unlink(tmp.name)
    return JSONResponse({"result": result})


@_api.post("/transcribe")
def _api_transcribe(file: _UF = _File(...), language: str = _Form("")):
    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    try:
        result = gr_transcribe(tmp.name, language)
    finally:
        os.unlink(tmp.name)
    return JSONResponse({"result": result})


@_api.post("/generate_image")
def _api_gen_img(prompt: str = _Form(...), seed: int = _Form(-1)):
    image = gr_generate_image(prompt, seed)
    if image is None:
        return JSONResponse({"error": "Не удалось"}, status_code=500)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp.name)
    return _FR(tmp.name, media_type="image/png")


@_api.post("/translate")
def _api_translate(text: str = _Form(...), source_lang: str = _Form(""), target_lang: str = _Form("Русский")):
    return JSONResponse({"result": gr_translate(text, source_lang, target_lang)})


@_api.post("/generate_code")
def _api_gen_code(task: str = _Form(...), language: str = _Form("Python")):
    return JSONResponse({"result": gr_generate_code(task, language)})


@_api.post("/summarize")
def _api_summarize(file: _UF = _File(...), prompt: str = _Form("")):
    suffix = os.path.splitext(file.filename or ".txt")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb")
    tmp.write(file.file.read())
    tmp.close()
    try:
        result = gr_summarize(tmp.name, prompt)
    finally:
        os.unlink(tmp.name)
    return JSONResponse({"result": result})


@_api.get("/health")
def _api_health():
    return JSONResponse({"status": "ok", "models": {
        "llm": _llm["model"] is not None,
        "vlm": _vlm["model"] is not None,
        "asr": _asr["model"] is not None,
        "t2i": _t2i["pipe"] is not None,
    }})


# Mount API sub-app BEFORE Gradio's routes by inserting at position 0
demo.app.mount("/omniff-api", _api)
# Move the /api mount to the front of the route list
api_route = demo.app.routes[-1]
demo.app.routes.remove(api_route)
demo.app.routes.insert(0, api_route)

demo.launch(show_error=True)
