"""
OmniFF FastAPI Backend — KazNU Server (2x A10 22GB)

GPU0: Qwen3.6-35B-A3B (4-bit), Qwen2.5-VL-3B (4-bit)
GPU1: whisper-large-v3-turbo (fp16), Z-Image-Turbo (fp16)
"""

import os
import re
import tempfile

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="OmniFF Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "Qwen/Qwen3.6-35B-A3B"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    _llm["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
    _llm["model"] = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map={"": "cuda:0"}
    )


def _load_vlm():
    if _vlm["model"] is not None:
        return
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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


def _load_t2i():
    if _t2i["pipe"] is not None:
        return
    from diffusers import DiffusionPipeline

    _t2i["pipe"] = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float16
    ).to("cuda:1")


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
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_think(response) or "Ответ не сгенерирован."


# --- API Endpoints ---

@app.post("/api/describe")
async def describe_image(file: UploadFile = File(...), prompt: str = Form("")):
    _load_vlm()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".jpg")[1])
    tmp.write(await file.read())
    tmp.close()

    from qwen_vl_utils import process_vision_info

    model = _vlm["model"]
    processor = _vlm["processor"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": tmp.name},
        {"type": "text", "text": prompt or "Опишите это изображение подробно."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    result = processor.decode(new_tokens, skip_special_tokens=True).strip()
    os.unlink(tmp.name)
    return {"result": result or "Ответ не сгенерирован."}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("")):
    _load_asr()
    import numpy as np
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".wav")[1])
    tmp.write(await file.read())
    tmp.close()

    audio_data, sr = sf.read(tmp.name, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
        sr = 16000

    model = _asr["model"]
    processor = _asr["processor"]
    input_features = processor(
        audio_data, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device=model.device, dtype=model.dtype)

    gen_kwargs = {}
    if language:
        gen_kwargs["language"] = language

    predicted_ids = model.generate(input_features, **gen_kwargs)
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    os.unlink(tmp.name)
    return {"result": result or "Ответ не сгенерирован."}


@app.post("/api/generate_text")
async def generate_text(prompt: str = Form(...), thinking: str = Form("off")):
    messages = [{"role": "user", "content": prompt}]
    return {"result": _llm_generate(messages, thinking)}


@app.post("/api/translate")
async def translate_text(text: str = Form(...), source_lang: str = Form(""), target_lang: str = Form("Русский")):
    src_hint = f" from {source_lang}" if source_lang else ""
    messages = [
        {"role": "system", "content": f"Translate the following text{src_hint} to {target_lang}. Return ONLY the translation."},
        {"role": "user", "content": text},
    ]
    return {"result": _llm_generate(messages, thinking="off", max_tokens=1024)}


@app.post("/api/generate_image")
async def generate_image(prompt: str = Form(...), seed: int = Form(-1)):
    _load_t2i()
    pipe = _t2i["pipe"]
    generator = torch.Generator("cuda:1")
    if seed >= 0:
        generator.manual_seed(seed)
    image = pipe(prompt=prompt, num_inference_steps=4, generator=generator).images[0]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp.name)
    return FileResponse(tmp.name, media_type="image/png")


@app.post("/api/generate_code")
async def generate_code(task: str = Form(...), language: str = Form("Python")):
    lang_hint = f" in {language}" if language else ""
    messages = [
        {"role": "system", "content": f"You are an expert programmer. Write clean, well-structured code{lang_hint}. Return ONLY the code, no explanations unless asked."},
        {"role": "user", "content": task},
    ]
    return {"result": _llm_generate(messages, thinking="off", max_tokens=2048)}


@app.post("/api/summarize")
async def summarize(file: UploadFile = File(...), prompt: str = Form("")):
    content = (await file.read()).decode("utf-8", errors="replace")
    if len(content) > 8000:
        content = content[:8000] + "\n\n[... обрезано ...]"
    user_prompt = prompt or "Кратко изложите этот документ."
    messages = [{"role": "user", "content": f"{user_prompt}\n\n---\n\n{content}"}]
    return {"result": _llm_generate(messages, thinking="off", max_tokens=1024)}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "llm": _llm["model"] is not None,
            "vlm": _vlm["model"] is not None,
            "asr": _asr["model"] is not None,
            "t2i": _t2i["pipe"] is not None,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
