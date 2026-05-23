"""
OmniFF — FFmpeg for AI | HuggingFace Space (ZeroGPU)

All pipelines: text→text, image→text, audio→text, text→image,
image→image, video→text, document→text, code generation
"""

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
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_think(response) or "No output generated."


# ---------------------------------------------------------------------------
# 1. Text → Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_text(text: str, thinking: str) -> str:
    if not text.strip():
        return "Please enter some text."
    return _llm_generate([{"role": "user", "content": text}], thinking)


# ---------------------------------------------------------------------------
# 2. Image → Text
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
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "No output generated."


# ---------------------------------------------------------------------------
# 3. Audio → Text
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
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip() or "No output generated."


# ---------------------------------------------------------------------------
# 4. Text → Image
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def generate_image(prompt: str, seed: int) -> str | None:
    if not prompt.strip():
        return None

    _load_t2i()
    pipe = _t2i["pipe"]

    generator = torch.Generator("cuda")
    if seed >= 0:
        generator.manual_seed(int(seed))

    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator).images[0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        return f.name


# ---------------------------------------------------------------------------
# 5. Image → Image
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def edit_image(image_path: str, prompt: str, strength: float) -> str | None:
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

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        return f.name


# ---------------------------------------------------------------------------
# 6. Video → Text
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
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": prompt or "Describe what happens in this video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip() or "No output generated."


# ---------------------------------------------------------------------------
# 7. Document → Text
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_document(file_path: str, prompt: str) -> str:
    if not file_path:
        return "Please upload a document."

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
    messages = [
        {"role": "user", "content": f"{user_prompt}\n\n---\n\n{text_content}"}
    ]
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
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="OmniFF — FFmpeg for AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎬 OmniFF — FFmpeg for AI\n"
        "Universal multimodal runtime — 8 pipelines. Select a tab.\n\n"
        "*Running on ZeroGPU — first call loads the model (may take 30-60s).*"
    )

    with gr.Tab("💬 Text → Text"):
        with gr.Row():
            with gr.Column():
                txt_input = gr.Textbox(label="Input", lines=3, placeholder="Ask anything...")
                txt_thinking = gr.Radio(
                    ["off", "normal"],
                    value="off",
                    label="Thinking mode",
                )
                txt_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                txt_output = gr.Textbox(label="Output", lines=8)
        txt_btn.click(process_text, [txt_input, txt_thinking], txt_output)

    with gr.Tab("🖼️ Image → Text"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="filepath", label="Upload image")
                img_prompt = gr.Textbox(
                    label="Question about image", value="Describe this image in detail."
                )
                img_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                img_output = gr.Textbox(label="Description", lines=8)
        img_btn.click(process_image, [img_input, img_prompt], img_output)

    with gr.Tab("🎤 Audio → Text"):
        with gr.Row():
            with gr.Column():
                aud_input = gr.Audio(type="filepath", label="Upload audio")
                aud_lang = gr.Dropdown(
                    ["", "en", "ru", "kk", "zh", "de", "fr", "es", "ja"],
                    value="",
                    label="Language (auto if empty)",
                )
                aud_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                aud_output = gr.Textbox(label="Transcription", lines=8)
        aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

    with gr.Tab("🎨 Text → Image"):
        with gr.Row():
            with gr.Column():
                gen_prompt = gr.Textbox(
                    label="Prompt", lines=2, placeholder="A cyberpunk city at night..."
                )
                gen_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                gen_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                gen_output = gr.Image(label="Generated image")
        gen_btn.click(generate_image, [gen_prompt, gen_seed], gen_output)

    with gr.Tab("🎭 Image → Image"):
        with gr.Row():
            with gr.Column():
                i2i_input = gr.Image(type="filepath", label="Upload image")
                i2i_prompt = gr.Textbox(
                    label="Edit prompt", value="Make it look like a watercolor painting"
                )
                i2i_strength = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Edit strength (higher = more change)"
                )
                i2i_btn = gr.Button("Transform", variant="primary")
            with gr.Column():
                i2i_output = gr.Image(label="Transformed image")
        i2i_btn.click(edit_image, [i2i_input, i2i_prompt, i2i_strength], i2i_output)

    with gr.Tab("🎬 Video → Text"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Upload video")
                vid_prompt = gr.Textbox(
                    label="Question about video",
                    value="Describe what happens in this video."
                )
                vid_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                vid_output = gr.Textbox(label="Description", lines=8)
        vid_btn.click(process_video, [vid_input, vid_prompt], vid_output)

    with gr.Tab("📄 Document → Text"):
        with gr.Row():
            with gr.Column():
                doc_input = gr.File(
                    label="Upload document (txt, md, csv, json, pdf)",
                    file_types=[".txt", ".md", ".csv", ".json", ".xml", ".html", ".log", ".pdf"]
                )
                doc_prompt = gr.Textbox(
                    label="Instruction",
                    value="Summarize this document."
                )
                doc_btn = gr.Button("Process", variant="primary")
            with gr.Column():
                doc_output = gr.Textbox(label="Result", lines=10)
        doc_btn.click(process_document, [doc_input, doc_prompt], doc_output)

    with gr.Tab("💻 Code Generation"):
        with gr.Row():
            with gr.Column():
                code_task = gr.Textbox(
                    label="Task", lines=3,
                    placeholder="Write a function that sorts a list using merge sort"
                )
                code_lang = gr.Dropdown(
                    ["Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++", ""],
                    value="Python",
                    label="Language",
                )
                code_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                code_output = gr.Code(label="Generated code", language="python", lines=20)
        code_btn.click(generate_code, [code_task, code_lang], code_output)

    gr.Markdown(
        "---\n"
        "**8 pipelines** · Text→Text · Image→Text · Audio→Text · Text→Image · "
        "Image→Image · Video→Text · Document→Text · Code Generation\n\n"
        "**[GitHub](https://github.com/stukenov/omniff)** · "
        "**[HuggingFace Repo](https://huggingface.co/stukenov/omniff)** · "
        "Built by [Saken Tukenov](https://github.com/stukenov)"
    )

demo.launch()
