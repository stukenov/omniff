"""
OmniFF — FFmpeg for AI | HuggingFace Space (ZeroGPU)

Multimodal runtime: text→text, image→text, audio→text, text→image
"""

import os
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
_img = {"pipe": None}


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


def _load_img():
    if _img["pipe"] is not None:
        return
    from diffusers import AutoPipelineForText2Image

    _img["pipe"] = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")


# ---------------------------------------------------------------------------
# Inference functions with ZeroGPU decorator
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def process_text(text: str, thinking: str) -> str:
    if not text.strip():
        return "Please enter some text."

    _load_llm()
    model = _llm["model"]
    tokenizer = _llm["tokenizer"]

    messages = [{"role": "user", "content": text}]
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if thinking == "off":
        chat_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    max_tokens = 2048 if thinking != "off" else 512
    generated = model.generate(**inputs, max_new_tokens=max_tokens)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    import re
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
    return response.strip() or "No output generated."


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


@spaces.GPU(duration=60)
def generate_image(prompt: str, seed: int) -> str | None:
    if not prompt.strip():
        return None

    _load_img()
    pipe = _img["pipe"]

    generator = torch.Generator("cuda")
    if seed >= 0:
        generator.manual_seed(seed)

    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator).images[0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        return f.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="OmniFF — FFmpeg for AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎬 OmniFF — FFmpeg for AI\n"
        "Universal multimodal runtime. Select a tab to try different pipelines.\n\n"
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

    gr.Markdown(
        "---\n"
        "**[GitHub](https://github.com/stukenov/omniff)** · "
        "**[HuggingFace Repo](https://huggingface.co/stukenov/omniff)** · "
        "Built by [Saken Tukenov](https://github.com/stukenov)"
    )

demo.launch()
