"""
Gradio demo for OmniFF — deploy as HuggingFace Space.

Usage:
  python -m omniff.demo

Or deploy to HF Spaces:
  1. huggingface-cli repo create omniff-runtime --type space --space-sdk gradio
  2. Copy this file as app.py + omniff package
  3. git push
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def create_demo():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio required: pip install gradio")

    from omniff.runtime.config import OmniFFConfig, RouterConfig
    from omniff.runtime.engine import OmniFFRuntime

    config = OmniFFConfig(
        name="omniff",
        version="1.0",
        router=RouterConfig(router_type="keyword", path=""),
    )
    runtime = OmniFFRuntime(config)

    def process_text(text: str, prompt: str, thinking: str) -> str:
        if not text.strip():
            return "Please enter some text."
        result = runtime.run(
            input=text,
            prompt=prompt or None,
            thinking=thinking,
        )
        return result.output_text or "No output generated."

    def process_image(image_path: str, prompt: str) -> str:
        if not image_path:
            return "Please upload an image."
        result = runtime.run(
            input=image_path,
            prompt=prompt or "Describe this image in detail.",
        )
        return result.output_text or "No output generated."

    def process_audio(audio_path: str, language: str) -> str:
        if not audio_path:
            return "Please upload an audio file."
        result = runtime.run(
            input=audio_path,
            controls={"language": language} if language else {},
        )
        return result.output_text or "No output generated."

    def generate_image(prompt: str, seed: int) -> str | None:
        if not prompt.strip():
            return None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name
        result = runtime.run(
            input=prompt,
            output_modality="image",
            output=output_path,
            controls={"seed": seed} if seed >= 0 else {},
        )
        return result.output_path

    with gr.Blocks(title="OmniFF — FFmpeg for AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 OmniFF — FFmpeg for AI\n"
            "Universal multimodal runtime. Select a tab to try different pipelines."
        )

        with gr.Tab("💬 Text → Text"):
            with gr.Row():
                with gr.Column():
                    txt_input = gr.Textbox(label="Input", lines=3, placeholder="Ask anything...")
                    txt_prompt = gr.Textbox(label="System prompt (optional)", lines=1)
                    txt_thinking = gr.Radio(
                        ["off", "fast", "normal", "deep"],
                        value="normal", label="Thinking level",
                    )
                    txt_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    txt_output = gr.Textbox(label="Output", lines=8)
            txt_btn.click(process_text, [txt_input, txt_prompt, txt_thinking], txt_output)

        with gr.Tab("🖼️ Image → Text"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="filepath", label="Upload image")
                    img_prompt = gr.Textbox(label="Question about image", value="Describe this image in detail.")
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
                        value="", label="Language (auto if empty)",
                    )
                    aud_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    aud_output = gr.Textbox(label="Transcription", lines=8)
            aud_btn.click(process_audio, [aud_input, aud_lang], aud_output)

        with gr.Tab("🎨 Text → Image"):
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="A cyberpunk city at night...")
                    gen_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    gen_output = gr.Image(label="Generated image")
            gen_btn.click(generate_image, [gen_prompt, gen_seed], gen_output)

    return demo


def main():
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
