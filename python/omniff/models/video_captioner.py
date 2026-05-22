from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class VideoCaptionerModel(OmniModel):
    """Video→text using Qwen2.5-VL. Extracts keyframes and captions them."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 512,
        max_frames: int = 8,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    def _extract_frames(self, video_path: str) -> list:
        import cv2
        from PIL import Image
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, min(self.max_frames, total_frames), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        import torch

        video_path = inputs.get("video_path", "")
        prompt = inputs.get("prompt", "Describe this video in detail.")

        frames = self._extract_frames(video_path)
        if not frames:
            return {"text": "Could not extract frames from video."}

        image_content = []
        for frame in frames:
            image_content.append({"type": "image", "image": frame})

        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt}],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)

        trimmed = generated[0][model_inputs["input_ids"].shape[1]:]
        response = self._processor.decode(trimmed, skip_special_tokens=True)
        return {"text": response.strip(), "num_frames": len(frames)}
