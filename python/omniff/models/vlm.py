from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class VLMModel(OmniModel):
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device,
        )

    def unload(self) -> None:
        del self._model, self._processor
        self._model = None
        self._processor = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        from qwen_vl_utils import process_vision_info

        image_path = inputs.get("image_path", "")
        prompt = inputs.get("prompt", "Describe this image.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[0][model_inputs["input_ids"].shape[-1]:]
        response = self._processor.decode(trimmed, skip_special_tokens=True)

        return {"text": response}
