from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class TextToImageModel(OmniModel):
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        device: str = "auto",
        num_inference_steps: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self._pipe = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        if self.device == "auto":
            self._pipe = self._pipe.to("cuda")
        else:
            self._pipe = self._pipe.to(self.device)

    def unload(self) -> None:
        del self._pipe
        self._pipe = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        seed = inputs.get("seed")
        output_path = inputs.get("output_path", "output.png")
        width = inputs.get("width", 512)
        height = inputs.get("height", 512)

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self._pipe.device).manual_seed(seed)

        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=self.num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            guidance_scale=0.0,
        ).images[0]

        image.save(output_path)
        return {"image_path": output_path}
