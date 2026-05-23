from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel

MODEL_SDXL_TURBO = "stabilityai/sdxl-turbo"
MODEL_Z_IMAGE = "Tongyi-MAI/Z-Image-Turbo"


class TextToImageModel(OmniModel):
    def __init__(
        self,
        model_id: str = MODEL_Z_IMAGE,
        device: str = "auto",
        num_inference_steps: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self._pipe = None
        self._backend = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        import torch

        if "z-image" in self.model_id.lower() or "tongyi" in self.model_id.lower():
            self._load_z_image()
        else:
            self._load_sdxl()

    def _load_sdxl(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        device = "cuda" if self.device == "auto" else self.device
        self._pipe = self._pipe.to(device)
        self._backend = "sdxl"

    def _load_z_image(self) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self._pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        device = "cuda" if self.device == "auto" else self.device
        self._pipe = self._pipe.to(device)
        self._backend = "z_image"

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        self._backend = None

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

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=self.num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        )

        if self._backend == "sdxl":
            kwargs["guidance_scale"] = 0.0

        image = self._pipe(**kwargs).images[0]
        image.save(output_path)

        return {
            "image_path": output_path,
            "backend": self._backend,
        }
