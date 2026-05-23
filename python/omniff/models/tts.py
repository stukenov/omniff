from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.models.base import OmniModel


class TTSModel(OmniModel):
    def __init__(
        self,
        model_id: str = "suno/bark-small",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import AutoProcessor, BarkModel
        import torch

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = BarkModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        if self.device == "auto" and torch.cuda.is_available():
            self._model = self._model.to("cuda")

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        text = inputs["text"]
        output_path = inputs.get("output_path", "output.wav")
        voice_preset = inputs.get("voice_preset", "v2/en_speaker_6")

        input_ids = self._processor(text, voice_preset=voice_preset)
        for k, v in input_ids.items():
            if hasattr(v, "to"):
                input_ids[k] = v.to(self._model.device)

        speech_output = self._model.generate(**input_ids)
        audio_array = speech_output[0].cpu().numpy().squeeze()

        sample_rate = self._model.generation_config.sample_rate

        import scipy.io.wavfile
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array)

        return {
            "audio_path": output_path,
            "sample_rate": sample_rate,
            "duration_s": len(audio_array) / sample_rate,
        }
