from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class ASRModel(OmniModel):
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "auto",
        language: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        device = self.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16,
        ).to(device)
        self._device = device

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        import numpy as np
        import soundfile as sf
        import torch

        audio_path = inputs.get("audio_path", "")
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            audio_data = resample(audio_data, int(len(audio_data) * 16000 / sr)).astype(np.float32)
            sr = 16000

        input_features = self._processor(
            audio_data, sampling_rate=sr, return_tensors="pt",
        ).input_features.to(self._device, dtype=torch.float16)

        generate_kwargs = {}
        lang = inputs.get("language") or self.language
        if lang:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=lang, task="transcribe",
            )
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        predicted_ids = self._model.generate(
            input_features, return_timestamps=True, **generate_kwargs,
        )
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {
            "text": text.strip(),
            "chunks": [],
        }
