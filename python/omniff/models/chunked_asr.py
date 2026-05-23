from __future__ import annotations

from collections.abc import Generator
from typing import Any

from omniff.models.base import OmniModel


class ChunkedASRModel(OmniModel):
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "auto",
        chunk_length_s: float = 30.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.chunk_length_s = chunk_length_s
        self._processor = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else "cpu"
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to(device)

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        chunks = list(self.infer_stream(inputs))
        return {"text": " ".join(chunks).strip(), "chunks": chunks}

    def infer_stream(self, inputs: dict[str, Any]) -> Generator[str, None, None]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        import numpy as np
        import soundfile as sf

        audio_path = inputs["audio_path"]
        audio_data, sr = sf.read(audio_path, dtype="float32")

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if sr != 16000:
            from scipy.signal import resample

            num_samples = int(len(audio_data) * 16000 / sr)
            audio_data = resample(audio_data, num_samples).astype(np.float32)
            sr = 16000

        chunk_samples = int(self.chunk_length_s * sr)

        for start in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[start : start + chunk_samples]
            if len(chunk) < sr * 0.5:
                continue

            input_features = self._processor(
                chunk, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(device=self._model.device, dtype=self._model.dtype)

            predicted_ids = self._model.generate(input_features)
            text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if text.strip():
                yield text.strip()
