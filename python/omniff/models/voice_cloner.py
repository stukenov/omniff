from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class VoiceClonerModel(OmniModel):
    """Voice cloning via CosyVoice2 (Alibaba FunAudioLLM).

    Accepts reference audio + text, produces speech in cloned voice.
    Falls back to basic TTS if CosyVoice not available.
    """

    def __init__(
        self,
        model_id: str = "FunAudioLLM/CosyVoice2-0.5B",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._backend = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            self._model = CosyVoice2(self.model_id, load_jit=False, load_trt=False)
            self._backend = "cosyvoice"
        except ImportError:
            from transformers import AutoProcessor, BarkModel

            self._processor = AutoProcessor.from_pretrained("suno/bark-small")
            self._model = BarkModel.from_pretrained(
                "suno/bark-small",
                torch_dtype=torch.float16,
            )
            if device != "cpu":
                self._model = self._model.to(device)
            self._backend = "bark_fallback"

    def unload(self) -> None:
        del self._model
        self._model = None
        self._backend = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        text = inputs["text"]
        reference_audio = inputs.get("reference_audio")
        output_path = inputs.get("output_path", "cloned_voice.wav")

        if self._backend == "cosyvoice":
            return self._infer_cosyvoice(text, reference_audio, output_path)
        return self._infer_bark_fallback(text, output_path)

    def _infer_cosyvoice(
        self, text: str, reference_audio: str | None, output_path: str
    ) -> dict[str, Any]:
        import torchaudio

        if reference_audio:
            output = list(
                self._model.inference_zero_shot(text, "", reference_audio, stream=False)
            )
        else:
            output = list(self._model.inference_sft(text, "中文女", stream=False))

        if not output:
            raise RuntimeError("CosyVoice produced no output")

        audio_tensor = output[0]["tts_speech"]
        sample_rate = 22050

        torchaudio.save(output_path, audio_tensor, sample_rate)

        duration_s = audio_tensor.shape[-1] / sample_rate

        return {
            "audio_path": output_path,
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "backend": "cosyvoice",
        }

    def _infer_bark_fallback(self, text: str, output_path: str) -> dict[str, Any]:
        voice_preset = "v2/en_speaker_6"
        input_ids = self._processor(text, voice_preset=voice_preset)
        for k, v in input_ids.items():
            if hasattr(v, "to"):
                input_ids[k] = v.to(self._model.device)

        speech_output = self._model.generate(**input_ids)
        audio_array = speech_output[0].cpu().float().numpy().squeeze()
        sample_rate = self._model.generation_config.sample_rate

        import scipy.io.wavfile

        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array)

        return {
            "audio_path": output_path,
            "sample_rate": sample_rate,
            "duration_s": len(audio_array) / sample_rate,
            "backend": "bark_fallback",
        }
