from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel

LANG_MAP = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "ru": "rus_Cyrl",
    "russian": "rus_Cyrl",
    "kk": "kaz_Cyrl",
    "kazakh": "kaz_Cyrl",
    "zh": "zho_Hans",
    "chinese": "zho_Hans",
    "fr": "fra_Latn",
    "french": "fra_Latn",
    "de": "deu_Latn",
    "german": "deu_Latn",
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    "ja": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    "ko": "kor_Hang",
    "korean": "kor_Hang",
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
}


def _resolve_lang(lang: str) -> str:
    stripped = lang.strip()
    if "_" in stripped and len(stripped) == 8:
        return stripped
    lang_lower = stripped.lower()
    return LANG_MAP.get(lang_lower, "eng_Latn")


class TranslatorModel(OmniModel):
    """NLLB-200 translation model — 200 languages, dedicated seq2seq."""

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-1.3B",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        self._device = device

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        text = inputs["text"]
        source_lang = _resolve_lang(inputs.get("source_language", "en"))
        target_lang = _resolve_lang(inputs.get("target_language", "en"))

        self._tokenizer.src_lang = source_lang
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        target_lang_id = self._tokenizer.convert_tokens_to_ids(target_lang)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=1024,
        )

        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return {
            "text": translated,
            "source_language": source_lang,
            "target_language": target_lang,
        }
