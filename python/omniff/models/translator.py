from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel

NLLB_LANG_MAP = {
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

LANG_MAP = NLLB_LANG_MAP

HYMT_LANG_MAP = {
    "en": "English",
    "english": "English",
    "ru": "Russian",
    "russian": "Russian",
    "kk": "Kazakh",
    "kazakh": "Kazakh",
    "zh": "Chinese",
    "chinese": "Chinese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "ar": "Arabic",
    "arabic": "Arabic",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "tr": "Turkish",
    "turkish": "Turkish",
}


def _resolve_lang(lang: str) -> str:
    stripped = lang.strip()
    if "_" in stripped and len(stripped) == 8:
        return stripped
    lang_lower = stripped.lower()
    return NLLB_LANG_MAP.get(lang_lower, "eng_Latn")


def _resolve_lang_hymt(lang: str) -> str:
    lang_lower = lang.strip().lower()
    return HYMT_LANG_MAP.get(lang_lower, lang.strip().capitalize())


# Default model IDs
MODEL_NLLB = "facebook/nllb-200-distilled-1.3B"
MODEL_HYMT2 = "tencent/Hy-MT2-30B-A3B"


class TranslatorModel(OmniModel):
    """Multi-backend translator: NLLB-200 (seq2seq) or Tencent HY-MT2 (causal LM)."""

    def __init__(
        self,
        model_id: str = MODEL_NLLB,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
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

        if "hy-mt" in self.model_id.lower() or "hymt" in self.model_id.lower():
            self._load_hymt(device)
        else:
            self._load_nllb(device)

    def _load_nllb(self, device: str) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        self._backend = "nllb"

    def _load_hymt(self, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self._backend = "hymt"

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._backend = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if self._backend == "hymt":
            return self._infer_hymt(inputs)
        return self._infer_nllb(inputs)

    def _infer_nllb(self, inputs: dict[str, Any]) -> dict[str, Any]:
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
            "backend": "nllb",
        }

    def _infer_hymt(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs["text"]
        source_lang = _resolve_lang_hymt(inputs.get("source_language", "en"))
        target_lang = _resolve_lang_hymt(inputs.get("target_language", "en"))

        prompt = f"Translate the following {source_lang} text to {target_lang}.\n{source_lang}: {text}\n{target_lang}:"
        messages = [{"role": "user", "content": prompt}]

        chat_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer([chat_text], return_tensors="pt").to(self._model.device)

        generated = self._model.generate(**model_inputs, max_new_tokens=1024)
        new_tokens = generated[0][model_inputs["input_ids"].shape[-1]:]
        translated = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "text": translated,
            "source_language": source_lang,
            "target_language": target_lang,
            "backend": "hymt",
        }
