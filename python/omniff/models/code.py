from __future__ import annotations

import re
from typing import Any

from omniff.models.base import OmniModel

_CODE_PATTERNS = [
    re.compile(r"```\w*\n"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\bconst\s+\w+\s*="),
    re.compile(r"#include\s*<"),
    re.compile(r"\bfn\s+\w+\s*\("),
    re.compile(r"\bpub\s+(fn|struct|enum)"),
]

_CODE_KEYWORDS = [
    "refactor", "debug", "compile", "syntax error", "function",
    "variable", "algorithm", "implement", "code review",
    "fix this code", "write a function", "write code",
]


def is_code_request(text: str) -> bool:
    for pattern in _CODE_PATTERNS:
        if pattern.search(text):
            return True
    text_lower = text.lower()
    matches = sum(1 for kw in _CODE_KEYWORDS if kw in text_lower)
    return matches >= 2


class CodeModel(OmniModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device,
        )

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        prompt = inputs.get("prompt", "")
        system_msg = (
            "You are a coding assistant. Respond with clean, well-structured code. "
            "Include brief comments only where the logic is non-obvious. "
            "Use proper formatting and idiomatic style for the language."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        new_tokens = generated[0][model_inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        import re as _re
        response = _re.sub(r"<think>.*?</think>\s*", "", response, flags=_re.DOTALL)
        response = _re.sub(r"<think>.*", "", response, flags=_re.DOTALL)

        return {"text": response.strip()}
