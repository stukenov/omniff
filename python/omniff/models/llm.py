from __future__ import annotations

import re
from typing import Any, Generator

from omniff.models.base import OmniModel

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


class LLMModel(OmniModel):
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
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
        thinking = inputs.get("thinking", True)
        messages = inputs.get("messages") or [{"role": "user", "content": prompt}]

        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if not thinking:
            chat_kwargs["enable_thinking"] = False
        text = self._tokenizer.apply_chat_template(messages, **chat_kwargs)
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        new_tokens = generated[0][model_inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = _THINK_RE.sub("", response)
        response = _THINK_UNCLOSED_RE.sub("", response).strip()

        return {"text": response}

    def infer_stream(self, inputs: dict[str, Any]) -> Generator[str, None, None]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        from transformers import TextIteratorStreamer
        import threading

        prompt = inputs.get("prompt", "")
        thinking = inputs.get("thinking", True)
        messages = inputs.get("messages") or [{"role": "user", "content": prompt}]

        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if not thinking:
            chat_kwargs["enable_thinking"] = False
        text = self._tokenizer.apply_chat_template(messages, **chat_kwargs)
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        in_think = False
        for token in streamer:
            if "<think>" in token:
                in_think = True
                continue
            if "</think>" in token:
                in_think = False
                continue
            if not in_think and token:
                yield token

        thread.join()
