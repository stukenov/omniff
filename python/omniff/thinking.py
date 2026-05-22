from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThinkingLevel(Enum):
    OFF = "off"
    FAST = "fast"
    NORMAL = "normal"
    DEEP = "deep"
    RESEARCH = "research"


@dataclass
class PromptControl:
    thinking: ThinkingLevel = ThinkingLevel.NORMAL
    max_tokens: int = 512
    temperature: float = 0.7
    language: str | None = None

    @classmethod
    def from_level(cls, level: str) -> PromptControl:
        thinking = ThinkingLevel(level) if level in [e.value for e in ThinkingLevel] else ThinkingLevel.NORMAL

        presets = {
            ThinkingLevel.OFF: {"max_tokens": 256, "temperature": 0.3},
            ThinkingLevel.FAST: {"max_tokens": 256, "temperature": 0.5},
            ThinkingLevel.NORMAL: {"max_tokens": 512, "temperature": 0.7},
            ThinkingLevel.DEEP: {"max_tokens": 1024, "temperature": 0.8},
            ThinkingLevel.RESEARCH: {"max_tokens": 2048, "temperature": 0.9},
        }

        p = presets[thinking]
        return cls(thinking=thinking, max_tokens=p["max_tokens"], temperature=p["temperature"])

    @property
    def enable_model_thinking(self) -> bool:
        return self.thinking not in (ThinkingLevel.OFF, ThinkingLevel.FAST)
