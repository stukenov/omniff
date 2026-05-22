from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    passed: bool
    score: float
    validator: str
    details: str | None = None


class TextValidator:
    def __init__(self, min_length: int = 0, max_length: int | None = None) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        text = output.get("text")
        if text is None:
            return ValidationResult(False, 0.0, "text", "no text in output")
        if len(text) < self.min_length:
            return ValidationResult(False, 0.1, "text", f"too short: {len(text)} < {self.min_length}")
        if self.max_length and len(text) > self.max_length:
            return ValidationResult(False, 0.2, "text", f"too long: {len(text)} > {self.max_length}")
        return ValidationResult(True, 1.0, "text")
