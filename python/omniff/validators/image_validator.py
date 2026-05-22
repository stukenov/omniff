from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.validators.text_validator import ValidationResult


class ImageValidator:
    def __init__(self, min_size: int = 10) -> None:
        self.min_size = min_size

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        image_path = output.get("image_path")
        if not image_path:
            return ValidationResult(False, 0.0, "image", "no image_path in output")
        p = Path(image_path)
        if not p.exists():
            return ValidationResult(False, 0.0, "image", f"file not found: {image_path}")
        if p.stat().st_size < self.min_size:
            return ValidationResult(False, 0.1, "image", "file too small")
        return ValidationResult(True, 1.0, "image")
