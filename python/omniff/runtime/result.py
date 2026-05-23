from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    output_text: str | None = None
    output_path: str | None = None
    route: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        if self.output_text is not None:
            Path(path).write_text(self.output_text, encoding="utf-8")
        elif self.output_path is not None:
            import shutil

            shutil.copy2(self.output_path, path)
        else:
            raise ValueError("No output to save")

    def __repr__(self) -> str:
        if self.output_text:
            preview = self.output_text[:100] + ("..." if len(self.output_text) > 100 else "")
            return f"RunResult(text={preview!r}, route={self.route})"
        return f"RunResult(path={self.output_path!r}, route={self.route})"
