from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileEntry:
    name: str
    duration_ms: float
    depth: int = 0


@dataclass
class LatencyProfiler:
    entries: list[ProfileEntry] = field(default_factory=list)
    _stack: list[tuple[str, float, int]] = field(default_factory=list)

    def start(self, name: str) -> None:
        depth = len(self._stack)
        self._stack.append((name, time.monotonic(), depth))

    def stop(self) -> None:
        if not self._stack:
            return
        name, start, depth = self._stack.pop()
        duration = (time.monotonic() - start) * 1000
        self.entries.append(ProfileEntry(name=name, duration_ms=round(duration, 2), depth=depth))

    def render(self) -> str:
        if not self.entries:
            return "(no profile data)"
        lines = []
        total = sum(e.duration_ms for e in self.entries if e.depth == 0)
        for e in self.entries:
            indent = "  " * e.depth
            pct = (e.duration_ms / total * 100) if total > 0 else 0
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            lines.append(f"{indent}{e.name}: {e.duration_ms:.1f}ms ({pct:.0f}%) {bar}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {"name": e.name, "duration_ms": e.duration_ms, "depth": e.depth} for e in self.entries
        ]
