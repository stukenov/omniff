from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class Span:
    name: str
    start_ns: int = 0
    end_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000


@dataclass
class RequestTrace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)
    start_ns: int = field(default_factory=time.monotonic_ns)

    @contextmanager
    def span(self, name: str, **meta: Any) -> Generator[Span, None, None]:
        s = Span(name=name, start_ns=time.monotonic_ns(), metadata=meta)
        try:
            yield s
        finally:
            s.end_ns = time.monotonic_ns()
            self.spans.append(s)

    @property
    def total_ms(self) -> float:
        return (time.monotonic_ns() - self.start_ns) / 1_000_000

    def timing_breakdown(self) -> dict[str, float]:
        return {s.name: round(s.duration_ms, 2) for s in self.spans}


_logger = logging.getLogger("omniff")


def get_logger(component: str | None = None) -> logging.Logger:
    name = f"omniff.{component}" if component else "omniff"
    return logging.getLogger(name)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("omniff")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        if json_output:
            import json as _json

            class JSONFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    data = {
                        "ts": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "msg": record.getMessage(),
                    }
                    if hasattr(record, "request_id"):
                        data["request_id"] = record.request_id
                    return _json.dumps(data)

            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
        logger.addHandler(handler)
