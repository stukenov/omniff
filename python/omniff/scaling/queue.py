from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class QueueItem:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.monotonic)
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    done: bool = False


class RequestQueue:
    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._items: list[QueueItem] = []
        self._processed = 0
        self._total_wait_ms = 0.0

    def enqueue(self, payload: dict[str, Any], priority: Priority = Priority.NORMAL) -> QueueItem:
        if len(self._items) >= self.max_size:
            raise RuntimeError(f"Queue full ({self.max_size} items)")
        item = QueueItem(priority=priority, payload=payload)
        self._items.append(item)
        self._items.sort(key=lambda x: (-x.priority, x.created_at))
        return item

    def dequeue(self) -> QueueItem | None:
        if not self._items:
            return None
        item = self._items.pop(0)
        wait = (time.monotonic() - item.created_at) * 1000
        self._total_wait_ms += wait
        self._processed += 1
        return item

    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def stats(self) -> dict[str, Any]:
        avg_wait = self._total_wait_ms / max(self._processed, 1)
        return {
            "queue_size": self.size(),
            "processed": self._processed,
            "avg_wait_ms": round(avg_wait, 2),
        }
