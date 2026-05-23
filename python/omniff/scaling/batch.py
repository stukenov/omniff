from __future__ import annotations

from typing import Any


class BatchInference:
    def __init__(self, max_batch_size: int = 8) -> None:
        self.max_batch_size = max_batch_size
        self._pending: list[dict[str, Any]] = []

    def add(self, request: dict[str, Any]) -> int:
        idx = len(self._pending)
        self._pending.append(request)
        return idx

    def is_full(self) -> bool:
        return len(self._pending) >= self.max_batch_size

    def size(self) -> int:
        return len(self._pending)

    def flush(self, model: Any) -> list[dict[str, Any]]:
        if not self._pending:
            return []

        if not hasattr(model, "infer_batch"):
            results = [model.infer(req) for req in self._pending]
        else:
            results = model.infer_batch(self._pending)

        self._pending = []
        return results

    def clear(self) -> None:
        self._pending = []
