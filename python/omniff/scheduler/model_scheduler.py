from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from omniff.models.base import OmniModel


class LoadPolicy(Enum):
    HOT = "hot"  # always loaded, never evicted
    WARM = "warm"  # loaded on demand, evicted after TTL
    COLD = "cold"  # loaded on demand, evicted immediately after use


@dataclass
class ModelSlot:
    name: str
    model: OmniModel
    policy: LoadPolicy
    last_used: float = 0.0
    use_count: int = 0
    ttl_seconds: float = 300.0


class ModelScheduler:
    def __init__(self, max_loaded: int = 4, default_ttl: float = 300.0) -> None:
        self._slots: dict[str, ModelSlot] = {}
        self.max_loaded = max_loaded
        self.default_ttl = default_ttl

    def register(
        self,
        name: str,
        model: OmniModel,
        policy: LoadPolicy = LoadPolicy.WARM,
        ttl: float | None = None,
    ) -> None:
        self._slots[name] = ModelSlot(
            name=name,
            model=model,
            policy=policy,
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
        )

    def acquire(self, name: str) -> OmniModel:
        if name not in self._slots:
            raise KeyError(f"Model not registered: {name}")

        slot = self._slots[name]
        if not slot.model.is_loaded:
            self._evict_if_needed()
            slot.model.load()

        slot.last_used = time.monotonic()
        slot.use_count += 1
        return slot.model

    def release(self, name: str) -> None:
        if name not in self._slots:
            return
        slot = self._slots[name]
        if slot.policy == LoadPolicy.COLD and slot.model.is_loaded:
            slot.model.unload()

    def evict_expired(self) -> list[str]:
        now = time.monotonic()
        evicted = []
        for name, slot in self._slots.items():
            if slot.policy == LoadPolicy.HOT:
                continue
            if not slot.model.is_loaded:
                continue
            if slot.policy == LoadPolicy.WARM and slot.last_used > 0:
                if (now - slot.last_used) > slot.ttl_seconds:
                    slot.model.unload()
                    evicted.append(name)
        return evicted

    def _evict_if_needed(self) -> None:
        loaded = [s for s in self._slots.values() if s.model.is_loaded]
        if len(loaded) < self.max_loaded:
            return

        candidates = [s for s in loaded if s.policy != LoadPolicy.HOT]
        if not candidates:
            return

        candidates.sort(key=lambda s: s.last_used)
        candidates[0].model.unload()

    def loaded_models(self) -> list[str]:
        return [n for n, s in self._slots.items() if s.model.is_loaded]

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "loaded": slot.model.is_loaded,
                "policy": slot.policy.value,
                "use_count": slot.use_count,
                "ttl": slot.ttl_seconds,
            }
            for name, slot in self._slots.items()
        }

    def has(self, name: str) -> bool:
        return name in self._slots

    def unload_all(self) -> None:
        for slot in self._slots.values():
            if slot.model.is_loaded:
                slot.model.unload()
