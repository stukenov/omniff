from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    pass


class VRAMInsufficientError(Exception):
    pass


def retry_on_oom(max_retries: int = 2, backoff_base: float = 2.0):
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "CUDA out of memory" not in msg and "OutOfMemoryError" not in msg:
                        raise
                    last_exc = e
                    if attempt < max_retries:
                        wait = backoff_base**attempt
                        try:
                            import torch

                            torch.cuda.empty_cache()
                        except ImportError:
                            pass
                        time.sleep(wait)
            raise last_exc

        return wrapper

    return decorator


def check_vram(required_gb: float) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free = 0
        for i in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free += free_bytes
        return free / (1024**3) >= required_gb
    except ImportError:
        return False


def get_free_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free = 0
        for i in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free += free_bytes
        return free / (1024**3)
    except ImportError:
        return 0.0


VRAM_ESTIMATES = {
    "Qwen/Qwen3-4B": 8.0,
    "Qwen/Qwen2.5-VL-3B-Instruct": 7.0,
    "openai/whisper-large-v3": 3.0,
    "stabilityai/sdxl-turbo": 5.0,
}


class ModelMutex:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def acquire(self, model_name: str) -> None:
        with self._global_lock:
            if model_name not in self._locks:
                self._locks[model_name] = threading.Lock()
        self._locks[model_name].acquire()

    def release(self, model_name: str) -> None:
        if model_name in self._locks:
            try:
                self._locks[model_name].release()
            except RuntimeError:
                pass
