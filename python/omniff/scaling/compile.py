from __future__ import annotations

from typing import Any

from omniff.observability import get_logger

_log = get_logger("compile")


def compile_model(model: Any, backend: str = "inductor") -> Any:
    try:
        import torch

        if not hasattr(torch, "compile"):
            _log.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model

        compiled = torch.compile(model, backend=backend)
        _log.info("Model compiled with %s backend", backend)
        return compiled
    except Exception as e:
        _log.warning("torch.compile failed: %s, using uncompiled model", e)
        return model


def should_compile(model_name: str, hot_threshold: int = 10, request_count: int = 0) -> bool:
    return request_count >= hot_threshold
