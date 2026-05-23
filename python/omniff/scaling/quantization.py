from __future__ import annotations

from pathlib import Path
from typing import Any


def detect_quantization(model_path: str) -> str | None:
    p = Path(model_path)
    name = p.name.lower() if p.exists() else model_path.lower()

    if "gptq" in name:
        return "gptq"
    if "awq" in name:
        return "awq"
    if name.endswith(".gguf"):
        return "gguf"
    if "int4" in name or "4bit" in name:
        return "int4"
    if "int8" in name or "8bit" in name:
        return "int8"
    return None


def get_quantization_config(quant_type: str) -> dict[str, Any]:
    if quant_type == "gptq":
        return {"quantization_config": {"bits": 4, "group_size": 128}}
    if quant_type == "awq":
        return {"quantization_config": {"bits": 4, "group_size": 128}}
    if quant_type == "int8":
        return {"load_in_8bit": True}
    if quant_type == "int4":
        return {"load_in_4bit": True}
    return {}


def estimate_quantized_size_gb(base_size_gb: float, quant_type: str | None) -> float:
    ratios = {
        "gptq": 0.25,
        "awq": 0.25,
        "int4": 0.25,
        "int8": 0.5,
        "gguf": 0.3,
    }
    if quant_type and quant_type in ratios:
        return base_size_gb * ratios[quant_type]
    return base_size_gb
