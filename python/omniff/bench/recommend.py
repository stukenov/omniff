from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelRecommendation:
    name: str
    model_id: str
    vram_gb: float
    quality: str
    pipelines: list[str]


MODELS = [
    ModelRecommendation("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.5, "basic", ["text"]),
    ModelRecommendation("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 4.0, "good", ["text", "code"]),
    ModelRecommendation("Qwen3-4B", "Qwen/Qwen3-4B", 8.0, "strong", ["text", "code"]),
    ModelRecommendation("Qwen3-8B", "Qwen/Qwen3-8B", 16.0, "excellent", ["text", "code"]),
    ModelRecommendation("Qwen2.5-VL-3B", "Qwen/Qwen2.5-VL-3B-Instruct", 7.0, "good", ["image", "video"]),
    ModelRecommendation("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct", 15.0, "excellent", ["image", "video"]),
    ModelRecommendation("Whisper-large-v3", "openai/whisper-large-v3", 3.0, "excellent", ["audio"]),
    ModelRecommendation("Whisper-medium", "openai/whisper-medium", 1.5, "good", ["audio"]),
    ModelRecommendation("SDXL-turbo", "stabilityai/sdxl-turbo", 5.0, "good", ["image_gen", "image_edit"]),
    ModelRecommendation("Bark-small", "suno/bark-small", 2.0, "basic", ["tts"]),
]


def recommend_models(
    vram_budget_gb: float,
    pipelines: list[str] | None = None,
) -> list[ModelRecommendation]:
    pipelines = pipelines or ["text", "image", "audio"]

    candidates = [
        m for m in MODELS
        if any(p in m.pipelines for p in pipelines)
    ]

    candidates.sort(key=lambda m: m.vram_gb, reverse=True)

    selected = []
    remaining = vram_budget_gb
    covered = set()

    for m in candidates:
        needed_pipelines = set(m.pipelines) & set(pipelines)
        if needed_pipelines <= covered:
            continue
        if m.vram_gb <= remaining:
            selected.append(m)
            remaining -= m.vram_gb
            covered.update(needed_pipelines)

    return selected


def format_recommendation(models: list[ModelRecommendation], budget: float) -> str:
    lines = [f"Model Recommendation (VRAM budget: {budget:.0f} GB)", "=" * 50]
    total = 0
    for m in models:
        lines.append(f"  {m.name}: {m.model_id} ({m.vram_gb:.1f} GB, {m.quality})")
        lines.append(f"    Pipelines: {', '.join(m.pipelines)}")
        total += m.vram_gb
    lines.append(f"\nTotal VRAM: {total:.1f} / {budget:.0f} GB")
    uncovered = budget - total
    if uncovered > 2:
        lines.append(f"Remaining: {uncovered:.1f} GB (room for additional models)")
    return "\n".join(lines)
