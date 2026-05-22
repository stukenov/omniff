from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RouterConfig:
    router_type: str
    path: str


@dataclass
class ExpertConfig:
    name: str
    model_type: str
    path: str
    loading: str = "warm"
    quantization: str | None = None
    device: str | None = None


@dataclass
class OmniFFConfig:
    name: str
    version: str
    router: RouterConfig
    experts: dict[str, ExpertConfig] = field(default_factory=dict)
    graph_templates_dir: str | None = None

    @classmethod
    def load(cls, path: Path) -> OmniFFConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)

        router = RouterConfig(
            router_type=raw["router"]["type"],
            path=raw["router"].get("path", ""),
        )

        experts = {}
        for name, spec in raw.get("experts", {}).items():
            experts[name] = ExpertConfig(
                name=spec.get("name", name),
                model_type=spec["model_type"],
                path=spec["path"],
                loading=spec.get("loading", "warm"),
                quantization=spec.get("quantization"),
                device=spec.get("device"),
            )

        return cls(
            name=raw["name"],
            version=raw["version"],
            router=router,
            experts=experts,
            graph_templates_dir=raw.get("graph_templates_dir"),
        )
