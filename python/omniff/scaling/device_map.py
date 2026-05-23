from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float
    assigned_models: list[str] = field(default_factory=list)


class DeviceMap:
    def __init__(self) -> None:
        self._assignments: dict[str, int] = {}
        self._gpus: list[GPUInfo] = []

    def scan_gpus(self) -> list[GPUInfo]:
        try:
            import torch
            if not torch.cuda.is_available():
                return []
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    total_gb=total / 1024**3,
                    free_gb=free / 1024**3,
                ))
            self._gpus = gpus
            return gpus
        except ImportError:
            return []

    def assign(self, model_name: str, gpu_index: int) -> None:
        self._assignments[model_name] = gpu_index
        for gpu in self._gpus:
            if gpu.index == gpu_index:
                gpu.assigned_models.append(model_name)

    def auto_assign(self, model_name: str, required_gb: float = 0) -> int:
        if not self._gpus:
            self.scan_gpus()
        if not self._gpus:
            return -1

        best = max(self._gpus, key=lambda g: g.free_gb)
        if required_gb > 0 and best.free_gb < required_gb:
            return -1

        self.assign(model_name, best.index)
        return best.index

    def get_device(self, model_name: str) -> str:
        if model_name in self._assignments:
            return f"cuda:{self._assignments[model_name]}"
        return "auto"

    def get_assignments(self) -> dict[str, int]:
        return dict(self._assignments)

    def get_gpu_info(self) -> list[GPUInfo]:
        if not self._gpus:
            self.scan_gpus()
        return self._gpus

    def get_multi_gpu_device_map(self, model_size_gb: float) -> dict[str, Any] | str:
        if not self._gpus:
            self.scan_gpus()
        if not self._gpus:
            return "cpu"
        if len(self._gpus) == 1:
            if self._gpus[0].free_gb >= model_size_gb:
                return f"cuda:0"
            return "cpu"

        total_free = sum(g.free_gb for g in self._gpus)
        if total_free < model_size_gb:
            return "cpu"

        if any(g.free_gb >= model_size_gb for g in self._gpus):
            best = max(self._gpus, key=lambda g: g.free_gb)
            return f"cuda:{best.index}"

        return "auto"
