from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkResult:
    pipeline: str
    metric_name: str
    metric_value: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    name: str = "omniff-bench"
    results: list[BenchmarkResult] = field(default_factory=list)

    def run_benchmark(
        self,
        pipeline: str,
        fn: Callable[[], dict[str, Any]],
        metric_fn: Callable[[dict[str, Any]], tuple[str, float]] | None = None,
        warmup: int = 1,
        iterations: int = 3,
    ) -> BenchmarkResult:
        for _ in range(warmup):
            fn()

        latencies = []
        last_result = None
        for _ in range(iterations):
            start = time.monotonic()
            last_result = fn()
            latencies.append((time.monotonic() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        metric_name, metric_value = "none", 0.0
        if metric_fn and last_result:
            metric_name, metric_value = metric_fn(last_result)

        result = BenchmarkResult(
            pipeline=pipeline,
            metric_name=metric_name,
            metric_value=metric_value,
            latency_ms=round(avg_latency, 2),
            metadata={
                "iterations": iterations,
                "latencies_ms": [round(l, 2) for l in latencies],
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
            },
        )
        self.results.append(result)
        return result

    def summary(self) -> str:
        lines = [f"Benchmark: {self.name}", "=" * 50]
        for r in self.results:
            lines.append(
                f"  {r.pipeline}: {r.metric_name}={r.metric_value:.3f} "
                f"latency={r.latency_ms:.1f}ms"
            )
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "pipeline": r.pipeline,
                "metric": r.metric_name,
                "value": r.metric_value,
                "latency_ms": r.latency_ms,
                **r.metadata,
            }
            for r in self.results
        ]
