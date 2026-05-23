from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from omniff.validators.text_validator import ValidationResult


@dataclass
class ValidationPass:
    name: str
    validator_fn: Callable[[dict[str, Any]], ValidationResult]
    required: bool = True


@dataclass
class PipelineResult:
    passed: bool
    results: list[ValidationResult] = field(default_factory=list)
    failed_pass: str | None = None

    @property
    def score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


class ValidationPipeline:
    def __init__(self) -> None:
        self._passes: list[ValidationPass] = []

    def add_pass(
        self,
        name: str,
        validator_fn: Callable[[dict[str, Any]], ValidationResult],
        required: bool = True,
    ) -> None:
        self._passes.append(ValidationPass(name, validator_fn, required))

    def run(self, output: dict[str, Any]) -> PipelineResult:
        results = []
        for vp in self._passes:
            result = vp.validator_fn(output)
            results.append(result)
            if not result.passed and vp.required:
                return PipelineResult(
                    passed=False,
                    results=results,
                    failed_pass=vp.name,
                )
        return PipelineResult(passed=True, results=results)

    @property
    def pass_count(self) -> int:
        return len(self._passes)
