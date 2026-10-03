"""Public simulation design protocol and estimate container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends import ComputeBackend


@dataclass(frozen=True)
class BatchContext:
    backend: ComputeBackend
    seed: int
    condition_index: int
    batch_index: int


@dataclass
class BatchEstimate:
    estimates: Any
    standard_errors: Any | None = None
    converged: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


Generator = Callable[[BatchContext, Mapping[str, Any], int], Any]
Estimator = Callable[[Any, BatchContext, Mapping[str, Any]], BatchEstimate | Any]


@dataclass(frozen=True)
class SimulationDesign:
    name: str
    conditions: Sequence[Mapping[str, Any]]
    generator: Generator
    estimator: Estimator
    truth: float | Callable[[Mapping[str, Any]], float]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("design name cannot be empty")
        if not self.conditions:
            raise ValueError("at least one design condition is required")

    def truth_for(self, condition: Mapping[str, Any]) -> float:
        value = self.truth(condition) if callable(self.truth) else self.truth
        if not np.isfinite(value):
            raise ValueError("truth must be finite")
        return float(value)


def coerce_batch_estimate(value: BatchEstimate | Any) -> BatchEstimate:
    return value if isinstance(value, BatchEstimate) else BatchEstimate(estimates=value)

