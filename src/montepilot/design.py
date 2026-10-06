"""Public simulation design protocol and estimate container."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends import ComputeBackend


def derive_seed(
    master_seed: int,
    condition_index: int,
    batch_index: int,
    stream_id: int = 0,
) -> int:
    """Derive one portable integer seed from a design position and stream id."""

    if stream_id < 0:
        raise ValueError("stream_id must be nonnegative")
    raw = f"{master_seed}:{condition_index}:{batch_index}:{stream_id}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(raw, digest_size=8).digest(), "little"
    ) % (2**63 - 1)


@dataclass(frozen=True)
class BatchContext:
    backend: ComputeBackend
    seed: int
    condition_index: int
    batch_index: int
    master_seed: int | None = None

    def seed_for(self, stream_id: int = 0) -> int:
        """Return a deterministic seed for one stream within this batch.

        Engine-created contexts retain the master seed and therefore hash the
        complete ``(master, condition, batch, stream)`` tuple. Manually created
        legacy contexts still receive stable distinct seeds derived from their
        batch seed.
        """

        if stream_id < 0:
            raise ValueError("stream_id must be nonnegative")
        if self.master_seed is not None:
            return derive_seed(
                self.master_seed,
                self.condition_index,
                self.batch_index,
                stream_id,
            )
        if stream_id == 0:
            return self.seed
        return derive_seed(self.seed, self.condition_index, self.batch_index, stream_id)


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
