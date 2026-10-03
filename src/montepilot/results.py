"""Serializable result models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConditionResult:
    condition_index: int
    condition: dict[str, Any]
    truth: float
    backend: str
    precision: str
    attempted_reps: int
    valid_reps: int
    failed_reps: int
    estimate_mean: float
    bias: float
    empirical_sd: float
    rmse: float
    estimate_mcse: float
    coverage: float | None
    coverage_mcse: float | None
    stopped_early: bool
    stop_reason: str
    runtime_seconds: float
    batches: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class SimulationReport:
    design_name: str
    design_description: str
    config: dict[str, Any]
    selected_backend: str
    selected_precision: str
    backend_timings: dict[str, float]
    backend_warmup_seconds: dict[str, float]
    backend_projected_seconds: dict[str, float]
    backend_probe_batch_size: int | None
    backend_failures: dict[str, str]
    conditions: list[ConditionResult]
    started_at_utc: str
    finished_at_utc: str
    total_runtime_seconds: float
    advice: list[str]
    schema_version: str = "1.1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)
