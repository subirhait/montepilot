"""Configuration objects for simulation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

BackendName = Literal["auto", "numpy", "torch_cpu", "torch_cuda", "torch_xpu"]
PrecisionName = Literal["auto", "float32", "float64"]
StoppingRule = Literal["all", "any"]


@dataclass(frozen=True)
class RunConfig:
    """Controls execution, reproducibility, stopping, and persistence."""

    backend: BackendName = "auto"
    precision: PrecisionName = "auto"
    batch_size: int = 512
    min_reps: int = 1000
    max_reps: int = 10000
    target_mcse: float | None = 0.005
    target_coverage_mcse: float | None = None
    stopping_rule: StoppingRule = "all"
    confidence_level: float = 0.95
    seed: int = 20261002
    checkpoint_dir: str | Path | None = None
    checkpoint_every_batches: int = 5
    auto_probe_batch_size: int | None = None
    auto_probe_max_batch_size: int = 2048
    auto_probe_conditions: int = 3
    auto_probe_repeats: int = 2
    auto_expected_reps: int | None = None
    finite_tolerance: float = 0.99

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.min_reps < 2:
            raise ValueError("min_reps must be at least 2")
        if self.max_reps < self.min_reps:
            raise ValueError("max_reps must be at least min_reps")
        if self.target_mcse is not None and self.target_mcse <= 0:
            raise ValueError("target_mcse must be positive or None")
        if self.target_coverage_mcse is not None and self.target_coverage_mcse <= 0:
            raise ValueError("target_coverage_mcse must be positive or None")
        if self.stopping_rule not in {"all", "any"}:
            raise ValueError("stopping_rule must be 'all' or 'any'")
        if self.precision not in {"auto", "float32", "float64"}:
            raise ValueError("precision must be 'auto', 'float32', or 'float64'")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie between 0 and 1")
        if self.checkpoint_every_batches < 1:
            raise ValueError("checkpoint_every_batches must be positive")
        if self.auto_probe_batch_size is not None and self.auto_probe_batch_size < 1:
            raise ValueError("auto_probe_batch_size must be positive or None")
        if self.auto_probe_max_batch_size < 1:
            raise ValueError("auto_probe_max_batch_size must be positive")
        if self.auto_probe_conditions < 1:
            raise ValueError("auto_probe_conditions must be positive")
        if self.auto_expected_reps is not None and not (
            self.min_reps <= self.auto_expected_reps <= self.max_reps
        ):
            raise ValueError("auto_expected_reps must lie between min_reps and max_reps")
        if not 0 < self.finite_tolerance <= 1:
            raise ValueError("finite_tolerance must lie in (0, 1]")

    def as_dict(self) -> dict:
        out = asdict(self)
        if out["checkpoint_dir"] is not None:
            out["checkpoint_dir"] = str(out["checkpoint_dir"])
        return out
