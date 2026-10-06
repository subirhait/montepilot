"""MontePilot public API."""

from .config import RunConfig
from .design import BatchEstimate, SimulationDesign
from .engine import SimulationRunner
from .results import ConditionResult, SimulationReport

__all__ = [
    "BatchEstimate",
    "ConditionResult",
    "RunConfig",
    "SimulationDesign",
    "SimulationReport",
    "SimulationRunner",
    "run",
]

__version__ = "0.6.2"


def run(
    design: SimulationDesign,
    *,
    backend: str = "auto",
    precision: str = "auto",
    batch_size: int = 512,
    min_reps: int = 1000,
    max_reps: int = 10000,
    target_mcse: float | None = 0.005,
    target_coverage_mcse: float | None = None,
    stopping_rule: str = "all",
    confidence_level: float = 0.95,
    seed: int = 20261002,
    checkpoint_dir=None,
    checkpoint_every_batches: int = 5,
    auto_expected_reps: int | None = None,
    advisor=None,
) -> SimulationReport:
    """Run a simulation design through the high-level MontePilot interface."""

    config = RunConfig(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        min_reps=min_reps,
        max_reps=max_reps,
        target_mcse=target_mcse,
        target_coverage_mcse=target_coverage_mcse,
        stopping_rule=stopping_rule,
        confidence_level=confidence_level,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_batches=checkpoint_every_batches,
        auto_expected_reps=auto_expected_reps,
    )
    return SimulationRunner(config, advisor=advisor).run(design)
