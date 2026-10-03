"""Controlled-input numerical validation across MontePilot backends.

Performance benchmarks intentionally permit each backend to use its native
random-number generator.  That is appropriate for throughput and statistical
validation, but it cannot isolate floating-point error because the simulated
samples differ.  This module instead generates one NumPy float64 data set and
passes the same values to every candidate backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .backends import ComputeBackend


@dataclass(frozen=True)
class PrecisionCase:
    """A deterministic numerical workload and its validation tolerances."""

    name: str
    family: str
    repetitions: int
    sample_size: int
    absolute_tolerance: float
    relative_tolerance: float
    evaluator: Callable[[ComputeBackend, int, int, int], tuple[np.ndarray, np.ndarray]]


def compare_estimates(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float | int | bool]:
    """Compare replication-level estimates with a mixed absolute/relative rule."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"reference and candidate shapes differ: {reference.shape} != {candidate.shape}"
        )
    if reference.ndim != 1 or reference.size < 1:
        raise ValueError("estimate arrays must be non-empty and one-dimensional")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("estimate arrays must contain only finite values")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")

    difference = candidate - reference
    absolute_difference = np.abs(difference)
    allowed = absolute_tolerance + relative_tolerance * np.abs(reference)
    within = absolute_difference <= allowed
    denominator = np.maximum(allowed, np.finfo(np.float64).tiny)
    reference_norm = float(np.linalg.norm(reference))

    return {
        "replications": int(reference.size),
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_difference": float(difference.mean()),
        "max_absolute_difference": float(absolute_difference.max()),
        "rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_error": (
            float(np.linalg.norm(difference) / reference_norm)
            if reference_norm
            else float(np.linalg.norm(difference))
        ),
        "within_tolerance_fraction": float(within.mean()),
        "max_scaled_error": float(np.max(absolute_difference / denominator)),
        "passed": bool(np.all(within)),
    }


def _mean_case(
    backend: ComputeBackend,
    seed: int,
    repetitions: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.5, 1.0, size=(repetitions, sample_size))
    reference = data.mean(axis=1, dtype=np.float64)
    candidate = backend.to_numpy(backend.mean(backend.asarray(data), axis=1))
    return reference, candidate


def _ols_case(
    backend: ComputeBackend,
    seed: int,
    repetitions: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictors = 8
    beta = np.linspace(0.25, 1.0, predictors, dtype=np.float64)
    x = rng.normal(size=(repetitions, sample_size, predictors))
    noise = rng.normal(size=(repetitions, sample_size))
    y = np.matmul(x, beta) + noise

    xt = np.swapaxes(x, -2, -1)
    reference = np.linalg.solve(
        np.matmul(xt, x),
        np.matmul(xt, np.expand_dims(y, axis=-1)),
    )[:, 0, 0]

    bx = backend.asarray(x)
    by = backend.asarray(y)
    bxt = backend.transpose_last2(bx)
    coefficients = backend.squeeze_last(
        backend.solve(
            backend.matmul(bxt, bx),
            backend.matmul(bxt, backend.expand_last(by)),
        )
    )
    candidate = backend.to_numpy(coefficients[:, 0])
    return reference, candidate


def _ipw_case(
    backend: ComputeBackend,
    seed: int,
    repetitions: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(repetitions, sample_size))
    propensity = np.clip(1.0 / (1.0 + np.exp(-0.7 * x)), 0.05, 0.95)
    treatment = rng.uniform(size=(repetitions, sample_size)) < propensity
    outcome = treatment.astype(np.float64) + 0.5 * x + rng.normal(
        size=(repetitions, sample_size)
    )

    treated = treatment.astype(np.float64)
    treated_weight = treated / propensity
    control_weight = (1.0 - treated) / (1.0 - propensity)
    reference = (
        np.sum(treated_weight * outcome, axis=1) / np.sum(treated_weight, axis=1)
        - np.sum(control_weight * outcome, axis=1) / np.sum(control_weight, axis=1)
    )

    bx = backend.asarray(x)
    bt = backend.asarray(treated)
    by = backend.asarray(outcome)
    bp = backend.clip(backend.sigmoid(0.7 * bx), 0.05, 0.95)
    btreated_weight = bt / bp
    bcontrol_weight = (1.0 - bt) / (1.0 - bp)
    candidate = backend.to_numpy(
        backend.sum(btreated_weight * by, axis=1)
        / backend.sum(btreated_weight, axis=1)
        - backend.sum(bcontrol_weight * by, axis=1)
        / backend.sum(bcontrol_weight, axis=1)
    )
    return reference, candidate


def _bootstrap_case(
    backend: ComputeBackend,
    seed: int,
    repetitions: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.normal(0.5, 1.0, size=2000)
    indices = rng.integers(0, source.size, size=(repetitions, sample_size))
    resamples = source[indices]
    reference = resamples.mean(axis=1, dtype=np.float64)
    candidate = backend.to_numpy(backend.mean(backend.asarray(resamples), axis=1))
    return reference, candidate


def precision_cases(profile: str) -> list[PrecisionCase]:
    """Return the controlled-input suite used by the command-line harness."""

    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")
    if profile == "quick":
        mean_reps, mean_n = 256, 1000
        ols_reps, ols_n = 96, 250
        ipw_reps, ipw_n = 128, 500
        bootstrap_reps, bootstrap_n = 128, 1000
    else:
        mean_reps, mean_n = 2000, 5000
        ols_reps, ols_n = 500, 500
        ipw_reps, ipw_n = 1000, 1000
        bootstrap_reps, bootstrap_n = 1000, 2000

    return [
        PrecisionCase(
            "mean_n5000" if profile == "full" else "mean_n1000",
            "mean",
            mean_reps,
            mean_n,
            5e-6,
            5e-6,
            _mean_case,
        ),
        PrecisionCase(
            "ols_n500_p8" if profile == "full" else "ols_n250_p8",
            "linear_regression",
            ols_reps,
            ols_n,
            2e-4,
            2e-4,
            _ols_case,
        ),
        PrecisionCase(
            "ipw_n1000" if profile == "full" else "ipw_n500",
            "causal_ipw",
            ipw_reps,
            ipw_n,
            2e-4,
            2e-4,
            _ipw_case,
        ),
        PrecisionCase(
            "bootstrap_n2000" if profile == "full" else "bootstrap_n1000",
            "bootstrap",
            bootstrap_reps,
            bootstrap_n,
            5e-6,
            5e-6,
            _bootstrap_case,
        ),
    ]


def evaluate_precision_case(
    case: PrecisionCase,
    backend: ComputeBackend,
    seed: int,
) -> dict[str, float | int | str | bool]:
    """Run one controlled-input case and return a machine-readable result."""

    reference, candidate = case.evaluator(
        backend,
        seed,
        case.repetitions,
        case.sample_size,
    )
    backend.synchronize()
    return {
        "workload": case.name,
        "family": case.family,
        "backend": backend.name,
        "precision": backend.dtype_name,
        "seed": int(seed),
        "sample_size": case.sample_size,
        "absolute_tolerance": case.absolute_tolerance,
        "relative_tolerance": case.relative_tolerance,
        **compare_estimates(
            reference,
            candidate,
            absolute_tolerance=case.absolute_tolerance,
            relative_tolerance=case.relative_tolerance,
        ),
    }
