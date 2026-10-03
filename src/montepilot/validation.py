"""Validation helpers for simulation outputs and reproducibility."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends import ComputeBackend
from .design import BatchEstimate


def validate_batch(
    batch: BatchEstimate,
    backend: ComputeBackend,
    expected_size: int,
    finite_tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    estimates = backend.to_numpy(batch.estimates).astype(float).reshape(-1)
    if estimates.size != expected_size:
        raise ValueError(f"estimator returned {estimates.size} values; expected {expected_size}")

    if batch.converged is None:
        converged = np.ones(expected_size, dtype=bool)
    else:
        converged = backend.to_numpy(batch.converged).astype(bool).reshape(-1)
        if converged.size != expected_size:
            raise ValueError("converged indicator length does not match estimates")

    ses = None
    if batch.standard_errors is not None:
        ses = backend.to_numpy(batch.standard_errors).astype(float).reshape(-1)
        if ses.size != expected_size:
            raise ValueError("standard error length does not match estimates")
        converged &= np.isfinite(ses) & (ses >= 0)

    finite = np.isfinite(estimates)
    valid = converged & finite
    warnings: list[str] = []
    finite_rate = float(finite.mean()) if finite.size else 0.0
    if finite_rate < finite_tolerance:
        warnings.append(
            f"Only {finite_rate:.1%} of estimates were finite; required tolerance is {finite_tolerance:.1%}."
        )
    return estimates, ses, valid, warnings

