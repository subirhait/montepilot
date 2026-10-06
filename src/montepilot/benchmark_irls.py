"""Vectorized logistic-IRLS routines for the frozen publication benchmark.

This module is deliberately not exported as a public simulation design. It is
benchmark infrastructure for version 0.6.2, which keeps the release a patch
update rather than adding a new public design API.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backends import ComputeBackend
from .design import derive_seed


@dataclass(frozen=True)
class IRLSSettings:
    tolerance: float = 1e-6
    maximum_iterations: int = 50
    probability_clip: float = 1e-6
    determinant_floor: float = 1e-12
    coefficient_bound: float = 25.0


@dataclass
class IRLSResult:
    estimates: np.ndarray
    standard_errors: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    failure_reason: np.ndarray


def _torch_module():
    import torch

    return torch


def _zeros(backend: ComputeBackend, size: int, *, boolean: bool = False, integer: bool = False):
    if backend.is_torch:
        torch = _torch_module()
        dtype = torch.bool if boolean else (torch.int64 if integer else backend._torch_dtype())
        return torch.zeros(size, device=backend.device, dtype=dtype)
    dtype = bool if boolean else (np.int64 if integer else backend._numpy_dtype())
    return np.zeros(size, dtype=dtype)


def _where(backend: ComputeBackend, condition, left, right):
    if backend.is_torch:
        return _torch_module().where(condition, left, right)
    return np.where(condition, left, right)


def _isfinite(backend: ComputeBackend, value):
    if backend.is_torch:
        return _torch_module().isfinite(value)
    return np.isfinite(value)


def _abs(backend: ComputeBackend, value):
    if backend.is_torch:
        return _torch_module().abs(value)
    return np.abs(value)


def _maximum(backend: ComputeBackend, left, right):
    if backend.is_torch:
        return _torch_module().maximum(left, right)
    return np.maximum(left, right)


def _logical_all(*values):
    out = values[0]
    for value in values[1:]:
        out = out & value
    return out


def _any(backend: ComputeBackend, value) -> bool:
    """Return a scalar activity flag with one explicit device synchronization."""

    if backend.is_torch:
        return bool(value.any().item())
    return bool(np.any(value))


def fit_logistic_irls(
    x: Any,
    y: Any,
    backend: ComputeBackend,
    settings: IRLSSettings,
) -> IRLSResult:
    """Fit independent intercept-plus-slope logistic models in one batch."""

    x = backend.asarray(x)
    y = backend.asarray(y)
    if len(x.shape) != 2 or x.shape != y.shape:
        raise ValueError("x and y must have the same two-dimensional shape")
    batch_size = int(x.shape[0])
    beta0 = _zeros(backend, batch_size)
    beta1 = _zeros(backend, batch_size)
    converged = _zeros(backend, batch_size, boolean=True)
    failed = _zeros(backend, batch_size, boolean=True)
    iterations = _zeros(backend, batch_size, integer=True)
    reason_code = _zeros(backend, batch_size, integer=True)

    for iteration in range(1, settings.maximum_iterations + 1):
        active = ~(converged | failed)
        if not _any(backend, active):
            break

        eta = beta0.reshape((-1, 1)) + beta1.reshape((-1, 1)) * x
        probability = backend.clip(
            backend.sigmoid(eta),
            settings.probability_clip,
            1.0 - settings.probability_clip,
        )
        weight = probability * (1.0 - probability)
        residual = y - probability
        gradient0 = backend.sum(residual, axis=1)
        gradient1 = backend.sum(residual * x, axis=1)
        h00 = backend.sum(weight, axis=1)
        h01 = backend.sum(weight * x, axis=1)
        h11 = backend.sum(weight * x * x, axis=1)
        determinant = h00 * h11 - h01 * h01

        finite_information = _logical_all(
            _isfinite(backend, gradient0),
            _isfinite(backend, gradient1),
            _isfinite(backend, h00),
            _isfinite(backend, h01),
            _isfinite(backend, h11),
            _isfinite(backend, determinant),
        )
        nonfinite = active & ~finite_information
        singular = active & finite_information & (determinant <= settings.determinant_floor)
        reason_code = _where(backend, nonfinite, 1, reason_code)
        reason_code = _where(backend, singular, 2, reason_code)
        failed = failed | nonfinite | singular

        usable = active & ~failed
        safe_determinant = _where(backend, usable, determinant, 1.0)
        step0 = (h11 * gradient0 - h01 * gradient1) / safe_determinant
        step1 = (-h01 * gradient0 + h00 * gradient1) / safe_determinant
        finite_step = _isfinite(backend, step0) & _isfinite(backend, step1)
        bad_step = usable & ~finite_step
        reason_code = _where(backend, bad_step, 1, reason_code)
        failed = failed | bad_step
        usable = usable & finite_step

        proposed0 = beta0 + step0
        proposed1 = beta1 + step1
        bound_exceeded = usable & (
            (_abs(backend, proposed0) > settings.coefficient_bound)
            | (_abs(backend, proposed1) > settings.coefficient_bound)
        )
        reason_code = _where(backend, bound_exceeded, 3, reason_code)
        failed = failed | bound_exceeded
        usable = usable & ~bound_exceeded

        beta0 = _where(backend, usable, proposed0, beta0)
        beta1 = _where(backend, usable, proposed1, beta1)
        step_norm = _maximum(backend, _abs(backend, step0), _abs(backend, step1))
        coefficient_norm = _maximum(
            backend, _abs(backend, beta0), _abs(backend, beta1)
        )
        converged_now = usable & (
            step_norm <= settings.tolerance * (1.0 + coefficient_norm)
        )
        converged = converged | converged_now
        just_finished = converged_now | nonfinite | singular | bad_step | bound_exceeded
        iterations = _where(backend, just_finished & (iterations == 0), iteration, iterations)

    hit_cap = ~(converged | failed)
    reason_code = _where(backend, hit_cap, 4, reason_code)
    failed = failed | hit_cap
    iterations = _where(
        backend,
        hit_cap & (iterations == 0),
        settings.maximum_iterations,
        iterations,
    )

    eta = beta0.reshape((-1, 1)) + beta1.reshape((-1, 1)) * x
    probability = backend.clip(
        backend.sigmoid(eta),
        settings.probability_clip,
        1.0 - settings.probability_clip,
    )
    weight = probability * (1.0 - probability)
    h00 = backend.sum(weight, axis=1)
    h01 = backend.sum(weight * x, axis=1)
    h11 = backend.sum(weight * x * x, axis=1)
    determinant = h00 * h11 - h01 * h01
    valid_se = converged & _isfinite(backend, determinant) & (
        determinant > settings.determinant_floor
    )
    safe_determinant = _where(backend, valid_se, determinant, 1.0)
    slope_variance = h00 / safe_determinant
    valid_se = valid_se & _isfinite(backend, slope_variance) & (slope_variance > 0)
    if backend.is_torch:
        standard_error = _torch_module().sqrt(
            _where(backend, valid_se, slope_variance, float("nan"))
        )
    else:
        standard_error = np.sqrt(
            _where(backend, valid_se, slope_variance, float("nan"))
        )
    se_failure = converged & ~valid_se
    reason_code = _where(backend, se_failure, 1, reason_code)
    converged = converged & valid_se
    failed = failed | se_failure

    reason_names = np.asarray(
        ["", "nonfinite", "singular_information", "coefficient_bound", "iteration_cap"],
        dtype=object,
    )
    codes = backend.to_numpy(reason_code).astype(int, copy=False)
    return IRLSResult(
        estimates=backend.to_numpy(beta1).astype(np.float64, copy=False),
        standard_errors=backend.to_numpy(standard_error).astype(np.float64, copy=False),
        converged=backend.to_numpy(converged).astype(bool, copy=False),
        iterations=backend.to_numpy(iterations).astype(int, copy=False),
        failure_reason=reason_names[codes],
    )


def generate_controlled_logistic_data(
    *,
    seed: int,
    replications: int,
    sample_size: int,
    intercept: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one float64 input shared by every controlled-input backend."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(replications, sample_size))
    eta = intercept + slope * x
    probability = np.empty_like(eta)
    positive = eta >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    probability[~positive] = exp_eta / (1.0 + exp_eta)
    y = (rng.random(size=eta.shape) < probability).astype(np.float64)
    return x, y


def run_native_condition(
    backend: ComputeBackend,
    condition: dict[str, Any],
    settings: IRLSSettings,
    *,
    master_seed: int,
    condition_index: int,
) -> tuple[dict[str, Any], float]:
    """Generate, fit, synchronize, and summarize one IRLS condition."""

    replications = int(condition["replications"])
    batch_size = int(condition["batch_size"])
    sample_size = int(condition["n"])
    estimates: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    converged: list[np.ndarray] = []
    iterations: list[np.ndarray] = []
    reasons: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, replications, batch_size)):
        current = min(batch_size, replications - start)
        x = backend.normal(
            (current, sample_size),
            seed=derive_seed(master_seed, condition_index, batch_index, 0),
        )
        probability = backend.sigmoid(
            float(condition["intercept"]) + float(condition["slope"]) * x
        )
        y = (
            backend.uniform(
                (current, sample_size),
                seed=derive_seed(master_seed, condition_index, batch_index, 1),
            )
            < probability
        ) * 1.0
        result = fit_logistic_irls(x, y, backend, settings)
        estimates.append(result.estimates)
        standard_errors.append(result.standard_errors)
        converged.append(result.converged)
        iterations.append(result.iterations)
        reasons.append(result.failure_reason)
    backend.synchronize()
    elapsed = time.perf_counter() - started
    estimate = np.concatenate(estimates)
    standard_error = np.concatenate(standard_errors)
    valid = np.concatenate(converged)
    iteration = np.concatenate(iterations)
    reason = np.concatenate(reasons)
    truth = float(condition["slope"])
    valid_estimate = estimate[valid]
    valid_se_values = standard_error[valid]
    if valid_estimate.size:
        error = valid_estimate - truth
        coverage = np.abs(error) <= 1.959963984540054 * valid_se_values
        bias = float(np.mean(error))
        rmse = float(np.sqrt(np.mean(error**2)))
        empirical_sd = float(np.std(valid_estimate, ddof=1)) if valid_estimate.size > 1 else 0.0
        coverage_value = float(np.mean(coverage))
    else:
        bias = rmse = empirical_sd = coverage_value = float("nan")
    reason_counts = {
        name: int(np.sum(reason == name))
        for name in ("nonfinite", "singular_information", "coefficient_bound", "iteration_cap")
    }
    valid_iterations = iteration[valid]
    metrics = {
        "condition": str(condition["name"]),
        "backend": backend.name,
        "precision": backend.dtype_name,
        "master_seed": int(master_seed),
        "attempted_reps": int(replications),
        "valid_reps": int(np.sum(valid)),
        "failed_reps": int(np.sum(~valid)),
        "failure_rate": float(np.mean(~valid)),
        "truth": truth,
        "estimate_mean": float(np.mean(valid_estimate)) if valid_estimate.size else float("nan"),
        "bias": bias,
        "rmse": rmse,
        "empirical_sd": empirical_sd,
        "coverage": coverage_value,
        "mean_iterations": float(np.mean(valid_iterations)) if valid_iterations.size else float("nan"),
        "median_iterations": float(np.median(valid_iterations)) if valid_iterations.size else float("nan"),
        "p95_iterations": float(np.quantile(valid_iterations, 0.95)) if valid_iterations.size else float("nan"),
        "max_iterations_observed": int(np.max(valid_iterations)) if valid_iterations.size else 0,
        **{f"fail_{key}": value for key, value in reason_counts.items()},
    }
    return metrics, elapsed


def settings_from_protocol(config: dict[str, Any]) -> IRLSSettings:
    values = config["logistic_irls"]["convergence"]
    return IRLSSettings(
        tolerance=float(values["tolerance"]),
        maximum_iterations=int(values["maximum_iterations"]),
        probability_clip=float(values["probability_clip"]),
        determinant_floor=float(values["determinant_floor"]),
        coefficient_bound=float(values["coefficient_bound"]),
    )
