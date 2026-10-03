"""Statistical helpers for reproducible performance benchmarking.

The numerical simulation engine intentionally has no dependency on a dedicated
benchmarking library.  These helpers provide deterministic bootstrap intervals,
calibrated timing-block sizes, and import of the small base-R benchmark files.
"""

from __future__ import annotations

import csv
import math
import statistics
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def calibrated_iterations(
    probe_seconds: float,
    minimum_block_seconds: float = 0.5,
    maximum_iterations: int = 512,
) -> int:
    """Choose inner iterations so a timed block is not dominated by timer noise."""

    if not math.isfinite(probe_seconds) or probe_seconds <= 0:
        raise ValueError("probe_seconds must be finite and positive")
    if not math.isfinite(minimum_block_seconds) or minimum_block_seconds <= 0:
        raise ValueError("minimum_block_seconds must be finite and positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    return min(maximum_iterations, max(1, math.ceil(minimum_block_seconds / probe_seconds)))


def refined_iterations(
    current_iterations: int,
    observed_block_seconds: float,
    minimum_block_seconds: float = 0.5,
    maximum_iterations: int = 512,
    safety_factor: float = 1.05,
) -> int:
    """Increase a calibration count after an observed block undershoots its target."""

    if current_iterations < 1:
        raise ValueError("current_iterations must be positive")
    if not math.isfinite(observed_block_seconds) or observed_block_seconds <= 0:
        raise ValueError("observed_block_seconds must be finite and positive")
    if not math.isfinite(minimum_block_seconds) or minimum_block_seconds <= 0:
        raise ValueError("minimum_block_seconds must be finite and positive")
    if maximum_iterations < current_iterations:
        raise ValueError("maximum_iterations cannot be below current_iterations")
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be at least one")
    if observed_block_seconds >= minimum_block_seconds:
        return current_iterations
    scaled = math.ceil(
        current_iterations
        * minimum_block_seconds
        / observed_block_seconds
        * safety_factor
    )
    return min(maximum_iterations, max(current_iterations + 1, scaled))


def _bootstrap_medians(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    return np.median(values[indices], axis=1)


def _percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def timing_summary(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 5000,
    seed: int = 20261002,
) -> dict[str, float | int]:
    """Summarize repeated timing blocks with a bootstrap CI for the median."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("at least one timing value is required")
    if np.any(~np.isfinite(samples)) or np.any(samples <= 0):
        raise ValueError("timing values must be finite and positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    mean = float(samples.mean())
    sd = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
    rng = np.random.default_rng(seed)
    boot = _bootstrap_medians(samples, bootstrap_resamples, rng)
    ci_low, ci_high = _percentile_interval(boot, confidence_level)
    return {
        "runs": int(samples.size),
        "median_seconds": float(np.median(samples)),
        "median_ci_low": ci_low,
        "median_ci_high": ci_high,
        "mean_seconds": mean,
        "sd_seconds": sd,
        "cv_percent": 100.0 * sd / mean if mean else 0.0,
        "q1_seconds": float(np.quantile(samples, 0.25)),
        "q3_seconds": float(np.quantile(samples, 0.75)),
        "min_seconds": float(samples.min()),
        "max_seconds": float(samples.max()),
    }


def speedup_summary(
    reference_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 5000,
    seed: int = 20261002,
) -> dict[str, float]:
    """Estimate a ratio-of-medians speedup and an independent bootstrap CI."""

    reference = np.asarray(reference_seconds, dtype=float)
    candidate = np.asarray(candidate_seconds, dtype=float)
    for name, values in (("reference", reference), ("candidate", candidate)):
        if values.ndim != 1 or values.size < 1:
            raise ValueError(f"{name} timings must contain at least one value")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} timings must be finite and positive")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    rng = np.random.default_rng(seed)
    reference_boot = _bootstrap_medians(reference, bootstrap_resamples, rng)
    candidate_boot = _bootstrap_medians(candidate, bootstrap_resamples, rng)
    ratios = reference_boot / candidate_boot
    ci_low, ci_high = _percentile_interval(ratios, confidence_level)
    return {
        "speedup": float(np.median(reference) / np.median(candidate)),
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
    }


def validation_summary(rows: Sequence[dict]) -> dict[str, float | int | None]:
    """Summarize bias and RMSE across distinct validation seeds."""

    if not rows:
        raise ValueError("at least one validation row is required")
    biases = np.asarray([float(row["bias"]) for row in rows], dtype=float)
    rmses = np.asarray([float(row["rmse"]) for row in rows], dtype=float)
    failed = sum(int(row.get("failed_reps", 0)) for row in rows)
    valid = sum(int(row.get("valid_reps", 0)) for row in rows)
    coverages = [
        float(row["coverage"])
        for row in rows
        if row.get("coverage") is not None
    ]
    return {
        "seeds": len(rows),
        "mean_bias": float(biases.mean()),
        "max_absolute_bias": float(np.abs(biases).max()),
        "mean_rmse": float(rmses.mean()),
        "min_rmse": float(rmses.min()),
        "max_rmse": float(rmses.max()),
        "valid_reps": valid,
        "failed_reps": failed,
        "mean_coverage": float(np.mean(coverages)) if coverages else None,
        "min_coverage": float(np.min(coverages)) if coverages else None,
        "max_coverage": float(np.max(coverages)) if coverages else None,
    }


def _read_r_csv(stream: Iterable[str], source: str) -> list[dict]:
    rows: list[dict] = []
    for record_index, row in enumerate(csv.DictReader(stream), start=1):
        required = {"implementation", "reps", "n", "elapsed_seconds"}
        if not required.issubset(row):
            continue
        rows.append(
            {
                "source": source,
                "record_index": record_index,
                "implementation": row["implementation"],
                "repeat": int(
                    row.get("timing_repeat") or row.get("repeat") or record_index
                ),
                "seed": int(row["seed"]) if row.get("seed") else None,
                "reps": int(row["reps"]),
                "n": int(row["n"]),
                "batch_size": int(row.get("batch_size", 1)),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "reps_per_second": float(row.get("reps_per_second", "nan")),
                "estimate_mean": float(row.get("estimate_mean", "nan")),
                "bias": float(row.get("bias", "nan")),
                "rmse": float(row.get("rmse", "nan")),
                "r_version": row.get("r_version", ""),
                "platform": row.get("platform", ""),
            }
        )
    return rows


def load_r_benchmark_rows(sources: Sequence[str | Path]) -> list[dict]:
    """Load MontePilot base-R benchmark CSVs from files, folders, or ZIP files."""

    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for raw_source in sources:
        source = Path(raw_source)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_dir():
            candidates = sorted(source.rglob("*.csv"))
            for candidate in candidates:
                with candidate.open(encoding="utf-8-sig", newline="") as stream:
                    parsed = _read_r_csv(stream, str(candidate))
                for row in parsed:
                    key = (row["source"], row["record_index"])
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                for member in sorted(archive.namelist()):
                    if not member.lower().endswith(".csv") or member.endswith("/"):
                        continue
                    with archive.open(member) as binary:
                        text = (line.decode("utf-8-sig") for line in binary)
                        parsed = _read_r_csv(text, f"{source}!{member}")
                    for row in parsed:
                        key = (row["source"], row["record_index"])
                        if key not in seen:
                            seen.add(key)
                            rows.append(row)
        else:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                parsed = _read_r_csv(stream, str(source))
            for row in parsed:
                key = (row["source"], row["record_index"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    if not rows:
        raise ValueError("no valid R benchmark rows were found")
    return rows


def median(values: Sequence[float]) -> float:
    """Small public helper used by report writers."""

    return float(statistics.median(values))
