"""Adaptive batch simulation engine."""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from .advice import Advisor, rule_based_advice
from .backends import ComputeBackend, probe_backends, resolve_backend
from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from .config import RunConfig
from .design import BatchContext, SimulationDesign, coerce_batch_estimate
from .metrics import BinaryMoments, OnlineMoments
from .results import ConditionResult, SimulationReport
from .validation import validate_batch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_seed(master: int, condition_index: int, batch_index: int) -> int:
    raw = f"{master}:{condition_index}:{batch_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % (2**63 - 1)


class SimulationRunner:
    def __init__(self, config: RunConfig | None = None, advisor: Advisor | None = None):
        self.config = config or RunConfig()
        self.advisor = advisor or rule_based_advice

    def _execute_batch(
        self,
        design: SimulationDesign,
        condition: dict[str, Any],
        condition_index: int,
        batch_index: int,
        batch_size: int,
        backend: ComputeBackend,
    ):
        context = BatchContext(
            backend=backend,
            seed=_derive_seed(self.config.seed, condition_index, batch_index),
            condition_index=condition_index,
            batch_index=batch_index,
        )
        data = design.generator(context, condition, batch_size)
        return coerce_batch_estimate(design.estimator(data, context, condition))

    def _probe_batch_size(self) -> int:
        requested = self.config.auto_probe_batch_size
        if requested is None:
            requested = min(self.config.batch_size, self.config.auto_probe_max_batch_size)
        return min(self.config.batch_size, requested)

    def _probe_condition_indices(self, design: SimulationDesign) -> list[int]:
        total = len(design.conditions)
        count = min(total, self.config.auto_probe_conditions)
        if count == 1:
            return [0]
        return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})

    def _select_backend(
        self, design: SimulationDesign
    ) -> tuple[ComputeBackend, dict, dict, dict, dict, int | None]:
        if self.config.backend != "auto":
            return (
                resolve_backend(self.config.backend, precision=self.config.precision),
                {},
                {},
                {},
                {},
                None,
            )
        probe_batch_size = self._probe_batch_size()
        condition_indices = self._probe_condition_indices(design)

        def workload(backend: ComputeBackend):
            return [
                self._execute_batch(
                    design,
                    dict(design.conditions[condition_index]),
                    condition_index=condition_index,
                    batch_index=-1,
                    batch_size=probe_batch_size,
                    backend=backend,
                )
                for condition_index in condition_indices
            ]

        _, timings, warmup_timings, failures = probe_backends(
            workload,
            repeats=self.config.auto_probe_repeats,
            precision=self.config.precision,
        )
        if self.config.auto_expected_reps is not None:
            planned_reps = self.config.auto_expected_reps
        elif self.config.target_mcse is None and self.config.target_coverage_mcse is None:
            planned_reps = self.config.max_reps
        else:
            planned_reps = self.config.min_reps
            if self.config.target_coverage_mcse is not None:
                worst_case_coverage_reps = math.ceil(
                    0.25 / (self.config.target_coverage_mcse**2)
                )
                planned_reps = max(planned_reps, worst_case_coverage_reps)
            planned_reps = min(planned_reps, self.config.max_reps)
        total_condition_batches = (
            math.ceil(planned_reps / self.config.batch_size) * len(design.conditions)
        )
        scale = self.config.batch_size / probe_batch_size
        projections = {
            name: warmup_timings[name]
            + (timing / len(condition_indices)) * scale * total_condition_batches
            for name, timing in timings.items()
        }
        winner = min(projections, key=projections.get)
        return (
            resolve_backend(winner, precision=self.config.precision),
            timings,
            warmup_timings,
            projections,
            failures,
            probe_batch_size,
        )

    def run(self, design: SimulationDesign) -> SimulationReport:
        started_at = _utc_now()
        total_started = time.perf_counter()
        backend, timings, warmups, projections, failures, probe_batch_size = (
            self._select_backend(design)
        )
        results: list[ConditionResult] = []
        for condition_index, raw_condition in enumerate(design.conditions):
            results.append(
                self._run_condition(
                    design,
                    dict(raw_condition),
                    condition_index,
                    backend,
                )
            )
        report = SimulationReport(
            design_name=design.name,
            design_description=design.description,
            config=self.config.as_dict(),
            selected_backend=backend.name,
            selected_precision=backend.dtype_name,
            backend_timings=timings,
            backend_warmup_seconds=warmups,
            backend_projected_seconds=projections,
            backend_probe_batch_size=probe_batch_size,
            backend_failures=failures,
            conditions=results,
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            total_runtime_seconds=time.perf_counter() - total_started,
            advice=[],
        )
        report.advice = self.advisor(report)
        return report

    def _run_condition(
        self,
        design: SimulationDesign,
        condition: dict[str, Any],
        condition_index: int,
        backend: ComputeBackend,
    ) -> ConditionResult:
        truth = design.truth_for(condition)
        estimate_moments = OnlineMoments()
        squared_error_moments = OnlineMoments()
        coverage_moments = BinaryMoments()
        attempted = failed = batches = 0
        warnings: list[str] = []
        checkpoint = None
        if self.config.checkpoint_dir is not None:
            checkpoint = checkpoint_path(self.config.checkpoint_dir, design.name, condition_index)
            restored = load_checkpoint(checkpoint)
            if restored is not None:
                if restored.get("condition") != condition or restored.get("backend") != backend.name:
                    raise RuntimeError(f"Checkpoint does not match current run: {checkpoint}")
                if restored.get("seed") != self.config.seed:
                    raise RuntimeError(f"Checkpoint seed does not match current run: {checkpoint}")
                if restored.get("batch_size") != self.config.batch_size:
                    raise RuntimeError(f"Checkpoint batch size does not match current run: {checkpoint}")
                if restored.get("precision", backend.dtype_name) != backend.dtype_name:
                    raise RuntimeError(f"Checkpoint precision does not match current run: {checkpoint}")
                restored_stopping = restored.get("stopping")
                if restored_stopping is not None and restored_stopping != self._stopping_signature():
                    raise RuntimeError(f"Checkpoint stopping rules do not match current run: {checkpoint}")
                attempted = int(restored["attempted"])
                failed = int(restored["failed"])
                batches = int(restored["batches"])
                estimate_moments = OnlineMoments.from_dict(restored["estimate_moments"])
                squared_error_moments = OnlineMoments.from_dict(restored["squared_error_moments"])
                coverage_moments = BinaryMoments(**restored["coverage_moments"])

        started = time.perf_counter()
        stop_reason = "maximum replications reached"
        stopped_early = False
        z = NormalDist().inv_cdf(0.5 + self.config.confidence_level / 2)

        while attempted < self.config.max_reps:
            current_size = min(self.config.batch_size, self.config.max_reps - attempted)
            batch = self._execute_batch(
                design,
                condition,
                condition_index,
                batches,
                current_size,
                backend,
            )
            estimates, ses, valid, batch_warnings = validate_batch(
                batch,
                backend,
                expected_size=current_size,
                finite_tolerance=self.config.finite_tolerance,
            )
            warnings.extend(x for x in batch_warnings if x not in warnings)
            attempted += current_size
            failed += int(np.sum(~valid))
            valid_estimates = estimates[valid]
            estimate_moments.update(valid_estimates)
            squared_error_moments.update((valid_estimates - truth) ** 2)
            if ses is not None:
                valid_ses = ses[valid]
                coverage_moments.update(np.abs(valid_estimates - truth) <= z * valid_ses)
            batches += 1

            if checkpoint is not None and batches % self.config.checkpoint_every_batches == 0:
                self._write_checkpoint(
                    checkpoint,
                    condition,
                    backend,
                    attempted,
                    failed,
                    batches,
                    estimate_moments,
                    squared_error_moments,
                    coverage_moments,
                )

            reached, precision_description = self._precision_reached(
                estimate_moments,
                coverage_moments,
            )
            if reached:
                stopped_early = attempted < self.config.max_reps
                stop_reason = f"precision target reached ({precision_description})"
                break

        if checkpoint is not None:
            self._write_checkpoint(
                checkpoint,
                condition,
                backend,
                attempted,
                failed,
                batches,
                estimate_moments,
                squared_error_moments,
                coverage_moments,
            )

        # Explicitly finish queued accelerator work before recording runtime.
        # validate_batch currently transfers estimates to CPU and therefore
        # also blocks, but this synchronization keeps the timing contract
        # correct for future designs that may validate entirely on-device.
        backend.synchronize()
        valid_n = estimate_moments.n
        estimate_mean = estimate_moments.mean if valid_n else float("nan")
        return ConditionResult(
            condition_index=condition_index,
            condition=condition,
            truth=truth,
            backend=backend.name,
            precision=backend.dtype_name,
            attempted_reps=attempted,
            valid_reps=valid_n,
            failed_reps=failed,
            estimate_mean=estimate_mean,
            bias=estimate_mean - truth,
            empirical_sd=estimate_moments.sd,
            rmse=math.sqrt(squared_error_moments.mean) if squared_error_moments.n else float("nan"),
            estimate_mcse=estimate_moments.mcse,
            coverage=coverage_moments.proportion if coverage_moments.n else None,
            coverage_mcse=coverage_moments.mcse if coverage_moments.n else None,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            runtime_seconds=time.perf_counter() - started,
            batches=batches,
            warnings=warnings,
        )

    def _stopping_signature(self) -> dict[str, Any]:
        return {
            "target_mcse": self.config.target_mcse,
            "target_coverage_mcse": self.config.target_coverage_mcse,
            "stopping_rule": self.config.stopping_rule,
            "min_reps": self.config.min_reps,
            "max_reps": self.config.max_reps,
            "confidence_level": self.config.confidence_level,
        }

    def _precision_reached(
        self,
        estimate_moments: OnlineMoments,
        coverage_moments: BinaryMoments,
    ) -> tuple[bool, str]:
        if estimate_moments.n < self.config.min_reps:
            return False, "minimum replications not reached"
        checks: list[tuple[str, float, float, bool]] = []
        if self.config.target_mcse is not None:
            checks.append(
                (
                    "estimate MCSE",
                    estimate_moments.mcse,
                    self.config.target_mcse,
                    estimate_moments.mcse <= self.config.target_mcse,
                )
            )
        if self.config.target_coverage_mcse is not None:
            checks.append(
                (
                    "coverage MCSE",
                    coverage_moments.mcse,
                    self.config.target_coverage_mcse,
                    coverage_moments.n > 0
                    and coverage_moments.mcse <= self.config.target_coverage_mcse,
                )
            )
        if not checks:
            return False, "no precision target configured"
        reached = (
            all(item[3] for item in checks)
            if self.config.stopping_rule == "all"
            else any(item[3] for item in checks)
        )
        description = "; ".join(
            f"{name} {value:.6g} <= {target:.6g}" for name, value, target, _ in checks
        )
        return reached, description

    def _write_checkpoint(
        self,
        path,
        condition,
        backend,
        attempted,
        failed,
        batches,
        estimate_moments,
        squared_error_moments,
        coverage_moments,
    ) -> None:
        save_checkpoint(
            path,
            {
                "schema_version": "1.1",
                "condition": condition,
                "backend": backend.name,
                "precision": backend.dtype_name,
                "seed": self.config.seed,
                "batch_size": self.config.batch_size,
                "stopping": self._stopping_signature(),
                "attempted": attempted,
                "failed": failed,
                "batches": batches,
                "estimate_moments": estimate_moments.as_dict(),
                "squared_error_moments": squared_error_moments.as_dict(),
                "coverage_moments": {
                    "n": coverage_moments.n,
                    "successes": coverage_moments.successes,
                },
            },
        )
