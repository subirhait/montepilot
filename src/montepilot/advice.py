"""Transparent diagnostic advice and an extension point for AI advisors."""

from __future__ import annotations

from typing import Callable, Protocol

from .results import SimulationReport


class Advisor(Protocol):
    def __call__(self, report: SimulationReport) -> list[str]: ...


def rule_based_advice(report: SimulationReport) -> list[str]:
    advice: list[str] = []
    accelerated = {"torch_cuda", "torch_xpu"}
    if (
        report.config.get("backend") == "auto"
        and report.selected_backend == "numpy"
        and not accelerated.intersection(report.backend_timings)
    ):
        advice.append("No usable GPU backend was detected; the run used vectorized NumPy on CPU.")
    if report.backend_failures:
        advice.append("At least one optional backend failed its probe; inspect backend_failures before publication.")
    if report.config.get("backend") == "auto" and report.backend_projected_seconds:
        advice.append(
            "The automatic scheduler selected the backend with the lowest projected total runtime "
            "after accounting for warm-up and the planned workload."
        )
    if report.selected_precision == "float32":
        advice.append(
            "This run used float32; validate numerical agreement against a float64 CPU reference "
            "before reporting scientific conclusions."
        )
    if any(x.failed_reps > 0 for x in report.conditions):
        advice.append("Some replications failed or produced non-finite estimates; investigate the affected conditions.")
    if any(x.stop_reason == "maximum replications reached" for x in report.conditions):
        advice.append("At least one condition did not reach the requested Monte Carlo precision.")
    if all(x.stopped_early for x in report.conditions):
        advice.append("Every condition met the precision target before the maximum replication count.")
    if any(x.coverage is None for x in report.conditions):
        advice.append("Coverage was not computed because the estimator did not return standard errors.")
    if report.config.get("target_coverage_mcse") is not None and any(
        x.coverage is None for x in report.conditions
    ):
        advice.append(
            "The coverage-MCSE target could not be evaluated for conditions without standard errors."
        )
    return advice


def attach_external_advisor(
    report: SimulationReport,
    advisor: Callable[[dict], str | list[str]],
) -> list[str]:
    """Call a user-supplied local or hosted advisor with a structured payload.

    The numerical engine never calls an external service by itself. This
    explicit callback keeps data movement visible to the user.
    """

    response = advisor(report.as_dict())
    return [response] if isinstance(response, str) else list(response)
