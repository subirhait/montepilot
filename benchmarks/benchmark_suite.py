"""Publication-oriented repeated benchmark suite for MontePilot."""

from __future__ import annotations

import os

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from montepilot import RunConfig, SimulationRunner
from montepilot.backends import available_backends
from montepilot.cli import doctor
from montepilot.examples import (
    bootstrap_mean_design,
    cluster_randomized_trial_design,
    congeneric_reliability_design,
    ipw_ate_design,
    linear_regression_design,
    normal_mean_design,
)


@dataclass(frozen=True)
class Workload:
    name: str
    family: str
    reps: int
    batch_size: int
    size_metric: int | None
    design_factory: Callable


# The v0.6.0 application timings reported in the manuscript used the one-pass
# variance form.  Set MONTEPILOT_APP_VARIANCE=centered to time the stable form.
APPLICATION_VARIANCE_METHOD = os.environ.get("MONTEPILOT_APP_VARIANCE", "one_pass")


def workloads(profile: str) -> list[Workload]:
    if profile == "quick":
        scales = [(1000, 5000), (5000, 10000)]
        regression_reps, ipw_reps, bootstrap_reps = 500, 1000, 1000
    else:
        scales = [
            (250, 10000),
            (500, 10000),
            (1000, 10000),
            (2000, 10000),
            (3000, 10000),
            (5000, 20000),
        ]
        regression_reps, ipw_reps, bootstrap_reps = 3000, 5000, 5000

    out: list[Workload] = []
    for n, reps in scales:
        out.append(
            Workload(
                name=f"mean_n{n}_r{reps}",
                family="mean_scale",
                reps=reps,
                batch_size=min(2000, reps),
                size_metric=n * reps,
                design_factory=lambda n=n: normal_mean_design(sample_sizes=[n]),
            )
        )
    out.extend(
        [
            Workload(
                name="ols_n500_p8",
                family="linear_regression",
                reps=regression_reps,
                batch_size=min(250, regression_reps),
                size_metric=500 * 8 * regression_reps,
                design_factory=lambda: linear_regression_design(
                    sample_sizes=[500], predictors=8
                ),
            ),
            Workload(
                name="ipw_n1000",
                family="causal_ipw",
                reps=ipw_reps,
                batch_size=min(500, ipw_reps),
                size_metric=1000 * ipw_reps,
                design_factory=lambda: ipw_ate_design(sample_sizes=[1000]),
            ),
            Workload(
                name="bootstrap_n2000",
                family="bootstrap",
                reps=bootstrap_reps,
                batch_size=min(500, bootstrap_reps),
                size_metric=2000 * bootstrap_reps,
                design_factory=lambda: bootstrap_mean_design(resample_sizes=[2000]),
            ),
            Workload(
                name="psychometric_alpha_n1000_k20",
                family="psychometric_reliability",
                reps=1000 if profile == "quick" else 2000,
                batch_size=100,
                size_metric=(1000 * 20 * (1000 if profile == "quick" else 2000)),
                design_factory=lambda: congeneric_reliability_design(
                    sample_sizes=[1000], item_counts=[20],
                    variance_method=APPLICATION_VARIANCE_METHOD,
                ),
            ),
            Workload(
                name="cluster_trial_j100_m30",
                family="multilevel_education",
                reps=2000 if profile == "quick" else 5000,
                batch_size=500,
                size_metric=(100 * 30 * (2000 if profile == "quick" else 5000)),
                design_factory=lambda: cluster_randomized_trial_design(
                    cluster_counts=[100], cluster_sizes=[30],
                    variance_method=APPLICATION_VARIANCE_METHOD,
                ),
            ),
        ]
    )
    return out


def execute(workload: Workload, backend: str, seed: int, precision: str = "auto") -> dict:
    report = SimulationRunner(
        RunConfig(
            backend=backend,
            precision=precision,
            batch_size=workload.batch_size,
            min_reps=workload.reps,
            max_reps=workload.reps,
            target_mcse=None,
            seed=seed,
        )
    ).run(workload.design_factory())
    result = report.conditions[0]
    return {
        "precision": report.selected_precision,
        "runtime_seconds": report.total_runtime_seconds,
        "estimate_mean": result.estimate_mean,
        "truth": result.truth,
        "bias": result.bias,
        "rmse": result.rmse,
        "empirical_sd": result.empirical_sd,
        "coverage": result.coverage,
        "coverage_mcse": result.coverage_mcse,
        "valid_reps": result.valid_reps,
        "failed_reps": result.failed_reps,
        "warnings": result.warnings,
    }


def summarize(times: list[float]) -> dict:
    values = np.asarray(times, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "runs": len(values),
        "median_seconds": float(np.median(values)),
        "mean_seconds": mean,
        "sd_seconds": sd,
        "cv_percent": 100.0 * sd / mean if mean else 0.0,
        "q1_seconds": float(np.percentile(values, 25)),
        "q3_seconds": float(np.percentile(values, 75)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(
    summary_rows: list[dict], crossovers: list[dict], precision: str = "auto"
) -> str:
    lines = [
        "# MontePilot benchmark summary",
        "",
        "| Workload | Backend | Median seconds | CV (%) | Speedup vs NumPy | Failed reps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        speedup = row.get("speedup_vs_numpy")
        rendered_speedup = "" if speedup is None else f"{speedup:.3f}x"
        lines.append(
            f"| {row['workload']} | {row['backend']} | "
            f"{row['median_seconds']:.6f} | {row['cv_percent']:.2f} | "
            f"{rendered_speedup} | {row['failed_reps']} |"
        )
    lines.extend(["", "## Estimated crossover", ""])
    if crossovers:
        for item in crossovers:
            if item["crossover_observations"] is None:
                lines.append(
                    f"- {item['backend']}: no crossover observed in the tested scale range."
                )
            else:
                lines.append(
                    f"- {item['backend']}: first faster-than-NumPy point at "
                    f"{item['crossover_observations']:,} simulated observations "
                    f"({item['workload']})."
                )
    else:
        lines.append("- No accelerated backend was available.")
    lines.extend(
        [
            "",
            (
                f"All backends requested {precision}."
                if precision != "auto"
                else "Automatic precision uses FP32 on Intel XPU and FP64 on CPU/CUDA."
            ),
            "Timing and numerical error should be interpreted jointly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--precision", choices=["auto", "float32", "float64"], default="auto"
    )
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    repeats = args.repeats or (3 if args.profile == "quick" else 5)
    if repeats < 1 or args.warmups < 0:
        raise SystemExit("repeats must be positive and warmups cannot be negative")

    backends = available_backends()
    suite = workloads(args.profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    failures: list[dict] = []

    for workload_index, workload in enumerate(suite):
        usable: list[str] = []
        for backend in backends:
            try:
                for warmup in range(args.warmups):
                    execute(workload, backend, args.seed + 900000 + warmup, args.precision)
                usable.append(backend)
            except Exception as exc:
                failures.append(
                    {
                        "workload": workload.name,
                        "backend": backend,
                        "stage": "warmup",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        for repeat in range(repeats):
            if not usable:
                break
            offset = (workload_index + repeat) % len(usable)
            order = usable[offset:] + usable[:offset]
            for backend in order:
                try:
                    metrics = execute(workload, backend, args.seed + repeat, args.precision)
                    raw_rows.append(
                        {
                            "workload": workload.name,
                            "family": workload.family,
                            "backend": backend,
                            "repeat": repeat + 1,
                            "reps": workload.reps,
                            "batch_size": workload.batch_size,
                            "size_metric": workload.size_metric,
                            **metrics,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "workload": workload.name,
                            "backend": backend,
                            "stage": f"repeat_{repeat + 1}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    summary_rows: list[dict] = []
    for workload in suite:
        by_backend: dict[str, list[dict]] = {}
        for row in raw_rows:
            if row["workload"] == workload.name:
                by_backend.setdefault(row["backend"], []).append(row)
        numpy_median = None
        if "numpy" in by_backend:
            numpy_median = statistics.median(
                row["runtime_seconds"] for row in by_backend["numpy"]
            )
        for backend, rows in by_backend.items():
            timing = summarize([row["runtime_seconds"] for row in rows])
            median = timing["median_seconds"]
            summary_rows.append(
                {
                    "workload": workload.name,
                    "family": workload.family,
                    "backend": backend,
                    "precision": rows[0]["precision"],
                    "reps": workload.reps,
                    "batch_size": workload.batch_size,
                    "size_metric": workload.size_metric,
                    **timing,
                    "speedup_vs_numpy": (
                        numpy_median / median if numpy_median is not None and median else None
                    ),
                    "median_bias": statistics.median(row["bias"] for row in rows),
                    "median_rmse": statistics.median(row["rmse"] for row in rows),
                    "failed_reps": sum(row["failed_reps"] for row in rows),
                }
            )

    crossovers: list[dict] = []
    accelerated = [name for name in backends if name in {"torch_cuda", "torch_xpu"}]
    for backend in accelerated:
        points = []
        for workload in suite:
            if workload.family != "mean_scale":
                continue
            numpy_row = next(
                (
                    row
                    for row in summary_rows
                    if row["workload"] == workload.name and row["backend"] == "numpy"
                ),
                None,
            )
            device_row = next(
                (
                    row
                    for row in summary_rows
                    if row["workload"] == workload.name and row["backend"] == backend
                ),
                None,
            )
            if numpy_row and device_row:
                points.append((workload, numpy_row, device_row))
        points.sort(key=lambda item: item[0].size_metric or math.inf)
        crossover = next(
            (
                item
                for item in points
                if item[2]["median_seconds"] < item[1]["median_seconds"]
            ),
            None,
        )
        crossovers.append(
            {
                "backend": backend,
                "crossover_observations": crossover[0].size_metric if crossover else None,
                "workload": crossover[0].name if crossover else None,
            }
        )

    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "1.0",
        "generated_at_utc": generated,
        "profile": args.profile,
        "repeats": repeats,
        "warmups": args.warmups,
        "seed": args.seed,
        "precision": args.precision,
        "environment": doctor(),
        "raw_runs": raw_rows,
        "summary": summary_rows,
        "crossovers": crossovers,
        "failures": failures,
        "precision_note": (
            f"All backends requested {args.precision}."
            if args.precision != "auto"
            else "Automatic precision uses FP32 on Intel XPU and FP64 on CPU/CUDA."
        ),
    }
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        output_dir / "benchmark_runs.csv",
        raw_rows,
        [
            "workload",
            "family",
            "backend",
            "precision",
            "repeat",
            "reps",
            "batch_size",
            "size_metric",
            "runtime_seconds",
            "estimate_mean",
            "truth",
            "bias",
            "rmse",
            "valid_reps",
            "failed_reps",
            "warnings",
        ],
    )
    write_csv(
        output_dir / "benchmark_summary.csv",
        summary_rows,
        [
            "workload",
            "family",
            "backend",
            "precision",
            "reps",
            "batch_size",
            "size_metric",
            "runs",
            "median_seconds",
            "mean_seconds",
            "sd_seconds",
            "cv_percent",
            "q1_seconds",
            "q3_seconds",
            "min_seconds",
            "max_seconds",
            "speedup_vs_numpy",
            "median_bias",
            "median_rmse",
            "failed_reps",
        ],
    )
    write_csv(
        output_dir / "benchmark_crossover.csv",
        crossovers,
        ["backend", "crossover_observations", "workload"],
    )
    summary_md = markdown_summary(summary_rows, crossovers, args.precision)
    (output_dir / "benchmark_summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)
    print(f"\nFull results: {json_path.resolve()}")


if __name__ == "__main__":
    main()
