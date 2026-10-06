"""Calibrated, multi-seed benchmark harness for publication evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_suite import Workload, execute, workloads
from montepilot.backends import available_backends
from montepilot.benchmarking import (
    calibrated_iterations,
    refined_iterations,
    speedup_summary,
    timing_summary,
    validation_summary,
)
from montepilot.cli import doctor


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_timed(
    workload: Workload,
    backend: str,
    seed: int,
    precision: str,
) -> tuple[float, dict]:
    started = time.perf_counter()
    metrics = execute(workload, backend, seed, precision)
    return time.perf_counter() - started, metrics


def measure_warm_backend(
    workload: Workload,
    backend: str,
    *,
    warmups: int,
    repeats: int,
    minimum_block_seconds: float,
    maximum_inner_iterations: int,
    seed: int,
    precision: str,
) -> tuple[list[dict], dict]:
    for warmup in range(warmups):
        execute(workload, backend, seed + 8_000_000 + warmup, precision)
    probe_seconds, _ = run_timed(workload, backend, seed + 7_000_000, precision)
    inner_iterations = calibrated_iterations(
        probe_seconds,
        minimum_block_seconds=minimum_block_seconds,
        maximum_iterations=maximum_inner_iterations,
    )
    initial_iterations = inner_iterations

    rows = []
    calibration_retries = 0
    for repeat in range(repeats):
        attempt = 0
        while True:
            block_started = time.perf_counter()
            validation_metrics = None
            for iteration in range(inner_iterations):
                current_seed = seed + repeat * 100_000 + iteration
                metrics = execute(workload, backend, current_seed, precision)
                if iteration == 0:
                    validation_metrics = metrics
            block_seconds = time.perf_counter() - block_started
            target_reached = block_seconds >= minimum_block_seconds
            if (
                target_reached
                or inner_iterations >= maximum_inner_iterations
                or attempt >= 5
            ):
                break
            updated_iterations = refined_iterations(
                inner_iterations,
                block_seconds,
                minimum_block_seconds=minimum_block_seconds,
                maximum_iterations=maximum_inner_iterations,
            )
            if updated_iterations == inner_iterations:
                break
            inner_iterations = updated_iterations
            calibration_retries += 1
            attempt += 1
        assert validation_metrics is not None
        rows.append(
            {
                "workload": workload.name,
                "family": workload.family,
                "backend": backend,
                "precision": precision,
                "repeat": repeat + 1,
                "validation_seed": seed + repeat * 100_000,
                "reps": workload.reps,
                "batch_size": workload.batch_size,
                "size_metric": workload.size_metric,
                "inner_iterations": inner_iterations,
                "block_seconds": block_seconds,
                "seconds_per_execution": block_seconds / inner_iterations,
                "block_target_reached": target_reached,
                **validation_metrics,
            }
        )
    calibration = {
        "workload": workload.name,
        "backend": backend,
        "precision": precision,
        "probe_seconds": probe_seconds,
        "minimum_block_seconds": minimum_block_seconds,
        "initial_iterations": initial_iterations,
        "final_iterations": inner_iterations,
        "calibration_retries": calibration_retries,
        "maximum_inner_iterations": maximum_inner_iterations,
        "all_recorded_blocks_reached_target": all(
            row["block_target_reached"] for row in rows
        ),
    }
    return rows, calibration


def fresh_process_runs(
    workload: Workload,
    backend: str,
    *,
    profile: str,
    repeats: int,
    seed: int,
    precision: str,
) -> list[dict]:
    worker = Path(__file__).with_name("benchmark_worker.py")
    rows = []
    for repeat in range(repeats):
        command = [
            sys.executable,
            str(worker),
            "--profile",
            profile,
            "--workload",
            workload.name,
            "--backend",
            backend,
            "--seed",
            str(seed + repeat),
            "--precision",
            precision,
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        fresh_process_seconds = time.perf_counter() - started
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        rows.append(
            {
                "workload": workload.name,
                "backend": backend,
                "precision": payload["precision"],
                "repeat": repeat + 1,
                "seed": seed + repeat,
                "fresh_process_seconds": fresh_process_seconds,
                "engine_seconds": payload["runtime_seconds"],
                "startup_and_import_seconds": max(
                    0.0, fresh_process_seconds - float(payload["runtime_seconds"])
                ),
                "valid_reps": payload["valid_reps"],
                "failed_reps": payload["failed_reps"],
            }
        )
    return rows


def summarize_warm(
    rows: list[dict],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["backend"])].append(row)

    timing_rows = []
    validation_rows = []
    for group_index, ((workload, backend), group) in enumerate(sorted(groups.items())):
        timing = timing_summary(
            [row["seconds_per_execution"] for row in group],
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + group_index,
        )
        median_seconds = float(timing["median_seconds"])
        size_metric = group[0].get("size_metric")
        timing_rows.append(
            {
                "workload": workload,
                "family": group[0]["family"],
                "backend": backend,
                "precision": group[0]["precision"],
                "reps": group[0]["reps"],
                "batch_size": group[0]["batch_size"],
                "size_metric": size_metric,
                "inner_iterations_min": min(row["inner_iterations"] for row in group),
                "inner_iterations_max": max(row["inner_iterations"] for row in group),
                "all_blocks_reached_target": all(
                    row["block_target_reached"] for row in group
                ),
                **timing,
                "observations_per_second": (
                    float(size_metric) / median_seconds if size_metric else ""
                ),
            }
        )
        validation_rows.append(
            {
                "workload": workload,
                "backend": backend,
                "precision": group[0]["precision"],
                **validation_summary(group),
            }
        )

    speedup_rows = []
    for workload_index, workload in enumerate(sorted({key[0] for key in groups})):
        numpy_group = groups.get((workload, "numpy"))
        if not numpy_group:
            continue
        reference = [row["seconds_per_execution"] for row in numpy_group]
        candidates = sorted(
            backend
            for name, backend in groups
            if name == workload and backend != "numpy"
        )
        for candidate_index, candidate in enumerate(candidates):
            candidate_values = [
                row["seconds_per_execution"]
                for row in groups[(workload, candidate)]
            ]
            speedup_rows.append(
                {
                    "workload": workload,
                    "reference": "numpy",
                    "candidate": candidate,
                    **speedup_summary(
                        reference,
                        candidate_values,
                        confidence_level=confidence_level,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=seed + workload_index * 100 + candidate_index,
                    ),
                }
            )
    return timing_rows, speedup_rows, validation_rows


def summarize_cold(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["backend"])].append(row)
    out = []
    for (workload, backend), group in sorted(groups.items()):
        fresh = [row["fresh_process_seconds"] for row in group]
        engine = [row["engine_seconds"] for row in group]
        overhead = [row["startup_and_import_seconds"] for row in group]
        out.append(
            {
                "workload": workload,
                "backend": backend,
                "precision": group[0]["precision"],
                "runs": len(group),
                "median_fresh_process_seconds": statistics.median(fresh),
                "median_engine_seconds": statistics.median(engine),
                "median_startup_and_import_seconds": statistics.median(overhead),
                "min_fresh_process_seconds": min(fresh),
                "max_fresh_process_seconds": max(fresh),
            }
        )
    return out


def markdown_summary(
    timing_rows: list[dict],
    speedup_rows: list[dict],
    cold_rows: list[dict],
    confidence_level: float,
    precision: str,
) -> str:
    lines = [
        "# MontePilot calibrated benchmark summary",
        "",
        "## Warm-state calibrated timing",
        "",
        "| Workload | Backend | Inner iterations | Median seconds | "
        f"{confidence_level:.0%} bootstrap CI | CV (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        inner_iterations = str(row["inner_iterations_min"])
        if row["inner_iterations_max"] != row["inner_iterations_min"]:
            inner_iterations += f"-{row['inner_iterations_max']}"
        lines.append(
            f"| {row['workload']} | {row['backend']} | {inner_iterations} | "
            f"{row['median_seconds']:.6f} | [{row['median_ci_low']:.6f}, "
            f"{row['median_ci_high']:.6f}] | {row['cv_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Speedup relative to NumPy",
            "",
            f"| Workload | Candidate | Speedup | {confidence_level:.0%} bootstrap CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in speedup_rows:
        lines.append(
            f"| {row['workload']} | {row['candidate']} | {row['speedup']:.3f}x | "
            f"[{row['speedup_ci_low']:.3f}, {row['speedup_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Fresh-process latency",
            "",
            "This includes Python startup, imports, backend initialization, "
            "computation, and process exit.",
            "",
            "| Workload | Backend | Median total seconds | Median engine seconds | "
            "Median startup/import seconds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cold_rows:
        lines.append(
            f"| {row['workload']} | {row['backend']} | "
            f"{row['median_fresh_process_seconds']:.6f} | {row['median_engine_seconds']:.6f} | "
            f"{row['median_startup_and_import_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Timing repetitions use distinct seeds. Accuracy summaries are reported separately.",
            (
                f"All backends requested {precision}."
                if precision != "auto"
                else "Automatic precision uses FP32 on Intel XPU and FP64 on CPU/CUDA."
            ),
            "Fresh-process latency is not a hardware-cold measurement because "
            "operating-system and driver caches may persist.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--min-block-seconds", type=float, default=0.5)
    parser.add_argument("--max-inner-iterations", type=int, default=512)
    parser.add_argument("--cold-start-repeats", type=int, default=3)
    parser.add_argument("--cold-workload")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--precision", choices=["auto", "float32", "float64"], default="auto"
    )
    parser.add_argument("--output-dir", default="benchmark_results_v04")
    parser.add_argument(
        "--workloads",
        nargs="+",
        help="Optional workload names; omit to run the complete profile.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Optional backend names; omit to use every available backend.",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("repeats must be at least two")
    if args.warmups < 0 or args.cold_start_repeats < 0:
        raise SystemExit("warmups and cold-start-repeats cannot be negative")

    suite = workloads(args.profile)
    if args.workloads:
        known = {item.name for item in suite}
        unknown = sorted(set(args.workloads) - known)
        if unknown:
            raise SystemExit(f"unknown workloads: {unknown}; available: {sorted(known)}")
        selected = set(args.workloads)
        suite = [item for item in suite if item.name in selected]
    available = available_backends()
    backends = args.backends or available
    unavailable = sorted(set(backends) - set(available))
    if unavailable:
        raise SystemExit(f"unavailable backends: {unavailable}; available: {available}")
    warm_rows = []
    calibrations = []
    failures = []
    for workload in suite:
        for backend in backends:
            try:
                rows, calibration = measure_warm_backend(
                    workload,
                    backend,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    minimum_block_seconds=args.min_block_seconds,
                    maximum_inner_iterations=args.max_inner_iterations,
                    seed=args.seed,
                    precision=args.precision,
                )
                warm_rows.extend(rows)
                calibrations.append(calibration)
            except Exception as exc:
                failures.append(
                    {
                        "workload": workload.name,
                        "backend": backend,
                        "stage": "warm_calibrated",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    timing_rows, speedup_rows, validation_rows = summarize_warm(
        warm_rows,
        confidence_level=args.confidence_level,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    cold_rows = []
    if args.cold_start_repeats:
        cold_workload_name = args.cold_workload
        if cold_workload_name is None:
            candidates = [item for item in suite if item.family == "mean_scale"]
            if not candidates:
                candidates = suite
            cold_workload_name = max(
                candidates, key=lambda item: item.size_metric or -math.inf
            ).name
        cold_workload = next(
            (item for item in suite if item.name == cold_workload_name),
            None,
        )
        if cold_workload is None:
            raise SystemExit(f"unknown cold workload: {cold_workload_name}")
        successful_pairs = {(row["workload"], row["backend"]) for row in warm_rows}
        for backend in backends:
            if (cold_workload.name, backend) not in successful_pairs:
                continue
            try:
                cold_rows.extend(
                    fresh_process_runs(
                        cold_workload,
                        backend,
                        profile=args.profile,
                        repeats=args.cold_start_repeats,
                        seed=args.seed + 6_000_000,
                        precision=args.precision,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "workload": cold_workload.name,
                        "backend": backend,
                        "stage": "fresh_process",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    cold_summary = summarize_cold(cold_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "warm_runs.csv", warm_rows)
    write_csv(output_dir / "calibration.csv", calibrations)
    write_csv(output_dir / "timing_summary.csv", timing_rows)
    write_csv(output_dir / "speedup_summary.csv", speedup_rows)
    write_csv(output_dir / "numerical_validation.csv", validation_rows)
    write_csv(output_dir / "fresh_process_runs.csv", cold_rows)
    write_csv(output_dir / "fresh_process_summary.csv", cold_summary)
    write_csv(output_dir / "failures.csv", failures)
    markdown = markdown_summary(
        timing_rows,
        speedup_rows,
        cold_summary,
        args.confidence_level,
        args.precision,
    )
    (output_dir / "benchmark_summary.md").write_text(markdown, encoding="utf-8")
    report = {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "minimum_block_seconds": args.min_block_seconds,
        "maximum_inner_iterations": args.max_inner_iterations,
        "cold_start_repeats": args.cold_start_repeats,
        "confidence_level": args.confidence_level,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "precision": args.precision,
        "selected_backends": backends,
        "environment": doctor(),
        "calibration": calibrations,
        "warm_runs": warm_rows,
        "timing_summary": timing_rows,
        "speedup_summary": speedup_rows,
        "numerical_validation": validation_rows,
        "fresh_process_runs": cold_rows,
        "fresh_process_summary": cold_summary,
        "failures": failures,
        "precision_note": (
            f"All backends requested {args.precision}."
            if args.precision != "auto"
            else "Automatic precision uses FP32 on Intel XPU and FP64 on CPU/CUDA."
        ),
    }
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(markdown)
    print(f"\nFull results: {report_path.resolve()}")


if __name__ == "__main__":
    main()
