"""Frozen logistic-IRLS timing, accuracy, and controlled-input benchmark."""

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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from montepilot.backends import ComputeBackend, available_backends
from montepilot.benchmark_irls import (
    fit_logistic_irls,
    generate_controlled_logistic_data,
    run_native_condition,
    settings_from_protocol,
)
from montepilot.benchmarking import (
    calibrated_iterations,
    refined_iterations,
    speedup_summary,
    timing_summary,
)
from montepilot.cli import doctor
from montepilot.design import derive_seed
from montepilot.precision_validation import compare_estimates


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("protocol_id") != "montepilot-irls-cuda-v0.6.2":
        raise ValueError("unexpected protocol_id")
    return config


def smoke_config(config: dict) -> dict:
    out = deepcopy(config)
    out["master_seeds"] = out["master_seeds"][:1]
    out["timing"].update(
        {
            "warmups": 0,
            "warm_block_repeats": 2,
            "minimum_block_seconds": 0.01,
            "maximum_inner_iterations": 2,
            "fresh_process_repeats": 1,
            "bootstrap_resamples": 200,
        }
    )
    out["logistic_irls"]["distribution_seeds"] = 1
    out["logistic_irls"]["controlled_input_replications"] = 32
    for condition in out["logistic_irls"]["conditions"]:
        condition["replications"] = 32
        condition["batch_size"] = 16
        condition["n"] = min(int(condition["n"]), 100)
    return out


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


def selected_backends(config: dict, phase: str, requested: list[str] | None) -> list[str]:
    return list(requested or config[phase]["backends"])


def selected_precisions(config: dict, phase: str, requested: list[str] | None) -> list[str]:
    return list(requested or config[phase]["precisions"])


def run_warm(
    config: dict,
    backends: list[str],
    precisions: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    settings = settings_from_protocol(config)
    timing = config["timing"]
    seeds = config["master_seeds"]
    repeats = int(timing["warm_block_repeats"])
    warmups = int(timing["warmups"])
    raw: list[dict] = []
    calibration: list[dict] = []
    failures: list[dict] = []
    available = set(available_backends())
    for precision in precisions:
        for condition_index, condition in enumerate(config["logistic_irls"]["conditions"]):
            for backend_name in backends:
                if backend_name not in available:
                    failures.append(
                        {
                            "condition": condition["name"],
                            "backend": backend_name,
                            "precision": precision,
                            "stage": "availability",
                            "error": f"backend unavailable; available={sorted(available)}",
                        }
                    )
                    continue
                backend = ComputeBackend(backend_name, precision=precision)
                try:
                    for warmup in range(warmups):
                        run_native_condition(
                            backend,
                            condition,
                            settings,
                            master_seed=seeds[0] + 8_000_000 + warmup,
                            condition_index=condition_index,
                        )
                    _, probe_seconds = run_native_condition(
                        backend,
                        condition,
                        settings,
                        master_seed=seeds[0] + 7_000_000,
                        condition_index=condition_index,
                    )
                    inner = calibrated_iterations(
                        probe_seconds,
                        minimum_block_seconds=float(timing["minimum_block_seconds"]),
                        maximum_iterations=int(timing["maximum_inner_iterations"]),
                    )
                    initial_inner = inner
                    retries = 0
                    for repeat in range(repeats):
                        attempt = 0
                        while True:
                            block_started = time.perf_counter()
                            retained_metrics = None
                            for iteration in range(inner):
                                master_seed = seeds[repeat % len(seeds)] + iteration
                                metrics, _ = run_native_condition(
                                    backend,
                                    condition,
                                    settings,
                                    master_seed=master_seed,
                                    condition_index=condition_index,
                                )
                                if iteration == 0:
                                    retained_metrics = metrics
                            block_seconds = time.perf_counter() - block_started
                            reached = block_seconds >= float(timing["minimum_block_seconds"])
                            if reached or inner >= int(timing["maximum_inner_iterations"]) or attempt >= 5:
                                break
                            updated = refined_iterations(
                                inner,
                                block_seconds,
                                minimum_block_seconds=float(timing["minimum_block_seconds"]),
                                maximum_iterations=int(timing["maximum_inner_iterations"]),
                            )
                            if updated == inner:
                                break
                            inner = updated
                            retries += 1
                            attempt += 1
                        assert retained_metrics is not None
                        raw.append(
                            {
                                **retained_metrics,
                                "n": condition["n"],
                                "batch_size": condition["batch_size"],
                                "repeat": repeat + 1,
                                "inner_iterations": inner,
                                "block_seconds": block_seconds,
                                "seconds_per_execution": block_seconds / inner,
                                "block_target_reached": reached,
                            }
                        )
                    calibration.append(
                        {
                            "condition": condition["name"],
                            "backend": backend_name,
                            "precision": precision,
                            "probe_seconds": probe_seconds,
                            "initial_iterations": initial_inner,
                            "final_iterations": inner,
                            "calibration_retries": retries,
                            "minimum_block_seconds": timing["minimum_block_seconds"],
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "condition": condition["name"],
                            "backend": backend_name,
                            "precision": precision,
                            "stage": "warm_timing",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return raw, calibration, failures


def summarize_warm(config: dict, raw: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in raw:
        groups[(row["condition"], row["precision"], row["backend"])].append(row)
    timing_rows: list[dict] = []
    validation_rows: list[dict] = []
    for index, (key, group) in enumerate(sorted(groups.items())):
        condition, precision, backend = key
        summary = timing_summary(
            [row["seconds_per_execution"] for row in group],
            confidence_level=float(config["timing"]["confidence_level"]),
            bootstrap_resamples=int(config["timing"]["bootstrap_resamples"]),
            seed=int(config["master_seeds"][0]) + index,
        )
        timing_rows.append(
            {
                "condition": condition,
                "precision": precision,
                "backend": backend,
                "inner_iterations_min": min(row["inner_iterations"] for row in group),
                "inner_iterations_max": max(row["inner_iterations"] for row in group),
                "all_blocks_reached_target": all(row["block_target_reached"] for row in group),
                **summary,
            }
        )
        validation_rows.append(
            {
                "condition": condition,
                "precision": precision,
                "backend": backend,
                "seeds": len(group),
                "mean_bias": float(np.mean([row["bias"] for row in group])),
                "mean_rmse": float(np.mean([row["rmse"] for row in group])),
                "mean_coverage": float(np.mean([row["coverage"] for row in group])),
                "mean_failure_rate": float(np.mean([row["failure_rate"] for row in group])),
                "total_attempted_reps": int(sum(row["attempted_reps"] for row in group)),
                "total_failed_reps": int(sum(row["failed_reps"] for row in group)),
                "mean_iterations": float(np.mean([row["mean_iterations"] for row in group])),
                "max_iterations_observed": int(max(row["max_iterations_observed"] for row in group)),
                "fail_nonfinite": int(sum(row["fail_nonfinite"] for row in group)),
                "fail_singular_information": int(sum(row["fail_singular_information"] for row in group)),
                "fail_coefficient_bound": int(sum(row["fail_coefficient_bound"] for row in group)),
                "fail_iteration_cap": int(sum(row["fail_iteration_cap"] for row in group)),
            }
        )
    speedups: list[dict] = []
    for condition, precision, backend in sorted(groups):
        if backend == "numpy":
            continue
        reference = groups.get((condition, precision, "numpy"))
        if not reference:
            continue
        candidate = groups[(condition, precision, backend)]
        speedups.append(
            {
                "condition": condition,
                "precision": precision,
                "reference": "numpy",
                "candidate": backend,
                **speedup_summary(
                    [row["seconds_per_execution"] for row in reference],
                    [row["seconds_per_execution"] for row in candidate],
                    confidence_level=float(config["timing"]["confidence_level"]),
                    bootstrap_resamples=int(config["timing"]["bootstrap_resamples"]),
                    seed=int(config["master_seeds"][0]) + len(speedups),
                ),
            }
        )
    return timing_rows, validation_rows, speedups


def controlled_input_rows(
    config: dict,
    backends: list[str],
    precisions: list[str],
) -> tuple[list[dict], list[dict]]:
    settings = settings_from_protocol(config)
    repetitions = int(config["logistic_irls"]["controlled_input_replications"])
    seed_count = int(config["logistic_irls"]["distribution_seeds"])
    available = set(available_backends())
    rows: list[dict] = []
    failures: list[dict] = []
    reference_backend = ComputeBackend("numpy", precision="float64")
    for condition_index, condition in enumerate(config["logistic_irls"]["conditions"]):
        for seed_index, master_seed in enumerate(config["master_seeds"][:seed_count]):
            data_seed = derive_seed(master_seed, condition_index, 0, 99)
            x, y = generate_controlled_logistic_data(
                seed=data_seed,
                replications=repetitions,
                sample_size=int(condition["n"]),
                intercept=float(condition["intercept"]),
                slope=float(condition["slope"]),
            )
            reference = fit_logistic_irls(x, y, reference_backend, settings)
            for precision in precisions:
                tolerance = config["controlled_input_tolerances"]["logistic_irls"][precision][condition["name"]]
                for backend_name in backends:
                    if backend_name not in available:
                        continue
                    try:
                        candidate = fit_logistic_irls(
                            x,
                            y,
                            ComputeBackend(backend_name, precision=precision),
                            settings,
                        )
                        joint = reference.converged & candidate.converged
                        disagreement = reference.converged != candidate.converged
                        if np.any(joint):
                            comparison = compare_estimates(
                                reference.estimates[joint],
                                candidate.estimates[joint],
                                absolute_tolerance=float(tolerance["absolute"]),
                                relative_tolerance=float(tolerance["relative"]),
                            )
                        else:
                            comparison = {
                                "replications": 0,
                                "reference_mean": float("nan"),
                                "candidate_mean": float("nan"),
                                "mean_difference": float("nan"),
                                "max_absolute_difference": float("nan"),
                                "rmse_difference": float("nan"),
                                "relative_l2_error": float("nan"),
                                "within_tolerance_fraction": float("nan"),
                                "max_scaled_error": float("nan"),
                                "passed": False,
                            }
                        rows.append(
                            {
                                "condition": condition["name"],
                                "backend": backend_name,
                                "precision": precision,
                                "seed_index": seed_index + 1,
                                "master_seed": master_seed,
                                "absolute_tolerance": tolerance["absolute"],
                                "relative_tolerance": tolerance["relative"],
                                "reference_converged": int(np.sum(reference.converged)),
                                "candidate_converged": int(np.sum(candidate.converged)),
                                "jointly_converged": int(np.sum(joint)),
                                "convergence_disagreements": int(np.sum(disagreement)),
                                "convergence_disagreement_rate": float(np.mean(disagreement)),
                                **comparison,
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "condition": condition["name"],
                                "backend": backend_name,
                                "precision": precision,
                                "stage": "controlled_input",
                                "master_seed": master_seed,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    return rows, failures


def fresh_process_rows(
    protocol_path: Path,
    config: dict,
    backends: list[str],
    precisions: list[str],
    *,
    smoke: bool,
) -> tuple[list[dict], list[dict]]:
    condition = config["logistic_irls"]["conditions"][0]
    repeats = int(config["timing"]["fresh_process_repeats"])
    available = set(available_backends())
    rows: list[dict] = []
    failures: list[dict] = []
    for precision in precisions:
        for backend in backends:
            if backend not in available:
                continue
            for repeat in range(repeats):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--config",
                    str(protocol_path.resolve()),
                    "--condition",
                    str(condition["name"]),
                    "--backend",
                    backend,
                    "--precision",
                    precision,
                    "--seed",
                    str(config["master_seeds"][repeat % len(config["master_seeds"])]),
                ]
                if smoke:
                    command.append("--smoke")
                started = time.perf_counter()
                try:
                    completed = subprocess.run(command, check=True, capture_output=True, text=True)
                    total = time.perf_counter() - started
                    payload = json.loads(completed.stdout.strip().splitlines()[-1])
                    rows.append(
                        {
                            "condition": condition["name"],
                            "backend": backend,
                            "precision": precision,
                            "repeat": repeat + 1,
                            "fresh_process_seconds": total,
                            "engine_seconds": payload["engine_seconds"],
                            "startup_and_import_seconds": max(0.0, total - payload["engine_seconds"]),
                            "valid_reps": payload["valid_reps"],
                            "failed_reps": payload["failed_reps"],
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "condition": condition["name"],
                            "backend": backend,
                            "precision": precision,
                            "stage": "fresh_process",
                            "repeat": repeat + 1,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return rows, failures


def markdown_summary(
    timing_rows: list[dict],
    speedups: list[dict],
    validation: list[dict],
    controlled: list[dict],
    fresh: list[dict],
    failures: list[dict],
) -> str:
    lines = [
        "# Frozen logistic-IRLS benchmark summary",
        "",
        "## Warm-state timing",
        "",
        "| Condition | Precision | Backend | Median seconds | 95% CI | CV (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            f"| {row['condition']} | {row['precision']} | {row['backend']} | "
            f"{row['median_seconds']:.6f} | [{row['median_ci_low']:.6f}, "
            f"{row['median_ci_high']:.6f}] | {row['cv_percent']:.2f} |"
        )
    lines.extend(["", "## Speedup relative to NumPy", "", "| Condition | Precision | Candidate | Speedup | 95% CI |", "|---|---:|---:|---:|---:|"])
    for row in speedups:
        lines.append(
            f"| {row['condition']} | {row['precision']} | {row['candidate']} | "
            f"{row['speedup']:.3f}x | [{row['speedup_ci_low']:.3f}, {row['speedup_ci_high']:.3f}] |"
        )
    lines.extend(["", "## Distributional validation", "", "| Condition | Precision | Backend | Bias | RMSE | Coverage | Failure rate | Mean iterations |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in validation:
        lines.append(
            f"| {row['condition']} | {row['precision']} | {row['backend']} | "
            f"{row['mean_bias']:.6g} | {row['mean_rmse']:.6g} | {row['mean_coverage']:.6f} | "
            f"{row['mean_failure_rate']:.4%} | {row['mean_iterations']:.2f} |"
        )
    lines.extend(["", "## Controlled-input validation", "", "| Condition | Precision | Backend | Max difference | Joint convergence | Status disagreement | Pass |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in controlled:
        lines.append(
            f"| {row['condition']} | {row['precision']} | {row['backend']} | "
            f"{row['max_absolute_difference']:.3e} | {row['jointly_converged']} | "
            f"{row['convergence_disagreement_rate']:.3%} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Fresh-process runs", ""])
    if fresh:
        lines.extend(["| Condition | Precision | Backend | Repeat | Total seconds | Engine seconds |", "|---|---:|---:|---:|---:|---:|"])
        for row in fresh:
            lines.append(
                f"| {row['condition']} | {row['precision']} | {row['backend']} | {row['repeat']} | "
                f"{row['fresh_process_seconds']:.6f} | {row['engine_seconds']:.6f} |"
            )
    else:
        lines.append("No fresh-process rows were produced.")
    lines.extend(["", "## Execution failures", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- {row.get('condition', 'unknown')} / {row.get('precision', 'unknown')} / "
                f"{row.get('backend', 'unknown')} / {row.get('stage', 'unknown')}: {row['error']}"
            )
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def worker(args) -> None:
    config = load_protocol(args.config)
    if args.smoke:
        config = smoke_config(config)
    condition = next(
        item for item in config["logistic_irls"]["conditions"] if item["name"] == args.condition
    )
    metrics, elapsed = run_native_condition(
        ComputeBackend(args.backend, precision=args.precision),
        condition,
        settings_from_protocol(config),
        master_seed=args.seed,
        condition_index=config["logistic_irls"]["conditions"].index(condition),
    )
    print(json.dumps({**metrics, "engine_seconds": elapsed}, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/protocol_v062.json")
    parser.add_argument("--phase", choices=["laptop", "a100"], default="laptop")
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--precisions", nargs="+", choices=["float32", "float64"])
    parser.add_argument("--output-dir", default="irls_benchmark_results_v062")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=["float32", "float64"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    protocol_path = Path(args.config)
    config = load_protocol(protocol_path)
    if args.smoke:
        config = smoke_config(config)
    backends = selected_backends(config, args.phase, args.backends)
    precisions = selected_precisions(config, args.phase, args.precisions)
    warm, calibration, failures = run_warm(config, backends, precisions)
    timing_rows, validation_rows, speedups = summarize_warm(config, warm)
    controlled, controlled_failures = controlled_input_rows(config, backends, precisions)
    failures.extend(controlled_failures)
    fresh, fresh_failures = fresh_process_rows(
        protocol_path, config, backends, precisions, smoke=bool(args.smoke)
    )
    failures.extend(fresh_failures)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "warm_runs.csv", warm)
    write_csv(output / "calibration.csv", calibration)
    write_csv(output / "timing_summary.csv", timing_rows)
    write_csv(output / "speedup_summary.csv", speedups)
    write_csv(output / "distribution_validation.csv", validation_rows)
    write_csv(output / "controlled_input.csv", controlled)
    write_csv(output / "fresh_process_runs.csv", fresh)
    write_csv(output / "failures.csv", failures)
    summary = markdown_summary(
        timing_rows, speedups, validation_rows, controlled, fresh, failures
    )
    (output / "benchmark_summary.md").write_text(summary, encoding="utf-8")
    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "protocol_tag": config["protocol_tag"],
        "smoke": bool(args.smoke),
        "phase": args.phase,
        "requested_backends": backends,
        "requested_precisions": precisions,
        "environment": doctor(),
        "warm_runs": warm,
        "calibration": calibration,
        "timing_summary": timing_rows,
        "speedup_summary": speedups,
        "distribution_validation": validation_rows,
        "controlled_input": controlled,
        "fresh_process_runs": fresh,
        "failures": failures,
    }
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary)
    print(f"\nFull results: {(output / 'benchmark_report.json').resolve()}")


if __name__ == "__main__":
    main()
