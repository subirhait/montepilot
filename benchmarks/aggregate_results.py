"""Combine Python and base-R benchmark runs into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from montepilot.benchmarking import (
    load_r_benchmark_rows,
    speedup_summary,
    timing_summary,
    validation_summary,
)


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


def load_python_report(path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("warm_runs", report.get("raw_runs", []))
    if not rows:
        raise ValueError(f"no benchmark runs found in {path}")
    normalized = []
    for row in rows:
        seconds = row.get("seconds_per_execution", row.get("runtime_seconds"))
        normalized.append(
            {
                "workload": row["workload"],
                "family": row.get("family", ""),
                "backend": row["backend"],
                "repeat": int(row.get("repeat", len(normalized) + 1)),
                "reps": int(row["reps"]),
                "batch_size": int(row["batch_size"]),
                "size_metric": int(row["size_metric"]) if row.get("size_metric") else None,
                "elapsed_seconds": float(seconds),
                "inner_iterations": int(row.get("inner_iterations", 1)),
                "estimate_mean": float(row["estimate_mean"]),
                "truth": float(row["truth"]),
                "bias": float(row["bias"]),
                "rmse": float(row["rmse"]),
                "valid_reps": int(row["valid_reps"]),
                "failed_reps": int(row["failed_reps"]),
                "source_type": "python",
            }
        )
    return report, normalized


def normalize_r_rows(rows: list[dict]) -> list[dict]:
    out = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        workload = f"mean_n{row['n']}_r{row['reps']}"
        key = (workload, row["implementation"])
        counters[key] += 1
        repeat = int(row.get("repeat") or counters[key])
        raw_source = str(row["source"])
        if "!" in raw_source:
            archive, member = raw_source.split("!", 1)
            source_label = f"{Path(archive).name}!{member}"
        else:
            source_label = Path(raw_source).name
        out.append(
            {
                "workload": workload,
                "family": "mean_scale",
                "backend": row["implementation"],
                "repeat": repeat,
                "validation_seed": row.get("seed"),
                "reps": row["reps"],
                "batch_size": row["batch_size"],
                "size_metric": row["n"] * row["reps"],
                "elapsed_seconds": row["elapsed_seconds"],
                "inner_iterations": 1,
                "estimate_mean": row["estimate_mean"],
                "truth": 0.5,
                "bias": row["bias"],
                "rmse": row["rmse"],
                "valid_reps": row["reps"],
                "failed_reps": 0,
                "source_type": "r_timing",
                "r_version": row["r_version"],
                "platform": row["platform"],
                "source": source_label,
            }
        )
    seeds_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in out:
        if row.get("validation_seed") is not None:
            seeds_by_group[(row["workload"], row["backend"])].add(
                int(row["validation_seed"])
            )
    for row in out:
        seeds = seeds_by_group[(row["workload"], row["backend"])]
        row["source_type"] = "r_multi_seed" if len(seeds) > 1 else "r_fixed_seed_timing"
    return out


def build_summaries(
    rows: list[dict],
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["workload"], row["backend"])].append(row)

    timing_rows = []
    validation_rows = []
    for group_index, ((workload, backend), group) in enumerate(sorted(grouped.items())):
        summary = timing_summary(
            [row["elapsed_seconds"] for row in group],
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + group_index,
        )
        size_metric = group[0].get("size_metric")
        median_seconds = float(summary["median_seconds"])
        timing_rows.append(
            {
                "workload": workload,
                "family": group[0].get("family", ""),
                "backend": backend,
                "source_type": group[0]["source_type"],
                "reps": group[0]["reps"],
                "size_metric": size_metric,
                **summary,
                "observations_per_second": (
                    float(size_metric) / median_seconds if size_metric else ""
                ),
            }
        )
        validation = validation_summary(group)
        validation_rows.append(
            {
                "workload": workload,
                "backend": backend,
                "seed_policy": (
                    "fixed seed repeated for timing"
                    if group[0]["source_type"] == "r_fixed_seed_timing"
                    else "distinct seed per repeat"
                ),
                **validation,
            }
        )

    speedup_rows = []
    workload_names = sorted({row["workload"] for row in rows})
    for workload_index, workload in enumerate(workload_names):
        available = {
            backend: group
            for (name, backend), group in grouped.items()
            if name == workload
        }
        comparisons: list[tuple[str, str]] = []
        if "numpy" in available:
            comparisons.extend(
                ("numpy", backend)
                for backend in available
                if backend != "numpy" and not backend.startswith("base_r_")
            )
        for reference in ("base_r_batched", "base_r_scalar_loop"):
            if reference in available:
                comparisons.extend(
                    (reference, candidate)
                    for candidate in ("numpy", "torch_cpu", "torch_xpu", "torch_cuda")
                    if candidate in available
                )
        for comparison_index, (reference, candidate) in enumerate(comparisons):
            result = speedup_summary(
                [row["elapsed_seconds"] for row in available[reference]],
                [row["elapsed_seconds"] for row in available[candidate]],
                confidence_level=confidence_level,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + workload_index * 100 + comparison_index,
            )
            speedup_rows.append(
                {
                    "workload": workload,
                    "reference": reference,
                    "candidate": candidate,
                    **result,
                }
            )
    return timing_rows, speedup_rows, validation_rows


def runtime_svg(rows: list[dict], workload: str) -> str:
    selected = [row for row in rows if row["workload"] == workload]
    selected.sort(key=lambda row: float(row["median_seconds"]), reverse=True)
    width = 920
    left = 205
    right = 875
    top = 70
    row_height = 54
    height = top + row_height * len(selected) + 70
    values = [float(row["median_seconds"]) for row in selected]
    log_min = math.log10(min(values))
    log_max = math.log10(max(values))
    span = max(log_max - log_min, 0.1)
    colors = {
        "base_r_batched": "#7A5195",
        "base_r_scalar_loop": "#BC5090",
        "numpy": "#003F5C",
        "torch_cpu": "#2F4B7C",
        "torch_xpu": "#FFA600",
        "torch_cuda": "#EF5675",
    }
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:Arial,sans-serif;fill:#17202A}'
            '.title{font-size:21px;font-weight:700}'
            '.label{font-size:15px}.value{font-size:14px;font-weight:700}'
            '.note{font-size:12px;fill:#5D6D7E}</style>'
        ),
        f'<text class="title" x="{left}" y="32">Median warm-state runtime (log scale)</text>',
        f'<text class="note" x="{left}" y="52">{html.escape(workload)}; shorter is faster</text>',
    ]
    for index, row in enumerate(selected):
        y = top + index * row_height
        value = float(row["median_seconds"])
        fraction = (math.log10(value) - log_min) / span
        bar_width = 34 + fraction * (right - left - 80)
        label = row["backend"].replace("_", " ")
        color = colors.get(row["backend"], "#5D6D7E")
        lines.extend(
            [
                (
                    f'<text class="label" x="{left - 12}" y="{y + 24}" '
                    f'text-anchor="end">{html.escape(label)}</text>'
                ),
                (
                    f'<rect x="{left}" y="{y + 7}" width="{bar_width:.1f}" '
                    f'height="24" rx="4" fill="{color}"/>'
                ),
                (
                    f'<text class="value" x="{left + bar_width + 9:.1f}" '
                    f'y="{y + 24}">{value:.6f} s</text>'
                ),
            ]
        )
    lines.append(
        (
            f'<text class="note" x="{left}" y="{height - 22}">'
            "Bars use log10 runtime; labels show untransformed seconds.</text>"
        )
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def markdown_report(
    timing_rows: list[dict],
    speedup_rows: list[dict],
    workload: str,
    confidence_level: float,
) -> str:
    selected = [row for row in timing_rows if row["workload"] == workload]
    selected.sort(key=lambda row: float(row["median_seconds"]))
    r_source_types = {
        row["source_type"]
        for row in selected
        if str(row["backend"]).startswith("base_r_")
    }
    r_seed_note = (
        "R and Python timing repetitions use distinct seeds."
        if "r_multi_seed" in r_source_types
        else "R timing repetitions use a fixed seed; Python repetitions use distinct seeds."
    )
    lines = [
        "# MontePilot publication benchmark summary",
        "",
        f"Representative workload: `{workload}`.",
        "",
        "| Implementation | Median seconds | Bootstrap CI | CV (%) | Observations/second |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        throughput = row.get("observations_per_second")
        rendered_throughput = f"{float(throughput):,.0f}" if throughput != "" else ""
        lines.append(
            f"| {row['backend']} | {row['median_seconds']:.6f} | "
            f"[{row['median_ci_low']:.6f}, {row['median_ci_high']:.6f}] | "
            f"{row['cv_percent']:.2f} | {rendered_throughput} |"
        )
    lines.extend(
        [
            "",
            "## Selected speedups",
            "",
            "| Reference | Candidate | Speedup | Bootstrap CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in speedup_rows:
        if row["workload"] != workload:
            continue
        lines.append(
            f"| {row['reference']} | {row['candidate']} | {row['speedup']:.2f}x | "
            f"[{row['speedup_ci_low']:.2f}, {row['speedup_ci_high']:.2f}] |"
        )
    lines.extend(
        [
            "",
            f"Intervals are percentile bootstrap intervals at the {confidence_level:.0%} level.",
            r_seed_note,
            "Intel XPU uses FP32 in this prototype, whereas the CPU implementations use FP64.",
            "The report distinguishes workload-specific evidence from a general hardware claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-report", required=True)
    parser.add_argument("--r-input", action="append", default=[])
    parser.add_argument("--output-dir", default="publication_results")
    parser.add_argument("--workload")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    python_report, python_rows = load_python_report(Path(args.python_report))
    rows = list(python_rows)
    if args.r_input:
        rows.extend(normalize_r_rows(load_r_benchmark_rows(args.r_input)))
    timing_rows, speedup_rows, validation_rows = build_summaries(
        rows,
        args.confidence_level,
        args.bootstrap_resamples,
        args.seed,
    )
    representative = args.workload
    if representative is None:
        representative = max(
            (row for row in timing_rows if row.get("size_metric")),
            key=lambda row: int(row["size_metric"]),
        )["workload"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "combined_timing_runs.csv", rows)
    write_csv(output_dir / "timing_summary.csv", timing_rows)
    write_csv(output_dir / "speedup_summary.csv", speedup_rows)
    write_csv(output_dir / "numerical_validation.csv", validation_rows)
    (output_dir / "runtime_comparison.svg").write_text(
        runtime_svg(timing_rows, representative), encoding="utf-8"
    )
    markdown = markdown_report(
        timing_rows,
        speedup_rows,
        representative,
        args.confidence_level,
    )
    (output_dir / "publication_summary.md").write_text(markdown, encoding="utf-8")
    report = {
        "schema_version": "2.0",
        "source_python_report": Path(args.python_report).name,
        "source_python_environment": python_report.get("environment", {}),
        "r_inputs": [Path(item).name for item in args.r_input],
        "confidence_level": args.confidence_level,
        "bootstrap_resamples": args.bootstrap_resamples,
        "representative_workload": representative,
        "timing_summary": timing_rows,
        "speedup_summary": speedup_rows,
        "numerical_validation": validation_rows,
    }
    (output_dir / "publication_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(markdown)
    print(f"\nArtifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
