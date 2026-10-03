"""Validate backend arithmetic against controlled NumPy float64 inputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from montepilot.backends import ComputeBackend, available_backends
from montepilot.cli import doctor
from montepilot.precision_validation import evaluate_precision_case, precision_cases


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict], failures: list[dict]) -> str:
    lines = [
        "# MontePilot controlled-input precision validation",
        "",
        "Every candidate received the same NumPy-generated data. The reference "
        "was evaluated in NumPy float64; candidates used the precision shown below.",
        "",
        "| Workload | Backend | Precision | Max abs. difference | RMSE difference | "
        "Within tolerance | Result |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['backend']} | {row['precision']} | "
            f"{row['max_absolute_difference']:.3e} | {row['rmse_difference']:.3e} | "
            f"{row['within_tolerance_fraction']:.3%} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if rows and all(bool(row["passed"]) for row in rows) and not failures:
        lines.append(
            "All evaluated replication-level estimates met the prespecified mixed "
            "absolute/relative tolerance against the NumPy float64 reference."
        )
    else:
        lines.append(
            "At least one comparison failed or could not run. Inspect the CSV and JSON "
            "files before making a numerical-agreement claim."
        )
    if failures:
        lines.extend(["", "## Execution failures", ""])
        for failure in failures:
            lines.append(
                f"- {failure['workload']} / {failure['backend']}: {failure['error']}"
            )
    lines.extend(
        [
            "",
            "This controlled-input analysis isolates arithmetic and reduction differences. "
            "It complements, rather than replaces, the multi-seed distributional validation "
            "in the publication benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--seed-step", type=int, default=100000)
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--output-dir", default="precision_validation_results")
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("seeds must be positive")

    available = available_backends()
    selected = args.backends or available
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise SystemExit(f"unavailable backends: {unknown}; available: {available}")

    rows: list[dict] = []
    failures: list[dict] = []
    for case in precision_cases(args.profile):
        for backend_name in selected:
            backend = ComputeBackend(backend_name, precision=args.precision)
            for seed_index in range(args.seeds):
                current_seed = args.seed + seed_index * args.seed_step
                try:
                    rows.append(evaluate_precision_case(case, backend, current_seed))
                except Exception as exc:
                    failures.append(
                        {
                            "workload": case.name,
                            "backend": backend_name,
                            "precision": args.precision,
                            "seed": current_seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "controlled_precision_runs.csv", rows)
    if failures:
        _write_csv(output_dir / "failures.csv", failures)
    markdown = _markdown(rows, failures)
    (output_dir / "precision_validation_summary.md").write_text(
        markdown, encoding="utf-8"
    )
    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "candidate_precision": args.precision,
        "reference": "NumPy float64 on identical controlled inputs",
        "seeds": args.seeds,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "selected_backends": selected,
        "environment": doctor(),
        "rows": rows,
        "failures": failures,
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows) and not failures,
    }
    report_path = output_dir / "precision_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nFull results: {report_path.resolve()}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
