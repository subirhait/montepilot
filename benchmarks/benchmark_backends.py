"""Compare all available MontePilot CPU and GPU backends."""

from __future__ import annotations

import argparse
import json

from montepilot import RunConfig, SimulationRunner
from montepilot.backends import available_backends
from montepilot.examples import normal_mean_design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--precision", choices=["auto", "float32", "float64"], default="auto"
    )
    args = parser.parse_args()

    results = []
    for backend in available_backends():
        config = RunConfig(
            backend=backend,
            precision=args.precision,
            batch_size=min(args.batch_size, args.reps),
            min_reps=args.reps,
            max_reps=args.reps,
            target_mcse=None,
            seed=args.seed,
        )
        report = SimulationRunner(config).run(
            normal_mean_design(sample_sizes=[args.n])
        )
        condition = report.conditions[0]
        results.append(
            {
                "backend": backend,
                "precision": report.selected_precision,
                "runtime_seconds": report.total_runtime_seconds,
                "estimate_mean": condition.estimate_mean,
                "bias": condition.bias,
                "rmse": condition.rmse,
                "valid_reps": condition.valid_reps,
            }
        )

    fastest = min(results, key=lambda item: item["runtime_seconds"])["backend"]
    print(
        json.dumps(
            {
                "reps": args.reps,
                "sample_size": args.n,
                "batch_size": min(args.batch_size, args.reps),
                "requested_precision": args.precision,
                "fastest_backend": fastest,
                "results": results,
                "precision_note": (
                    f"All backends requested {args.precision}; compare runtime and numerical error."
                    if args.precision != "auto"
                    else "Automatic precision uses FP32 on Intel XPU and FP64 on CPU/CUDA; "
                    "compare runtime and numerical error."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
