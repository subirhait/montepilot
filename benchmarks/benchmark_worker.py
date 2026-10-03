"""Fresh-process worker used to measure startup-inclusive benchmark latency."""

from __future__ import annotations

import argparse
import json

from benchmark_suite import execute, workloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--precision", choices=["auto", "float32", "float64"], default="auto"
    )
    args = parser.parse_args()

    selected = next(
        (item for item in workloads(args.profile) if item.name == args.workload),
        None,
    )
    if selected is None:
        raise SystemExit(f"unknown workload: {args.workload}")
    result = execute(selected, args.backend, args.seed, args.precision)
    print(
        json.dumps(
            {
                "workload": selected.name,
                "family": selected.family,
                "backend": args.backend,
                "reps": selected.reps,
                "batch_size": selected.batch_size,
                "size_metric": selected.size_metric,
                **result,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
