"""Compare scalar loops with MontePilot's vectorized NumPy execution."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from montepilot import RunConfig, SimulationRunner
from montepilot.examples import normal_mean_design


def scalar_loop(reps: int, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    estimates = np.array([rng.normal(0.5, 1.0, n).mean() for _ in range(reps)])
    return time.perf_counter() - started, float(estimates.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    loop_seconds, loop_mean = scalar_loop(args.reps, args.n, args.seed)
    config = RunConfig(
        backend="numpy",
        batch_size=min(1000, args.reps),
        min_reps=args.reps,
        max_reps=args.reps,
        target_mcse=None,
        seed=args.seed,
    )
    report = SimulationRunner(config).run(normal_mean_design(sample_sizes=[args.n]))
    vector_seconds = report.total_runtime_seconds
    print(
        json.dumps(
            {
                "scalar_loop_seconds": loop_seconds,
                "vectorized_seconds": vector_seconds,
                "speedup": loop_seconds / vector_seconds,
                "scalar_mean": loop_mean,
                "vectorized_mean": report.conditions[0].estimate_mean,
                "note": "Means differ because the implementations use distinct documented RNG stream layouts.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

