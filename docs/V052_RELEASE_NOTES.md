# MontePilot 0.5.2 development release

This release adds controlled-input numerical validation for publication use.

## Why this was added

The publication benchmark uses native random-number generators on each backend.
Its multi-seed bias and RMSE summaries validate statistical behavior, but the
samples are not identical across NumPy, Torch CPU, and accelerator backends.
Consequently, those summaries cannot isolate floating-point disagreement.

## New evidence

`benchmarks/validate_precision.py` generates inputs once in NumPy float64 and
evaluates the same values on every selected backend. It reports direct
replication-level differences for:

- normal-mean reductions;
- batched ordinary least squares;
- inverse-probability weighted ATE estimation; and
- bootstrap-mean reductions.

The float64 NumPy result is the reference. Each workload has a prespecified
mixed absolute/relative tolerance. CSV, JSON, and Markdown outputs record every
seed and backend, along with the software and hardware environment.

This test is intentionally separate from speed benchmarking: it establishes
numerical agreement and does not claim that the candidate backend uses the same
random-number generator as NumPy.
