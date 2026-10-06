# Frozen pre-submission benchmark protocol: MontePilot 0.6.2

## Purpose and status

This document prespecifies the logistic-IRLS and NVIDIA CUDA evidence to be
collected for the MontePilot software paper. It must be committed and tagged
as `protocol-v0.6.2` before any timed full run. The machine-readable source of
the fixed values is `benchmarks/protocol_v062.json`.

The protocol is intentionally separate from the adaptive-stopping methods
paper. Adaptive stopping is evaluated here only as an implemented software
capability; no new sequential-inference claim is introduced.

## Frozen design

Five master seeds are used: 20261002, 20361002, 20461002, 20561002, and
20661002. Each timed workload receives one warm-up. Warm-state results use five
calibrated timing blocks of at least 0.5 seconds, with at most 512 inner
executions. Fresh-process latency uses five independent process launches. The
median, coefficient of variation, and a 95% percentile bootstrap interval for
the median are reported. Speedup is the ratio of the NumPy median to the
candidate-backend median, with 5,000 bootstrap resamples.

The laptop study uses NumPy, Torch CPU, and Torch XPU at matched float32. The
A100 study uses NumPy, Torch CPU, and Torch CUDA at matched float32 and again at
matched float64. Every backend, precision, condition, and metric specified in
the JSON file must appear in the supplement, including failures or unfavorable
results. A missing or unavailable backend is reported as such and is not
silently removed.

## Logistic-IRLS conditions

Each replication contains one normally distributed predictor and an intercept.
Binary responses are generated from the stated logistic model.

| Condition | n | Intercept | Slope | Replications | Batch size |
|---|---:|---:|---:|---:|---:|
| Regular | 1,000 | -0.5 | 1.0 | 5,000 | 1,000 |
| Rare event | 500 | -4.0 | 1.5 | 5,000 | 1,000 |
| Near separation | 100 | 0.0 | 8.0 | 5,000 | 1,000 |

The IRLS estimate is declared converged when

`max_abs_step <= 1e-6 * (1 + max_abs_coefficient)`.

The maximum iteration count is 50. Fitted probabilities are clipped to
`[1e-6, 1 - 1e-6]` during the numerical iteration. A replication fails if any
coefficient, Newton step, information-matrix component, or standard error is
non-finite; if the weighted-information determinant is not greater than
`1e-12`; if either absolute coefficient exceeds 25; or if the convergence
criterion is not met within 50 iterations. Exceeding the coefficient bound is
also the operational definition of separation for this benchmark.

Failed replications remain in the attempted count and failure-rate denominator.
They are excluded from bias, RMSE, empirical standard deviation, and coverage.
Failure rate and reason-specific counts are outcomes and must be reported.
Iteration counts are summarized among converged replications.

## Controlled-input comparisons

Controlled inputs are generated once in NumPy float64 and supplied unchanged
to each backend. The NumPy float64 result is the numerical reference. For IRLS,
estimate differences are computed only for replications that converge on both
the reference and the candidate. A disagreement in convergence status is
reported separately; it is never silently discarded. If no replication is
jointly converged, the numerical comparison is reported as unavailable and the
condition fails validation.

The mixed comparison rule is

`abs(candidate - reference) <= absolute_tolerance + relative_tolerance * abs(reference)`.

The fixed generic tolerances are:

| Family | float32 absolute/relative | float64 absolute/relative |
|---|---:|---:|
| Mean and bootstrap | 5e-6 / 5e-6 | 1e-12 / 1e-12 |
| OLS and IPW | 2e-4 / 2e-4 | 1e-10 / 1e-10 |

The fixed IRLS tolerances are:

| Condition | float32 absolute/relative | float64 absolute/relative |
|---|---:|---:|
| Regular | 5e-4 / 5e-4 | 1e-7 / 1e-7 |
| Rare event | 1e-3 / 1e-3 | 5e-7 / 5e-7 |
| Near separation | 5e-3 / 5e-3 | 1e-6 / 1e-6 |

Passing numerical tolerance does not erase convergence disagreement. Both are
reported and interpreted jointly.

## Environment and timing boundary

Each run records the operating system, Python and package versions, CPU model,
logical CPU count, total memory, relevant thread environment variables, GPU
model, driver, CUDA runtime, and the complete `nvidia-smi` query when available.
Warm timing includes simulation generation, model fitting, metric extraction,
and device synchronization, including the scalar global-active check between
IRLS iterations, but excludes Python startup and imports. The fresh
process measure includes startup, imports, backend initialization, computation,
and process exit. Operating-system and driver caches may persist, so
fresh-process latency is not described as a hardware-cold measurement.

## Execution and audit rules

1. Commit this file, the JSON configuration, the harness, and tests.
2. Tag that commit `protocol-v0.6.2` and record its commit hash in
   `RESULT_PROVENANCE.md` before a full timed run.
3. Run `python benchmarks/run_frozen_protocol.py --phase laptop --smoke`.
4. Run the full laptop phase without `--smoke`.
5. On the A100, run the smoke test and then the full `a100` phase.
6. Verify the generated SHA-256 manifest after downloading the A100 results and
   before terminating the instance.
7. Preserve raw outputs. Report all prespecified conditions and outcomes.
8. Any deviation is appended to a dated deviation log; the frozen files are
   not overwritten after timing begins.

The smoke mode uses smaller counts solely to find execution errors. Smoke
results cannot be included in the manuscript.
