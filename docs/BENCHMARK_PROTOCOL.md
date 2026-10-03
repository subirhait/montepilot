# MontePilot benchmark protocol

## Purpose

The benchmark separates four questions that should not be collapsed into one
speedup number:

1. How fast is repeated simulation after a backend has been initialized?
2. What latency does a user experience in a fresh Python process?
3. Does the accelerated backend retain acceptable statistical accuracy?
4. At what workload size does acceleration become worthwhile?

## Warm-state timing

`benchmarks/benchmark_publication.py` performs the requested warm-ups and then
times repeated blocks. A probe determines how many complete executions are
needed to reach `--min-block-seconds`; the default is 0.5 seconds. Reported
seconds are block time divided by the number of complete executions. This
reduces timer and scheduling noise for accelerator workloads that otherwise
finish in only a few milliseconds.

Because the first probe can include initialization overhead, the harness checks
the observed duration of every candidate block. An undersized block is
discarded, its inner-iteration count is increased with a safety margin, and it
is rerun. The report records the observed iteration range and whether all
accepted blocks reached the requested duration.

Every execution includes simulation generation, estimation, device-to-host
transfer, validation, and online metric updates. The engine explicitly
synchronizes accelerator work before recording condition runtime.

The primary timing estimand is median seconds per complete execution. Reports
include the coefficient of variation and a deterministic percentile-bootstrap
confidence interval for the median. Speedup is the ratio of the reference and
candidate median runtimes, with an independent bootstrap interval.

## Fresh-process latency

The selected representative workload is also launched in separate Python
processes. Parent-process wall time includes interpreter startup, imports,
backend initialization, simulation, validation, and process exit. This is
called *fresh-process latency*, not hardware-cold latency, because operating
system, filesystem, and driver caches can persist between processes.

## Numerical validation

Timing repetitions use distinct deterministic seeds. Bias, RMSE, valid
replications, and failed replications are summarized separately from runtime.
Different compute backends use different random-number generators, so equality
of replication-level estimates is not expected. Comparisons are distributional
unless a design explicitly supplies shared input arrays.

With `--precision auto`, Intel client XPU execution uses FP32 because FP64
kernels are not consistently available, while NumPy, Torch CPU, and Torch CUDA
use FP64. Version 0.5 adds `--precision float32` to the benchmark suites so all
backends can be compared at matched FP32 precision. The automatic-precision
results remain useful operational evidence, but runtime and statistical error
must be interpreted jointly.

## Base-R reference

`r/benchmarks/benchmark_base_r.R` supports `--timing-repeats` and changes the
seed by `--seed-step` for each repetition. The batched matrix stores one
replication per column, matching R's column-major random-number order. Batched
and scalar implementations consequently receive the same replication-level
draws for a given seed.

## Minimum reporting standard

A paper or technical report should state:

- processor, accelerator, memory, operating system, and software versions;
- workload dimensions, batch size, precision, and total simulated values;
- warm-up count, block-duration target, timing repetitions, and seed policy;
- median runtime with an uncertainty interval and variability measure;
- fresh-process latency separately from warm-state runtime;
- bias, RMSE, failures, and precision differences;
- whether speedup is a tested-workload result or a broader claim;
- whether measurements were reproduced on an independent machine.

The development Intel Arc results establish feasibility on one system. They do
not by themselves justify a universal GPU-performance claim.
