# MontePilot MVP verification record

Verified on 2026-10-02 in the project execution environment:

- 23 unit tests passed;
- deterministic repetition with a fixed seed passed;
- adaptive MCSE stopping passed;
- checkpoint resume and checkpoint mismatch protection passed;
- invalid and non-finite output checks passed;
- the command-line environment diagnostic passed;
- the built-in end-to-end simulation produced valid JSON;
- a Python wheel was built and installed into a fresh virtual environment;
- the installed command-line entry point executed successfully.

Not verified in this environment:

- PyTorch CPU execution, because PyTorch was not installed;
- CUDA/GPU execution, because no CUDA device was exposed;
- the R package, because R was not installed;
- Windows and macOS behavior.

User-device verification completed on Windows 11 with Python 3.11.17:

- PyTorch 2.14.1+xpu installed successfully;
- Intel XPU availability returned `True`;
- one Intel(R) Arc(TM) 140V GPU (16GB) was detected.
- MontePilot 0.2.0.dev0 selected `torch_xpu` explicitly and completed the
  normal-mean smoke test across three design conditions;
- all 600 attempted replications were valid, with zero failures and no
  warnings;
- every condition reached the requested MCSE target and stopped early.

Five repeated large mean-simulation benchmarks (20,000 replications of
5,000 observations each) produced median runtimes of 1.447 seconds for NumPy,
1.974 seconds for Torch CPU, and 0.251 seconds for Intel XPU. The ratio of
median runtimes was 5.77x for XPU versus NumPy and 7.88x for XPU versus Torch
CPU. XPU was fastest in all five runs, and every run returned 20,000 valid
replications with no failures.

Version 0.3.0.dev0 added repeated warm-up benchmarking, backend-order rotation,
scale-sweep crossover detection, batched OLS, causal-IPW, bootstrap workloads,
and JSON/CSV/Markdown benchmark artifacts. All 17 v0.3 tests passed on the
user's Windows system and the full suite completed on Intel XPU without failed
replications.

Version 0.4.0.dev0 adds calibrated minimum-duration timing blocks, deterministic
bootstrap intervals for median runtime and speedup, distinct-seed numerical
validation, fresh-process latency measurement, direct import of base-R CSV/ZIP
results, and paper-ready CSV/JSON/Markdown/SVG aggregation. The R reference now
supports multi-seed timing repetitions and paired replication-level random
draws across batched and scalar implementations.

The first calibrated v0.4 full suite completed on the Intel Arc system with
zero failures and zero warnings across 135 timed rows. Warm-state XPU speedups
relative to NumPy ranged from 10.46x for OLS to 48.09x for the largest mean
workload. Fresh-process XPU latency was also lower than NumPy for that workload.

That run exposed a calibration issue: 50 of 135 recorded blocks were shorter
than the requested 0.5 seconds because the initial probe sometimes included
extra initialization cost. Version 0.4.1.dev0 now discards an undersized block,
increases the iteration count, and reruns it. Reports record the iteration
range and whether every accepted block reached the target. The full v0.4.1
suite therefore requires one final Intel Arc execution.

Public-release blockers remain multi-platform CI, independent hardware
replication, package-name clearance, and expanded tests on installed R and
accelerator environments.

Version 0.4.2.dev0 corrects the base-R benchmark parser by replacing the
reserved output name `repeat` with `timing_repeat`. Python aggregation accepts
both names for backward compatibility. No Python hardware rerun is required.

Version 0.5.0.dev0 adds a high-level `montepilot.run()` function, explicit
precision policies, joint estimate/coverage MCSE stopping, checkpoint guards
for precision and stopping-rule changes, and a workload-aware automatic
scheduler. The scheduler probes representative design conditions, records
accelerator warm-up separately, and projects total runtime for the planned
replication workload before selecting a backend. Thirty-three unit tests pass in the
development environment.

Version 0.5.1.dev0 makes the matched-FP32 benchmark fairer by using NumPy's
native float32 normal and uniform random-number generators. Version 0.5.0
generated NumPy normal and uniform draws in float64 and then converted them,
so its FP32 results remain valid operational timings but are not the final
matched-precision comparison.

Version 0.5.2.dev0 adds a controlled-input validation suite. Each candidate
backend receives identical NumPy-generated inputs and its replication-level
mean, OLS, IPW, and bootstrap estimates are compared directly with a NumPy
float64 reference under prespecified mixed tolerances. Thirty-five unit tests
pass in the CPU development environment, and the quick NumPy float32 suite
passes.

The full validation subsequently completed on the user's Windows Intel Arc
system. All 60 comparisons passed across four workloads, three float32
backends, and five seeds, with zero execution failures and 100% of estimates
within tolerance. The largest Intel XPU absolute difference from the NumPy
float64 reference was 5.730e-7 for batched OLS. The largest scaled error was
0.014820, less than 1.5% of the prespecified pass boundary. The Intel XPU
precision-validation blocker is therefore cleared for the tested workloads.

Version 0.6.0.dev0 adds two substantive applications. The psychometric design
simulates congeneric measurements and evaluates sample Cronbach alpha against
its analytic population value. The multilevel educational design simulates a
balanced school-randomized trial with a prespecified ICC, estimates the
treatment effect from school means, and evaluates interval coverage. The
publication harness can select these workloads without repeating earlier
benchmarks. Thirty-seven CPU-side unit tests pass, and a targeted NumPy
application-benchmark smoke test completes with zero failed replications.

These unverified items are release blockers for public PyPI or CRAN submission,
but they do not prevent evaluation of the CPU-based Python MVP.
