# MontePilot 0.6 development release

## New applications

- Continuous congeneric reliability simulation with an analytic population
  Cronbach-alpha target.
- Two-level educational cluster-randomized trial with school random effects,
  treatment-effect standard errors, and coverage evaluation.

## Benchmark improvements

- The publication harness accepts `--workloads` so a new application can be
  benchmarked without rerunning completed workloads.
- Numerical summaries retain empirical standard deviation and summarize
  coverage across validation seeds when standard errors are available.

## Verification

- Thirty-seven unit tests pass in the CPU development environment.
- A targeted matched-float32 NumPy smoke benchmark completed for both new
  applications with zero failed replications.
- Intel XPU benchmarking of the two new applications remains to be completed.
