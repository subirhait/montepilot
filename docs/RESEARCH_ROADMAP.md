# Research roadmap

## Central research question

Can a verified, device-aware compiler transform statistical Monte Carlo studies
into efficient heterogeneous-computing workflows without changing their
inferential targets or compromising reproducibility?

## Methodological contributions required beyond the MVP

1. **Backend selection under uncertainty.** Develop a cost model that predicts
   runtime and memory for CPU, multicore CPU, and GPU execution from short,
   censored probes.
2. **Adaptive allocation across design cells.** Allocate new replications to
   cells according to decision-relevant Monte Carlo uncertainty rather than an
   equal fixed replication count.
3. **Valid sequential precision control.** Establish conditions under which
   adaptive stopping controls the error of reported bias, coverage, power, and
   Type I error summaries.
4. **Semantics-preserving vectorization.** Define and test transformations from
   replication loops to batched tensor graphs, including independent random
   streams and failure handling.
5. **Verified AI assistance.** Evaluate agents that propose code transformations
   while a deterministic verifier checks shape, seed, device, numerical-parity,
   and estimand invariants.

## Two-paper strategy

### Paper 1: statistical computing method

Provisional title: **Adaptive Heterogeneous Monte Carlo: Precision-Aware
Allocation and Verified CPU/GPU Execution for Statistical Simulation**

Core evidence: theory or guarantees, ablation studies, calibrated simulation
experiments, and comparisons with sequential R/Python loops, multicore
execution, and hand-vectorized implementations.

### Paper 2: AI evaluation benchmark

Provisional title: **Fast Is Not Enough: Benchmarking AI Agents for
Statistically Valid Simulation Optimization**

Benchmark domains: causal inference, multilevel models, mediation,
psychometrics, survival analysis, and machine learning. Every task should have
frozen reference outputs and hidden tests for random-stream independence,
coverage, estimand preservation, numerical stability, and failure reporting.

## MVP limitations

- Only scalar estimands are summarized.
- Auto mode compares NumPy CPU with available CUDA and Intel XPU accelerators;
  broader parity and device coverage remain to be validated.
- Benchmark-suite workloads now include normal means, batched OLS, causal IPW,
  and a nonparametric bootstrap, but publication claims require validation on
  multiple CPU and GPU architectures.
- Checkpoint compatibility is not yet version-migrated.
- No cluster or cloud scheduler is implemented.
- The advisor interface is pluggable, but the default advisor is deliberately
  rule-based and cannot modify a design.
- GPU claims require real-device benchmarks before publication.
