# MontePilot

MontePilot is an early research prototype for adaptive, reproducible Monte
Carlo simulation. It runs simulations in batches, selects an available compute
backend, monitors Monte Carlo uncertainty, and can stop once a prespecified
precision target has been achieved.

The current MVP supports:

- deterministic, independent random streams for every design condition and batch;
- NumPy CPU execution with optional PyTorch CPU/CUDA/Intel XPU execution;
- workload-aware CPU/GPU probing with warm-up cost projection and safe fallback;
- vectorized batches rather than one-replication-at-a-time loops;
- adaptive stopping based on estimate MCSE, coverage MCSE, or both;
- explicit `auto`, `float32`, and `float64` precision policies;
- bias, empirical standard deviation, RMSE, coverage, failure rate, and runtime summaries;
- checkpoint files that permit interrupted runs to resume;
- machine-readable JSON audit records and deterministic diagnostic advice;
- a command-line demo and an initial R interface package.

This is a development prototype, not yet a public PyPI or CRAN release. The
working name must be checked against package registries before publication.

## Quick start

From the project root:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python -m montepilot.cli doctor
PYTHONPATH=src python -m montepilot.cli demo --backend auto
```

Install locally:

```bash
python -m pip install -e .
montepilot doctor
montepilot demo --target-mcse 0.01
```

Install PyTorch support separately:

```bash
python -m pip install -e ".[torch]"
```

For Intel Arc GPUs on Windows or Linux, install the official XPU wheel before
installing MontePilot:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Then confirm that `montepilot doctor` lists `torch_xpu`. With
`precision="auto"`, Intel client GPUs use FP32 because FP64 kernels are not
consistently available on those devices; NumPy, Torch CPU, and Torch CUDA use
FP64. Set the same explicit precision across backends for matched-precision
validation.

Compare every available backend on the same workload:

```bash
python benchmarks/benchmark_backends.py --reps 10000 --n 1000 --batch-size 1000
```

Run the repeated multi-workload suite with warm-ups, backend-order rotation,
scale-dependent crossover detection, and JSON/CSV/Markdown outputs:

```bash
python benchmarks/benchmark_suite.py --profile full --output-dir benchmark_results
```

For publication work, use the calibrated v0.4 harness. Fast backends are run
multiple times inside each timing block, accuracy is evaluated under distinct
seeds, and fresh-process latency is reported separately from warm-state timing:

```bash
python benchmarks/benchmark_publication.py --profile full --min-block-seconds 0.5 --output-dir benchmark_results_v04
```

The suite covers normal-mean simulation, batched OLS, causal IPW, and a
nonparametric bootstrap. A base-R reference is also included:

```bash
Rscript r/benchmarks/benchmark_base_r.R --reps 20000 --n 5000 --batch-size 2000 --scalar-reps 20000 --timing-repeats 5 --output r_benchmark_v04.csv
```

The v0.4 R benchmark changes the seed across timing repetitions and uses a
column-major layout that gives the batched and scalar implementations the same
replication-level random draws. Combine Python and R evidence with bootstrap
intervals and an SVG runtime figure:

```bash
python benchmarks/aggregate_results.py --python-report benchmark_results_v04/benchmark_report.json --r-input r_benchmark_v04.csv --output-dir publication_results
```

See `docs/BENCHMARK_PROTOCOL.md` for the estimands, timing boundaries, and
reporting rules. `docs/current_hardware_results/` contains the automatically
generated summary of the preliminary Intel Arc 140V experiment supplied during
development; its sub-40-millisecond XPU measurements motivated calibrated
timing blocks and should not be treated as the final v0.4 benchmark.

## Python API

The v0.5 high-level interface accepts a design and returns a complete report:

```python
import montepilot
from montepilot.examples import normal_mean_design

report = montepilot.run(
    design=normal_mean_design(sample_sizes=[50, 200, 1000]),
    backend="auto",
    precision="auto",
    batch_size=2000,
    min_reps=20000,
    max_reps=100000,
    target_mcse=0.001,
    target_coverage_mcse=0.0015,
    stopping_rule="all",
    checkpoint_dir="checkpoints",
    auto_expected_reps=25000,
)

print(report.to_json(indent=2))
```

The same example is included as `examples/run_v05.py` and can be launched
from Anaconda Prompt with `python examples\run_v05.py`.

The lower-level configuration API remains available:

```python
from montepilot import RunConfig, SimulationDesign, SimulationRunner
from montepilot.examples import normal_mean_design

design = normal_mean_design(
    sample_sizes=[50, 200],
    true_mean=0.5,
    true_sd=1.0,
)

config = RunConfig(
    backend="auto",
    precision="float32",
    batch_size=500,
    min_reps=1000,
    max_reps=20000,
    target_mcse=0.005,
    target_coverage_mcse=0.005,
    stopping_rule="all",
    seed=20261002,
)

report = SimulationRunner(config).run(design)
print(report.to_json(indent=2))
```

## Design contract

A generator receives a `BatchContext`, a condition dictionary, and a batch
size. An estimator receives the generated data, the same context, and the
condition. It returns either a one-dimensional array/tensor of estimates or a
`BatchEstimate` with estimates, standard errors, and convergence indicators.

This batch contract is what allows one implementation to run on NumPy or
PyTorch while avoiding slow Python-level replication loops.

## What "AI-ready" means in this version

MontePilot does not send data or code to an external model. It creates a
structured audit payload and accepts an optional advisor callback. This keeps
the numerical engine deterministic and testable while allowing a local or
hosted language model to be attached later. The default advisor is a transparent
rule-based diagnostic that never changes the simulation design.

See `docs/RESEARCH_ROADMAP.md` for the proposed methodological and AI-evaluation
research program.

## Project status

Version `0.6.0.dev0` is an MVP for design validation. Before public release it
still needs multi-platform CI, wider estimator coverage, device benchmarks on
independent GPUs, and a finalized package name.

## Controlled-input precision validation

The publication benchmark evaluates statistical behavior under independent,
backend-native random streams.  For direct numerical agreement, the controlled
precision suite generates each input once in NumPy float64 and sends the same
values to every available backend:

```bash
python benchmarks/validate_precision.py --profile full --precision float32 --output-dir precision_validation_results
```

The suite covers means, batched OLS, causal IPW, and bootstrap reductions. It
writes CSV, JSON, and Markdown evidence and exits with a nonzero status if any
replication-level comparison exceeds its prespecified mixed absolute/relative
tolerance.

The full five-seed suite passed on an Intel Arc 140V under PyTorch 2.14.1+xpu:
all 60 backend-workload-seed comparisons met tolerance, with no execution
failures. See `docs/CONTROLLED_PRECISION_VALIDATION_2026-10-03.md` and the
machine-readable files under
`docs/current_hardware_results/controlled_precision_v052/`.

## Substantive applications

Version 0.6 adds a psychometric congeneric-reliability simulation with an
analytic population Cronbach-alpha target and a two-level educational
cluster-randomized trial with school random intercepts, treatment-effect
standard errors, and coverage evaluation. See `docs/APPLICATIONS.md` and run:

```bash
python examples/run_psychometric.py
python examples/run_multilevel_education.py
```
