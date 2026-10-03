# MontePilot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23127965.svg)](https://doi.org/10.5281/zenodo.23127965)
[![MontePilot tests](https://github.com/subirhait/montepilot/actions/workflows/tests.yml/badge.svg)](https://github.com/subirhait/montepilot/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MontePilot is an adaptive, device-aware framework for reproducible Monte Carlo
simulation in statistical research. It executes simulations in vectorized
batches, selects among available CPU and GPU backends, monitors Monte Carlo
uncertainty, and can stop automatically once prespecified precision targets are
met.

Version `0.6.0.dev0` is a public research prototype. It is archived on Zenodo
but is not yet distributed through PyPI or CRAN.

## Features

- deterministic, independent random streams for each design condition and batch;
- NumPy CPU execution with optional PyTorch CPU, CUDA, and Intel XPU backends;
- workload-aware backend selection that accounts for accelerator warm-up cost;
- vectorized batches instead of one-replication-at-a-time Python loops;
- adaptive stopping based on estimate MCSE, coverage MCSE, or both;
- explicit `auto`, `float32`, and `float64` precision policies;
- bias, empirical standard deviation, RMSE, coverage, failure-rate, and runtime summaries;
- checkpointing and recovery for interrupted simulations;
- machine-readable JSON audit records and deterministic diagnostic advice;
- controlled-input numerical validation across available backends;
- substantive psychometric and multilevel educational simulation examples; and
- a command-line interface, Python API, and initial R interface.

## Installation

Clone or download the repository, open a terminal in the project root, and
install the package in editable mode:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
montepilot doctor
montepilot demo --target-mcse 0.01
```

Run the complete automated test suite:

```bash
python -m unittest discover -s tests -v
```

The same test suite runs automatically on Linux, Windows, and macOS through
GitHub Actions.

## Optional PyTorch acceleration

Install PyTorch support separately:

```bash
python -m pip install -e ".[torch]"
```

For Intel Arc GPUs on Windows or Linux, install the official PyTorch XPU wheel
before installing MontePilot:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Then run:

```bash
montepilot doctor
```

The diagnostic report should list `torch_xpu` when a compatible Intel GPU and
PyTorch XPU installation are available.

With `precision="auto"`, this prototype uses FP32 on Intel client GPUs because
FP64 kernels are not consistently available on those devices. NumPy, Torch CPU,
and Torch CUDA use FP64 by default. For scientifically interpretable backend
comparisons, request the same explicit precision across all backends and examine
runtime and numerical error jointly.

## Python API

The high-level interface accepts a simulation design and returns a complete
report:

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

The automatic scheduler probes eligible backends, estimates warm-up and
batch-processing costs, and selects the backend with the lowest projected total
runtime for the planned workload. A backend can also be selected explicitly,
for example with `backend="numpy"`, `backend="torch_cpu"`, or
`backend="torch_xpu"`.

The corresponding runnable example is available in
[`examples/run_v05.py`](examples/run_v05.py).

### Lower-level configuration API

```python
from montepilot import RunConfig, SimulationRunner
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
condition. It returns either a one-dimensional array or tensor of estimates or
a `BatchEstimate` containing estimates, standard errors, and convergence
indicators.

This batch contract allows one simulation design to run on NumPy or PyTorch
while avoiding slow Python-level replication loops.

## Benchmarks

Compare every available backend on the same normal-mean workload:

```bash
python benchmarks/benchmark_backends.py --reps 10000 --n 1000 --batch-size 1000
```

Run the repeated multi-workload benchmark suite:

```bash
python benchmarks/benchmark_suite.py --profile full --output-dir benchmark_results
```

Run the calibrated publication benchmark with matched FP32 precision:

```bash
python benchmarks/benchmark_publication.py --profile full --precision float32 --min-block-seconds 0.5 --output-dir publication_benchmark_results
```

The publication benchmark separates warm-state throughput from fresh-process
latency, varies seeds across timing repetitions, and reports bootstrap
confidence intervals for runtime and speedup. It covers normal-mean simulation,
batched OLS, causal inverse-probability weighting, and a nonparametric
bootstrap.

A base-R comparison is also included:

```bash
Rscript r/benchmarks/benchmark_base_r.R --reps 20000 --n 5000 --batch-size 2000 --scalar-reps 20000 --timing-repeats 5 --output r_benchmark_results.csv
```

See [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) for timing
boundaries, estimands, calibration rules, and reporting requirements. Benchmark
results are hardware-, software-, workload-, and precision-specific; they
should not be generalized to unsupported devices or workloads.

## Controlled-input precision validation

The publication benchmark evaluates statistical behavior under independent,
backend-native random streams. The controlled-input suite instead generates
each input once in NumPy float64 and supplies the same values to every eligible
backend, isolating arithmetic and reduction differences:

```bash
python benchmarks/validate_precision.py --profile full --precision float32 --output-dir precision_validation_results
```

The suite covers mean estimation, batched OLS, causal inverse-probability
weighting, and bootstrap reductions. It writes CSV, JSON, and Markdown evidence
and exits with a nonzero status if any replication-level comparison exceeds its
prespecified mixed absolute/relative tolerance.

In the five-seed Intel Arc 140V validation, all 60
backend-workload-seed comparisons met tolerance without execution failures.
See
[`docs/CONTROLLED_PRECISION_VALIDATION_2026-10-03.md`](docs/CONTROLLED_PRECISION_VALIDATION_2026-10-03.md)
and the machine-readable evidence in
[`docs/current_hardware_results/controlled_precision_v052/`](docs/current_hardware_results/controlled_precision_v052/).

## Substantive applications

Version 0.6 includes two substantive simulation designs:

- a psychometric congeneric-reliability simulation with an analytic population
  Cronbach's alpha target; and
- a two-level cluster-randomized educational trial with school random
  intercepts, treatment-effect standard errors, and coverage evaluation.

Run the examples with:

```bash
python examples/run_psychometric.py
python examples/run_multilevel_education.py
```

Application documentation is available in
[`docs/APPLICATIONS.md`](docs/APPLICATIONS.md).

## What "AI-ready" means in this version

MontePilot does not send data or code to an external model. The numerical engine
remains deterministic and testable. It produces a structured audit payload and
accepts an optional advisor callback, allowing a local or hosted language model
to be connected later without giving that model control over the simulation
design. The default advisor is a transparent, rule-based diagnostic.

See [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) for the proposed
methodological and AI-evaluation research program.

## Project status

Version `0.6.0.dev0` is a public development pre-release for design validation.
Automated continuous integration tests the base installation on Linux, Windows,
and macOS. Before a stable release, MontePilot still needs wider estimator
coverage, benchmarks on independent accelerator hardware, expanded integration
testing for optional PyTorch backends, and final package-name clearance.

## Citation

If you use MontePilot, cite the archived software release:

> Hait, S. (2026). *MontePilot: Adaptive Device-Aware Monte Carlo Simulation*
> (Version 0.6.0.dev0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.23127965

Citation metadata are also provided in [`CITATION.cff`](CITATION.cff).

## License

MontePilot is released under the [MIT License](LICENSE).
