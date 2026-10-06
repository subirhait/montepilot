# MontePilot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23127964.svg)](https://doi.org/10.5281/zenodo.23127964)

MontePilot is an adaptive, device-aware framework for reproducible Monte Carlo
simulation in statistical research. It executes simulations in vectorized
batches, selects among available CPU and GPU backends, monitors Monte Carlo
uncertainty, and can stop automatically when prespecified precision targets are
met.

Version `0.6.2` is a public research prototype. The DOI above is the Zenodo
concept DOI for all releases. For exact replication, use this source snapshot
and the machine-readable result files under `results/`.

## Features

- deterministic, position- and stream-specific random seeds for every design condition;
- NumPy CPU execution with optional PyTorch CPU, CUDA, and Intel XPU backends;
- workload-aware backend probing with explicit warm-up cost projection;
- vectorized batches instead of one-replication-at-a-time Python loops;
- stopping based on estimate MCSE, coverage MCSE, or both;
- explicit `auto`, `float32`, and `float64` precision policies;
- bias, empirical SD, RMSE, coverage, failure-rate, and runtime summaries;
- checkpointing and exact recovery for interrupted runs;
- machine-readable JSON audit records;
- controlled-input numerical validation across eligible backends;
- psychometric reliability and multilevel educational-trial examples; and
- a command-line interface, Python API, and initial R interface.

## Installation

From the project root:

```bash
python -m pip install -e .
montepilot doctor
python -m unittest discover -s tests -v
```

Install optional research and PyTorch dependencies with:

```bash
python -m pip install -e ".[research,torch]"
```

For Intel Arc GPUs, follow the official PyTorch XPU installation instructions
and then confirm that `montepilot doctor` lists `torch_xpu`.

## Python API

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

The automatic scheduler probes eligible backends and projects total runtime for
the requested workload. A backend can also be selected explicitly with
`backend="numpy"`, `backend="torch_cpu"`, `backend="torch_cuda"`, or
`backend="torch_xpu"`.

## Validation and benchmarks

Run the controlled-input precision suite:

```bash
python benchmarks/validate_precision.py \
  --profile full \
  --precision float32 \
  --output-dir precision_validation_results
```

Run the calibrated publication benchmark:

```bash
python benchmarks/benchmark_publication.py \
  --profile full \
  --precision float32 \
  --min-block-seconds 0.5 \
  --output-dir publication_benchmark_results
```

Run the hardware-independent supplementary experiments:

```bash
python benchmarks/supplementary_experiments.py
```

The prespecified v0.6.2 logistic-IRLS and CUDA extension is frozen in
`PROTOCOL.md` and `benchmarks/protocol_v062.json`. Before a full timed run,
commit and tag the protocol as described there. Smoke tests are launched with:

```bash
python benchmarks/run_frozen_protocol.py --phase laptop --smoke
python benchmarks/run_frozen_protocol.py --phase a100 --smoke
```

The archived article evidence is organized as follows:

- `results/publication_fp32_v051/`: matched-FP32 generic benchmarks;
- `results/application_v060/`: psychometric and cluster-trial benchmarks;
- `results/supplementary_experiments.json`: variance, stopping, checkpoint,
  and analytic checks; and
- `docs/current_hardware_results/controlled_precision_v052/`: five-seed
  controlled-input validation.

See `RESULT_PROVENANCE.md` for the exact mapping from manuscript results to
the 0.5.1, 0.5.2, 0.6.0, 0.6.1, and prespecified 0.6.2 releases. The original
source bundles for the three earlier hardware runs are preserved under
`archived_versions/`.

See `REPRODUCIBILITY.md` and `docs/BENCHMARK_PROTOCOL.md` before interpreting
timings. Performance is specific to the hardware, software versions, workload,
precision, and timing boundary.

## Numerical precision

With `precision="auto"`, this prototype uses FP32 on Intel client GPUs because
native FP64 support is limited on those devices. CPU and CUDA backends use FP64
by default. Request the same explicit precision for scientifically interpretable
backend comparisons and examine runtime and numerical error jointly.

Version 0.6.1 and later use centered two-pass variance calculations by default
in the application examples. The one-pass form is retained only to reproduce
the earlier v0.6.0 timing configuration.

## Substantive examples

```bash
python examples/run_psychometric.py
python examples/run_multilevel_education.py
```

These implement a congeneric-reliability study with analytic population alpha
and a two-level cluster-randomized educational trial.

## AI integration boundary

MontePilot does not transmit data or code to an external AI service. Its
numerical engine is deterministic and testable. It emits a structured audit
payload and accepts an optional advisor callback, allowing a separately
configured local or hosted model to be connected later. The built-in advisor is
a transparent rule-based diagnostic and cannot alter the simulation design.

## Development status

Version `0.6.2` is a research pre-release. Forty-five automated tests cover the
base NumPy installation. Before a stable release, the project still needs wider
estimator coverage, installed-backend integration tests, and independent
accelerator benchmarks.

## Citation

Please cite the archived project:

> Hait, S. (2026). *MontePilot: Adaptive Device-Aware Monte Carlo Simulation*
> [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23127964

The DOI is the all-version concept DOI. When citing an exact release, use its
version-specific Zenodo DOI if available. Citation metadata are also provided in
`CITATION.cff`.

## License

MontePilot is released under the MIT License.
