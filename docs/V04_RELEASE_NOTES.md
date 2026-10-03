# MontePilot 0.4.2.dev0 release notes

This development release turns the original benchmark demonstration into an
auditable performance protocol.

Version 0.4.1 corrects an issue discovered during the first Intel XPU run: a
single initialization-sensitive probe could overestimate steady-state runtime,
causing later blocks to undershoot the requested minimum. The harness now
discards undersized blocks, increases the iteration count, and reruns them.

Version 0.4.2 renames the R output column `repeat` to `timing_repeat` because
`repeat` is an R reserved keyword. The Python benchmark engine and validated
v0.4.1 hardware results are unchanged.

## Added

- Calibrated timing blocks with a configurable minimum duration.
- Deterministic bootstrap intervals for median runtime and speedup ratios.
- Distinct-seed accuracy validation separated from timing summaries.
- Fresh-process latency measured in isolated Python processes.
- Base-R result import from CSV files, directories, or ZIP archives.
- Combined Python/R CSV, JSON, Markdown, and SVG publication artifacts.
- Multi-seed R timing and paired replication-level random draws.
- Unit tests for calibration, bootstrap inference, validation summaries, and
  R archive import.

## Timing contract

Simulation generation, estimation, accelerator-to-host transfer, output
validation, and online statistical summaries are included. Accelerator work is
explicitly synchronized before condition runtime is finalized.

## Compatibility

The public Python simulation API is unchanged. Existing v0.3 commands continue
to work. The R benchmark retains all previous arguments and adds optional
`--timing-repeats` and `--seed-step` arguments.

## Remaining validation

The calibrated v0.4.1 full suite must be executed on the Intel Arc 140V system.
The current bundled hardware summary was generated from the earlier v0.3 runs
and is retained as preliminary evidence only.
