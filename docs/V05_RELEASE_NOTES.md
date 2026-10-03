# MontePilot 0.5.1.dev0 release notes

Version 0.5 turns the prototype's adaptive behavior into an explicit public
API while retaining all v0.4 benchmark and command-line workflows.

Version 0.5.1 uses NumPy's native float32 random-number generation when FP32
is requested. This avoids generating FP64 arrays and converting them before
the timed simulation, making matched-FP32 backend comparisons more defensible.

## Added

- `montepilot.run()` as a concise high-level entry point.
- `precision="auto"`, `"float32"`, or `"float64"` across NumPy and Torch
  backends.
- A separate coverage-MCSE target and `all`/`any` stopping rules.
- Workload-aware backend selection using representative design conditions,
  warm-up measurements, and projected total runtime.
- Optional `auto_expected_reps` guidance for simulations whose likely stopping
  point is known from prior runs or domain calculations.
- Backend warm-up and projected-runtime fields in JSON reports.
- Checkpoint compatibility guards for precision and stopping rules.
- Command-line options for precision and coverage-MCSE stopping.

## Compatibility

Existing `RunConfig`, `SimulationRunner`, `target_mcse`, benchmark scripts, and
v0.4 command lines remain supported. `target_mcse` continues to mean the MCSE
of the simulation estimate.

## Interpretation

Automatic backend selection optimizes the configured workload rather than
assuming that a GPU is always faster. Short jobs may correctly select NumPy;
large vectorized jobs may select CUDA or Intel XPU. Matched precision should be
used when attributing speed differences to hardware or software rather than to
floating-point format.
