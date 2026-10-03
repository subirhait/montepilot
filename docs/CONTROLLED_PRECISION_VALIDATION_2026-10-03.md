# Controlled-input precision validation on Intel Arc 140V

## Purpose

The publication benchmark uses each backend's native random-number generator.
That design evaluates statistical behavior and speed, but the simulated samples
are not identical across backends. This validation isolates numerical arithmetic
by generating each input once with NumPy and supplying the same values to every
candidate backend. NumPy float64 is the reference.

## Environment

- Windows 10 build 26300
- Python 3.11.17
- NumPy 2.4.6
- PyTorch 2.14.1+xpu
- Intel(R) Arc(TM) 140V GPU (16GB)
- MontePilot 0.5.2.dev0
- Candidate precision: float32

## Design

The full profile evaluated four workloads on NumPy, Torch CPU, and Torch XPU
using five prespecified seeds. This produced 60 backend-workload-seed
comparisons. Every comparison used a mixed absolute/relative tolerance defined
before the hardware run.

## Results

All 60 comparisons passed, no workload failed to execute, and 100% of the
replication-level estimates were within tolerance.

| Backend | Comparisons | Largest absolute difference | Largest scaled error |
|---|---:|---:|---:|
| NumPy float32 | 20 | 2.649e-7 | 0.013364 |
| Torch CPU float32 | 20 | 2.987e-7 | 0.014777 |
| Torch XPU float32 | 20 | 5.730e-7 | 0.014820 |

The largest Intel XPU absolute difference occurred for batched OLS. The largest
scaled error occurred for the bootstrap workload. A scaled error of 1.0 is the
prespecified pass boundary; the observed maximum of 0.014820 used less than
1.5% of the allowed tolerance.

| Workload | Largest absolute difference | Largest scaled error |
|---|---:|---:|
| Mean, n = 5,000 | 1.111e-7 | 0.014777 |
| OLS, n = 500, p = 8 | 5.730e-7 | 0.002172 |
| IPW, n = 1,000 | 2.987e-7 | 0.000708 |
| Bootstrap, n = 2,000 | 1.126e-7 | 0.014820 |

## Interpretation

For these four workloads and this hardware/software environment, float32
execution on Intel XPU produced replication-level estimates that closely agreed
with NumPy float64. These results support numerical agreement for the tested
operations; they do not establish equivalence for every statistical estimator,
ill-conditioned problem, device, library version, or precision regime.

The controlled-input result complements the publication benchmark's multi-seed
bias, RMSE, failure, and timing evidence. The two analyses should be reported
together.

## Archived files

The complete machine-readable evidence is stored in
`docs/current_hardware_results/controlled_precision_v052/`.
