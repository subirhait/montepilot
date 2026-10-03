# Intel Arc 140V verification benchmark

> Historical v0.2 record. The expanded v0.3 Python and base-R results are in
> `docs/current_hardware_results/`. Version 0.4 introduces calibrated timing;
> therefore neither record should be presented as the final v0.4 benchmark.

Date: 2026-10-02  
Operating system: Windows 11 (`Windows-10-10.0.26300-SP0` platform string)  
Python: 3.11.17  
PyTorch: 2.14.1+xpu  
Device: Intel(R) Arc(TM) 140V GPU (16GB)  
MontePilot tested build: 0.2.0.dev0

## Workload

- 20,000 Monte Carlo replications
- 5,000 normal observations per replication
- 2,000 replications per batch
- 100,000,000 total simulated observations per backend
- Five repeated executions

## Timing summary

| Backend | Median seconds | Mean seconds | SD seconds | CV | Ratio of median to XPU |
|---|---:|---:|---:|---:|---:|
| NumPy CPU | 1.446519 | 1.446506 | 0.008282 | 0.57% | 5.77x |
| Torch CPU | 1.974350 | 1.969328 | 0.046831 | 2.38% | 7.88x |
| Torch XPU | 0.250570 | 0.251209 | 0.013113 | 5.22% | 1.00x |

XPU was the fastest backend in all five executions. Even its slowest execution
(0.271473 seconds) was more than five times faster than the fastest NumPy
execution (1.435968 seconds).

## Numerical checks

Each backend completed all 20,000 replications with zero reported failures.
Within each backend, estimates were identical across the five executions under
the fixed seed. Across backends, absolute bias was below 0.00009 and RMSE ranged
from 0.014178 to 0.014242. Cross-backend estimates are not expected to be
identical because the implementations use different random-number streams and
XPU uses FP32 while CPU backends use FP64.

## Interpretation

The result establishes functional Intel XPU execution and a stable speed
advantage for this large vectorized workload on this machine. It is not a
general GPU-performance claim. Publication evidence requires warm-ups,
backend-order control, scale sweeps, multiple workload families, and additional
hardware. MontePilot 0.3.0.dev0 adds that benchmark infrastructure.
