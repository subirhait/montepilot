# Intel Arc 140V v0.4 diagnostic run

Date: 2026-10-02  
Device: Intel(R) Arc(TM) 140V GPU (16GB)  
Python: 3.11.17  
PyTorch: 2.14.1+xpu  
MontePilot: 0.4.0.dev0

## Successful checks

- The full suite produced 135 warm timing rows across nine workloads and three
  backends.
- Every simulation replication was valid; no failures or warnings were
  reported.
- Warm-state XPU speedups relative to NumPy ranged from 10.46x for batched OLS
  to 48.09x for 20,000 replications of 5,000 observations.
- The largest mean workload had median warm runtimes of 1.515 seconds for
  NumPy, 2.001 seconds for Torch CPU, and 0.0315 seconds for Torch XPU.
- Fresh-process median latency for the same workload was 3.926 seconds for
  NumPy and 2.484 seconds for XPU, a 1.58x end-to-end advantage.
- Across five validation seeds for the largest mean workload, mean RMSE was
  0.014153 for NumPy, 0.014158 for Torch CPU, and 0.014102 for XPU. These are
  close to the theoretical value of 0.014142.

## Calibration finding

Fifty of the 135 recorded blocks were shorter than the requested 0.5 seconds.
The single probe sometimes included initialization cost and therefore selected
too few steady-state inner iterations. This does not explain away the observed
XPU advantage, but it means the run is diagnostic rather than the final
publication benchmark.

Version 0.4.1.dev0 corrects this behavior by discarding undersized blocks,
increasing the iteration count, and rerunning them. The final hardware result
must come from the corrected harness.
