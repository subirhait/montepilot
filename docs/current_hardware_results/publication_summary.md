# MontePilot publication benchmark summary

Representative workload: `mean_n5000_r20000`.

| Implementation | Median seconds | Bootstrap CI | CV (%) | Observations/second |
|---|---:|---:|---:|---:|
| torch_xpu | 0.036460 | [0.031857, 0.038367] | 7.83 | 2,742,724,238 |
| numpy | 1.478412 | [1.452677, 1.654625] | 5.58 | 67,640,130 |
| torch_cpu | 1.955700 | [1.915432, 2.221139] | 6.20 | 51,132,584 |
| base_r_scalar_loop | 2.910000 | [2.900000, 2.950000] | 0.74 | 34,364,261 |
| base_r_batched | 3.200000 | [3.160000, 3.280000] | 1.49 | 31,250,000 |

## Selected speedups

| Reference | Candidate | Speedup | Bootstrap CI |
|---|---|---:|---:|
| numpy | torch_xpu | 40.55x | [38.53, 48.23] |
| numpy | torch_cpu | 0.76x | [0.67, 0.85] |
| base_r_batched | numpy | 2.16x | [1.93, 2.22] |
| base_r_batched | torch_cpu | 1.64x | [1.44, 1.68] |
| base_r_batched | torch_xpu | 87.77x | [83.40, 100.45] |
| base_r_scalar_loop | numpy | 1.97x | [1.76, 2.02] |
| base_r_scalar_loop | torch_cpu | 1.49x | [1.31, 1.52] |
| base_r_scalar_loop | torch_xpu | 79.81x | [75.85, 91.34] |

Intervals are percentile bootstrap intervals at the 95% level.
R timing repetitions use a fixed seed; Python repetitions use distinct seeds.
Intel XPU uses FP32 in this prototype, whereas the CPU implementations use FP64.
The report distinguishes workload-specific evidence from a general hardware claim.
