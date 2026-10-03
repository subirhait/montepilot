# MontePilot calibrated benchmark summary

## Warm-state calibrated timing

| Workload | Backend | Inner iterations | Median seconds | 95% bootstrap CI | CV (%) |
|---|---:|---:|---:|---:|---:|
| cluster_trial_j100_m30 | numpy | 3 | 0.209013 | [0.197723, 0.217102] | 3.55 |
| cluster_trial_j100_m30 | torch_cpu | 7 | 0.077089 | [0.075760, 0.079191] | 2.00 |
| cluster_trial_j100_m30 | torch_xpu | 44 | 0.011948 | [0.011816, 0.012049] | 0.82 |
| psychometric_alpha_n1000_k20 | numpy | 1 | 0.640445 | [0.629442, 0.651768] | 1.24 |
| psychometric_alpha_n1000_k20 | torch_cpu | 3 | 0.195648 | [0.188345, 0.205780] | 3.44 |
| psychometric_alpha_n1000_k20 | torch_xpu | 14 | 0.037536 | [0.037014, 0.051503] | 15.59 |

## Speedup relative to NumPy

| Workload | Candidate | Speedup | 95% bootstrap CI |
|---|---:|---:|---:|
| cluster_trial_j100_m30 | torch_cpu | 2.711x | [2.565, 2.816] |
| cluster_trial_j100_m30 | torch_xpu | 17.494x | [16.549, 18.171] |
| psychometric_alpha_n1000_k20 | torch_cpu | 3.273x | [3.112, 3.405] |
| psychometric_alpha_n1000_k20 | torch_xpu | 17.062x | [12.435, 17.364] |

## Fresh-process latency

This includes Python startup, imports, backend initialization, computation, and process exit.

| Workload | Backend | Median total seconds | Median engine seconds | Median startup/import seconds |
|---|---:|---:|---:|---:|

Timing repetitions use distinct seeds. Accuracy summaries are reported separately.
All backends requested float32.
Fresh-process latency is not a hardware-cold measurement because operating-system and driver caches may persist.
