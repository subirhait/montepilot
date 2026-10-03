# MontePilot controlled-input precision validation

Every candidate received the same NumPy-generated data. The reference was evaluated in NumPy float64; candidates used the precision shown below.

| Workload | Backend | Precision | Max abs. difference | RMSE difference | Within tolerance | Result |
|---|---|---:|---:|---:|---:|---:|
| mean_n5000 | numpy | float32 | 9.376e-08 | 2.623e-08 | 100.000% | PASS |
| mean_n5000 | numpy | float32 | 8.445e-08 | 2.600e-08 | 100.000% | PASS |
| mean_n5000 | numpy | float32 | 8.289e-08 | 2.570e-08 | 100.000% | PASS |
| mean_n5000 | numpy | float32 | 8.821e-08 | 2.676e-08 | 100.000% | PASS |
| mean_n5000 | numpy | float32 | 7.941e-08 | 2.639e-08 | 100.000% | PASS |
| mean_n5000 | torch_cpu | float32 | 9.868e-08 | 2.935e-08 | 100.000% | PASS |
| mean_n5000 | torch_cpu | float32 | 1.052e-07 | 3.012e-08 | 100.000% | PASS |
| mean_n5000 | torch_cpu | float32 | 1.025e-07 | 2.981e-08 | 100.000% | PASS |
| mean_n5000 | torch_cpu | float32 | 1.045e-07 | 3.074e-08 | 100.000% | PASS |
| mean_n5000 | torch_cpu | float32 | 1.111e-07 | 3.008e-08 | 100.000% | PASS |
| mean_n5000 | torch_xpu | float32 | 9.712e-08 | 2.883e-08 | 100.000% | PASS |
| mean_n5000 | torch_xpu | float32 | 8.941e-08 | 2.999e-08 | 100.000% | PASS |
| mean_n5000 | torch_xpu | float32 | 8.904e-08 | 2.973e-08 | 100.000% | PASS |
| mean_n5000 | torch_xpu | float32 | 9.736e-08 | 3.022e-08 | 100.000% | PASS |
| mean_n5000 | torch_xpu | float32 | 1.061e-07 | 2.946e-08 | 100.000% | PASS |
| ols_n500_p8 | numpy | float32 | 2.649e-07 | 6.241e-08 | 100.000% | PASS |
| ols_n500_p8 | numpy | float32 | 2.174e-07 | 6.148e-08 | 100.000% | PASS |
| ols_n500_p8 | numpy | float32 | 2.370e-07 | 6.184e-08 | 100.000% | PASS |
| ols_n500_p8 | numpy | float32 | 2.026e-07 | 5.812e-08 | 100.000% | PASS |
| ols_n500_p8 | numpy | float32 | 1.926e-07 | 6.133e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_cpu | float32 | 1.305e-07 | 3.507e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_cpu | float32 | 1.218e-07 | 3.392e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_cpu | float32 | 1.085e-07 | 3.611e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_cpu | float32 | 1.158e-07 | 3.304e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_cpu | float32 | 1.026e-07 | 3.503e-08 | 100.000% | PASS |
| ols_n500_p8 | torch_xpu | float32 | 4.860e-07 | 1.304e-07 | 100.000% | PASS |
| ols_n500_p8 | torch_xpu | float32 | 5.564e-07 | 1.305e-07 | 100.000% | PASS |
| ols_n500_p8 | torch_xpu | float32 | 4.079e-07 | 1.298e-07 | 100.000% | PASS |
| ols_n500_p8 | torch_xpu | float32 | 4.281e-07 | 1.355e-07 | 100.000% | PASS |
| ols_n500_p8 | torch_xpu | float32 | 5.730e-07 | 1.383e-07 | 100.000% | PASS |
| ipw_n1000 | numpy | float32 | 2.367e-07 | 6.791e-08 | 100.000% | PASS |
| ipw_n1000 | numpy | float32 | 2.266e-07 | 7.190e-08 | 100.000% | PASS |
| ipw_n1000 | numpy | float32 | 2.251e-07 | 6.692e-08 | 100.000% | PASS |
| ipw_n1000 | numpy | float32 | 2.225e-07 | 7.060e-08 | 100.000% | PASS |
| ipw_n1000 | numpy | float32 | 2.589e-07 | 6.990e-08 | 100.000% | PASS |
| ipw_n1000 | torch_cpu | float32 | 2.987e-07 | 7.832e-08 | 100.000% | PASS |
| ipw_n1000 | torch_cpu | float32 | 2.880e-07 | 7.989e-08 | 100.000% | PASS |
| ipw_n1000 | torch_cpu | float32 | 2.752e-07 | 7.776e-08 | 100.000% | PASS |
| ipw_n1000 | torch_cpu | float32 | 2.834e-07 | 8.281e-08 | 100.000% | PASS |
| ipw_n1000 | torch_cpu | float32 | 2.547e-07 | 7.775e-08 | 100.000% | PASS |
| ipw_n1000 | torch_xpu | float32 | 2.400e-07 | 6.973e-08 | 100.000% | PASS |
| ipw_n1000 | torch_xpu | float32 | 2.692e-07 | 7.193e-08 | 100.000% | PASS |
| ipw_n1000 | torch_xpu | float32 | 2.265e-07 | 7.178e-08 | 100.000% | PASS |
| ipw_n1000 | torch_xpu | float32 | 2.254e-07 | 7.105e-08 | 100.000% | PASS |
| ipw_n1000 | torch_xpu | float32 | 2.407e-07 | 7.127e-08 | 100.000% | PASS |
| bootstrap_n2000 | numpy | float32 | 7.302e-08 | 1.969e-08 | 100.000% | PASS |
| bootstrap_n2000 | numpy | float32 | 7.009e-08 | 1.861e-08 | 100.000% | PASS |
| bootstrap_n2000 | numpy | float32 | 1.028e-07 | 3.008e-08 | 100.000% | PASS |
| bootstrap_n2000 | numpy | float32 | 8.109e-08 | 1.971e-08 | 100.000% | PASS |
| bootstrap_n2000 | numpy | float32 | 8.300e-08 | 2.150e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_cpu | float32 | 9.917e-08 | 2.489e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_cpu | float32 | 8.141e-08 | 2.506e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_cpu | float32 | 1.066e-07 | 3.163e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_cpu | float32 | 1.050e-07 | 2.525e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_cpu | float32 | 7.887e-08 | 2.470e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_xpu | float32 | 1.068e-07 | 3.191e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_xpu | float32 | 1.109e-07 | 2.943e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_xpu | float32 | 1.126e-07 | 3.896e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_xpu | float32 | 9.275e-08 | 2.867e-08 | 100.000% | PASS |
| bootstrap_n2000 | torch_xpu | float32 | 1.072e-07 | 3.098e-08 | 100.000% | PASS |

## Interpretation

All evaluated replication-level estimates met the prespecified mixed absolute/relative tolerance against the NumPy float64 reference.

This controlled-input analysis isolates arithmetic and reduction differences. It complements, rather than replaces, the multi-seed distributional validation in the publication benchmark.
