# Substantive simulation applications

MontePilot 0.6 adds two applications chosen to exercise statistical operations
that are common in educational measurement and multilevel research.

## Psychometric reliability

`congeneric_reliability_design()` simulates continuous standardized item scores
under a one-factor congeneric measurement model. Item loadings vary from 0.60
to 0.90 and residual variances standardize every item. The population
Cronbach-alpha value follows directly from the known covariance matrix, so bias
and RMSE can be evaluated without a numerical approximation to the truth.

```bash
python examples/run_psychometric.py
```

The example varies the number of respondents and items, selects a backend for
the planned workload, stops after the requested MCSE is reached, and writes
restartable checkpoints.

## Multilevel educational cluster trial

`cluster_randomized_trial_design()` simulates balanced school-randomized
studies. Outcomes contain a school random intercept and a student residual,
with their variances determined by a specified intraclass correlation. The
estimator is the treated-versus-control difference in school means. Its
standard error is computed from the between-school variation within the two
randomized groups, allowing both treatment-effect MCSE and interval-coverage
MCSE to be monitored.

```bash
python examples/run_multilevel_education.py
```

## Targeted application benchmark

The publication harness can run only the new applications, avoiding repetition
of the previously completed benchmark suite:

```bash
python benchmarks/benchmark_publication.py --profile full --precision float32 --workloads psychometric_alpha_n1000_k20 cluster_trial_j100_m30 --min-block-seconds 0.5 --output-dir application_benchmark_results_v06
```

This command compares NumPy, Torch CPU, and every available accelerator under
the same matched-float32 timing protocol used for the main publication study.
