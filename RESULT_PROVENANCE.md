# Result provenance for the IJDSA manuscript

The manuscript reports results generated during four successive development
releases. The original source archive for each hardware experiment and the
machine-readable output used in the article are retained here. This mapping is
intentional: the earlier hardware runs were not relabeled as later-version results.

The frozen pre-submission extension is defined by `PROTOCOL.md` and
`benchmarks/protocol_v062.json`. Before any full timed run, replace the marker
below with the commit resolved by tag `protocol-v0.6.2`:

```text
protocol-v0.6.2 commit: 8fd8a3eb61e24fc66f73d4e704e5b699f72ac5cd
```

| Manuscript evidence | Generating release | Source archive | Machine-readable output |
|---|---|---|---|
| Generic matched-FP32 warm and fresh-process benchmarks (Q1, Q2, and generic Q4 rows) | 0.5.1.dev0 | `archived_versions/MontePilot_MVP_v0.5.1_NativeFP32.zip` | `results/publication_fp32_v051/` |
| Controlled-input validation (Q3) | 0.5.2.dev0 | `archived_versions/MontePilot_MVP_v0.5.2_Validated.zip` | `docs/current_hardware_results/controlled_precision_v052/` |
| Reliability and cluster-trial hardware benchmarks and multi-seed validation (application rows of Q1 and Q4) | 0.6.0.dev0 | `archived_versions/MontePilot_MVP_v0.6.0_ApplicationSuite.zip` | `results/application_v060/` |
| Variance stress test, adaptive stopping, and checkpoint recovery (Q5 and Q6) | 0.6.1 | current source tree | `results/supplementary_experiments.json` |
| Prespecified logistic-IRLS laptop study | 0.6.2 | tag `protocol-v0.6.2` | `protocol_v062_laptop_full/` (pending run) |
| Prespecified A100 CUDA FP32/FP64 study | 0.6.2 | tag `protocol-v0.6.2` | `protocol_v062_a100_full/` (pending run) |

Version 0.6.1 adds centered variance calculations and stream-specific seed
derivation for the built-in application designs. The older benchmark harnesses
used the one-pass variance option described in the paper, so their timed
configuration is preserved in their archived source snapshots. New timing runs
on different hardware are expected to yield different runtimes.

Version 0.6.2 adds only prespecified benchmark and validation infrastructure;
logistic IRLS is not exported as a public built-in design. Pending rows must
remain labeled as pending until their raw output, environment record, and
verified checksum manifest have been archived.

## SHA-256 checksums of archived source bundles

```text
772d803b9bc81dafbedac5b2e8f5440c89286e3194c8489bf3c89653812c57b7  MontePilot_MVP_v0.5.1_NativeFP32.zip
9d1f7ce7f014597a0bc3d2bfa0f4513e4aeb242663735e17093229aef2914914  MontePilot_MVP_v0.5.2_Validated.zip
eb10102ed506f349d585d427a337cf37e76c1e8b07cd2e937b278b36961faa3a  MontePilot_MVP_v0.6.0_ApplicationSuite.zip
```
