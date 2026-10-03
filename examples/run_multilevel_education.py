"""Multilevel educational cluster-trial application for MontePilot 0.6."""

import montepilot
from montepilot.examples import cluster_randomized_trial_design


report = montepilot.run(
    design=cluster_randomized_trial_design(
        cluster_counts=[40, 100],
        cluster_sizes=[20, 30],
        treatment_effect=0.2,
        intraclass_correlation=0.15,
    ),
    backend="auto",
    precision="auto",
    batch_size=500,
    min_reps=5000,
    max_reps=50000,
    target_mcse=0.002,
    target_coverage_mcse=0.003,
    stopping_rule="all",
    checkpoint_dir="checkpoints_multilevel",
    auto_expected_reps=10000,
)

print(report.to_json(indent=2))
