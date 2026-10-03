"""Psychometric reliability application for MontePilot 0.6."""

import montepilot
from montepilot.examples import congeneric_reliability_design


report = montepilot.run(
    design=congeneric_reliability_design(
        sample_sizes=[250, 1000],
        item_counts=[10, 20],
    ),
    backend="auto",
    precision="auto",
    batch_size=250,
    min_reps=5000,
    max_reps=50000,
    target_mcse=0.001,
    checkpoint_dir="checkpoints_psychometric",
    auto_expected_reps=10000,
)

print(report.to_json(indent=2))
