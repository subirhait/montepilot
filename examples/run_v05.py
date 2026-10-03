"""Complete MontePilot v0.5 example for Anaconda Prompt users."""

from pathlib import Path

import montepilot
from montepilot.examples import normal_mean_design


design = normal_mean_design(
    sample_sizes=[50, 200, 1000],
    true_mean=0.5,
    true_sd=1.0,
)

report = montepilot.run(
    design=design,
    backend="auto",
    precision="auto",
    batch_size=2000,
    min_reps=20000,
    max_reps=100000,
    target_mcse=0.001,
    target_coverage_mcse=0.0015,
    stopping_rule="all",
    checkpoint_dir="checkpoints_v05",
    auto_expected_reps=25000,
)

rendered = report.to_json(indent=2)
Path("montepilot_v05_report.json").write_text(rendered + "\n", encoding="utf-8")
print(rendered)
