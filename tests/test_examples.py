import unittest

from montepilot import RunConfig, SimulationRunner
from montepilot.examples import (
    bootstrap_mean_design,
    cluster_randomized_trial_design,
    congeneric_reliability_design,
    ipw_ate_design,
    linear_regression_design,
)


def run_design(design, reps=1000):
    config = RunConfig(
        backend="numpy",
        batch_size=100,
        min_reps=reps,
        max_reps=reps,
        target_mcse=None,
        seed=1234,
    )
    return SimulationRunner(config).run(design).conditions[0]


class ExampleDesignTests(unittest.TestCase):
    def test_linear_regression_targets_coefficient(self):
        result = run_design(
            linear_regression_design(sample_sizes=[100], predictors=3),
            reps=1000,
        )
        self.assertEqual(result.failed_reps, 0)
        self.assertAlmostEqual(result.estimate_mean, result.truth, delta=0.05)

    def test_ipw_targets_ate(self):
        result = run_design(ipw_ate_design(sample_sizes=[500]), reps=1000)
        self.assertEqual(result.failed_reps, 0)
        self.assertAlmostEqual(result.estimate_mean, result.truth, delta=0.08)

    def test_bootstrap_targets_fixed_sample_mean(self):
        result = run_design(
            bootstrap_mean_design(data_size=500, resample_sizes=[500]),
            reps=1000,
        )
        self.assertEqual(result.failed_reps, 0)
        self.assertAlmostEqual(result.estimate_mean, result.truth, delta=0.03)

    def test_congeneric_reliability_targets_population_alpha(self):
        result = run_design(
            congeneric_reliability_design(sample_sizes=[400], item_counts=[10]),
            reps=1000,
        )
        self.assertEqual(result.failed_reps, 0)
        self.assertAlmostEqual(result.estimate_mean, result.truth, delta=0.02)

    def test_cluster_trial_targets_treatment_effect_and_coverage(self):
        result = run_design(
            cluster_randomized_trial_design(
                cluster_counts=[80],
                cluster_sizes=[25],
                treatment_effect=0.2,
                intraclass_correlation=0.15,
            ),
            reps=1500,
        )
        self.assertEqual(result.failed_reps, 0)
        self.assertAlmostEqual(result.estimate_mean, result.truth, delta=0.03)
        self.assertIsNotNone(result.coverage)
        self.assertAlmostEqual(result.coverage, 0.95, delta=0.04)


if __name__ == "__main__":
    unittest.main()
