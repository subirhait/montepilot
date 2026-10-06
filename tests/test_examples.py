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

    def test_centered_and_one_pass_variance_agree_in_float64(self):
        for factory in (
            lambda m: congeneric_reliability_design(
                sample_sizes=[200], item_counts=[8], variance_method=m
            ),
            lambda m: cluster_randomized_trial_design(
                cluster_counts=[20], cluster_sizes=[10], variance_method=m
            ),
        ):
            centered = run_design(factory("centered"), reps=300)
            one_pass = run_design(factory("one_pass"), reps=300)
            self.assertAlmostEqual(centered.estimate_mean, one_pass.estimate_mean, places=10)
            self.assertAlmostEqual(centered.rmse, one_pass.rmse, places=10)

    def test_centered_variance_is_stable_in_float32_with_large_location(self):
        import numpy as np
        from montepilot.backends import resolve_backend
        from montepilot.examples import _batched_variance

        rng = np.random.default_rng(7)
        x64 = rng.standard_normal((50, 1000)) + 100.0
        reference = x64.var(axis=1, ddof=1)
        backend = resolve_backend("numpy", precision="float32")
        x32 = backend.asarray(x64)
        centered = np.asarray(_batched_variance(backend, x32, 1, 1000, "centered"))
        one_pass = np.asarray(_batched_variance(backend, x32, 1, 1000, "one_pass"))
        centered_error = np.max(np.abs(centered - reference))
        one_pass_error = np.max(np.abs(one_pass - reference))
        self.assertLess(centered_error, 1e-4)
        self.assertGreater(one_pass_error, 10 * centered_error)

    def test_invalid_variance_method_rejected(self):
        with self.assertRaises(ValueError):
            congeneric_reliability_design(variance_method="naive")


if __name__ == "__main__":
    unittest.main()
