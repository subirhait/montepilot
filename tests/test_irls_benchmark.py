import json
import unittest
from pathlib import Path

import numpy as np

from montepilot.backends import ComputeBackend
from montepilot.benchmark_irls import (
    IRLSSettings,
    fit_logistic_irls,
    generate_controlled_logistic_data,
)


class LogisticIRLSBenchmarkTests(unittest.TestCase):
    def test_regular_condition_recovers_slope(self):
        x, y = generate_controlled_logistic_data(
            seed=19,
            replications=128,
            sample_size=300,
            intercept=-0.5,
            slope=1.0,
        )
        result = fit_logistic_irls(
            x,
            y,
            ComputeBackend("numpy", precision="float64"),
            IRLSSettings(),
        )
        self.assertGreaterEqual(int(result.converged.sum()), 120)
        self.assertAlmostEqual(float(result.estimates[result.converged].mean()), 1.0, delta=0.12)
        self.assertTrue(np.all(np.isfinite(result.standard_errors[result.converged])))

    def test_complete_separation_is_reported_as_failure(self):
        base = np.linspace(-3.0, 3.0, 100)
        x = np.tile(base, (16, 1))
        y = (x > 0).astype(float)
        result = fit_logistic_irls(
            x,
            y,
            ComputeBackend("numpy", precision="float64"),
            IRLSSettings(maximum_iterations=50, coefficient_bound=25.0),
        )
        self.assertEqual(int(result.converged.sum()), 0)
        self.assertTrue(
            np.all(
                np.isin(
                    result.failure_reason,
                    ["coefficient_bound", "iteration_cap"],
                )
            )
        )

    def test_float32_controlled_input_preserves_status_shape(self):
        x, y = generate_controlled_logistic_data(
            seed=31,
            replications=32,
            sample_size=200,
            intercept=-0.5,
            slope=1.0,
        )
        result = fit_logistic_irls(
            x,
            y,
            ComputeBackend("numpy", precision="float32"),
            IRLSSettings(),
        )
        self.assertEqual(result.estimates.shape, (32,))
        self.assertEqual(result.converged.shape, (32,))
        self.assertEqual(result.iterations.shape, (32,))

    def test_protocol_is_machine_readable(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "benchmarks" / "protocol_v062.json").read_text())
        self.assertEqual(config["software_version"], "0.6.2")
        self.assertEqual(config["protocol_tag"], "protocol-v0.6.2")
        self.assertEqual(len(config["logistic_irls"]["conditions"]), 3)


if __name__ == "__main__":
    unittest.main()
