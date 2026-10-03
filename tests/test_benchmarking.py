import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from montepilot.benchmarking import (
    calibrated_iterations,
    load_r_benchmark_rows,
    refined_iterations,
    speedup_summary,
    timing_summary,
    validation_summary,
)
from montepilot.backends import ComputeBackend
from montepilot.precision_validation import (
    compare_estimates,
    evaluate_precision_case,
    precision_cases,
)


class BenchmarkingTests(unittest.TestCase):
    def test_calibrated_iterations_reaches_requested_block(self):
        self.assertEqual(calibrated_iterations(0.1, 0.5), 5)
        self.assertEqual(calibrated_iterations(1.2, 0.5), 1)
        self.assertEqual(calibrated_iterations(0.001, 1.0, 100), 100)

    def test_refined_iterations_corrects_an_undershoot(self):
        self.assertEqual(refined_iterations(10, 0.25, 0.5, 100), 21)
        self.assertEqual(refined_iterations(10, 0.6, 0.5, 100), 10)
        self.assertEqual(refined_iterations(80, 0.1, 0.5, 100), 100)

    def test_timing_summary_is_deterministic(self):
        values = [1.0, 1.1, 0.9, 1.05, 0.95]
        first = timing_summary(values, bootstrap_resamples=500, seed=7)
        second = timing_summary(values, bootstrap_resamples=500, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["median_seconds"], 1.0)
        self.assertLessEqual(first["median_ci_low"], first["median_seconds"])
        self.assertGreaterEqual(first["median_ci_high"], first["median_seconds"])

    def test_speedup_summary_uses_ratio_of_medians(self):
        result = speedup_summary(
            [4.0, 4.1, 3.9, 4.2, 3.8],
            [1.0, 1.1, 0.9, 1.2, 0.8],
            bootstrap_resamples=500,
            seed=9,
        )
        self.assertAlmostEqual(result["speedup"], 4.0)
        self.assertGreater(result["speedup_ci_low"], 0)

    def test_validation_summary_counts_failures(self):
        result = validation_summary(
            [
                {
                    "bias": 0.1,
                    "rmse": 0.2,
                    "coverage": 0.94,
                    "valid_reps": 99,
                    "failed_reps": 1,
                },
                {
                    "bias": -0.1,
                    "rmse": 0.3,
                    "coverage": 0.96,
                    "valid_reps": 98,
                    "failed_reps": 2,
                },
            ]
        )
        self.assertEqual(result["seeds"], 2)
        self.assertEqual(result["valid_reps"], 197)
        self.assertEqual(result["failed_reps"], 3)
        self.assertAlmostEqual(result["mean_bias"], 0.0)
        self.assertAlmostEqual(result["mean_coverage"], 0.95)

    def test_load_r_benchmark_rows_from_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "runs.zip"
            row = {
                "implementation": "base_r_batched",
                "timing_repeat": "4",
                "seed": "20261002",
                "reps": "20000",
                "n": "5000",
                "batch_size": "2000",
                "elapsed_seconds": "3.2",
                "reps_per_second": "6250",
                "estimate_mean": "0.5",
                "bias": "0",
                "rmse": "0.014",
                "r_version": "R test",
                "platform": "test-platform",
            }
            csv_path = Path(directory) / "run.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                writer.writerow(row)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(csv_path, arcname="run.csv")
            rows = load_r_benchmark_rows([archive_path])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["implementation"], "base_r_batched")
            self.assertEqual(rows[0]["elapsed_seconds"], 3.2)
            self.assertEqual(rows[0]["repeat"], 4)
            self.assertEqual(rows[0]["seed"], 20261002)

    def test_precision_comparison_detects_pass_and_failure(self):
        reference = [1.0, 2.0, 3.0]
        passing = compare_estimates(
            reference,
            [1.0, 2.000001, 2.999999],
            absolute_tolerance=1e-5,
            relative_tolerance=0.0,
        )
        failing = compare_estimates(
            reference,
            [1.0, 2.01, 3.0],
            absolute_tolerance=1e-5,
            relative_tolerance=0.0,
        )
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])

    def test_quick_precision_cases_pass_numpy_float32(self):
        backend = ComputeBackend("numpy", precision="float32")
        for case in precision_cases("quick"):
            result = evaluate_precision_case(case, backend, seed=17)
            self.assertTrue(result["passed"], msg=result)


if __name__ == "__main__":
    unittest.main()
