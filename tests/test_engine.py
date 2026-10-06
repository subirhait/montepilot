import json
import tempfile
import unittest
from unittest.mock import patch

from montepilot import RunConfig, SimulationDesign, SimulationRunner, run
from montepilot.backends import ComputeBackend
from montepilot.design import BatchContext, BatchEstimate, derive_seed
from montepilot.examples import normal_mean_design


class EngineTests(unittest.TestCase):
    def test_stream_seeds_hash_the_full_design_position(self):
        context = BatchContext(
            backend=ComputeBackend("numpy"),
            seed=derive_seed(42, 3, 7, 0),
            condition_index=3,
            batch_index=7,
            master_seed=42,
        )
        self.assertEqual(context.seed_for(0), derive_seed(42, 3, 7, 0))
        self.assertEqual(context.seed_for(1), derive_seed(42, 3, 7, 1))
        self.assertNotEqual(context.seed_for(0), context.seed_for(1))
        with self.assertRaises(ValueError):
            context.seed_for(-1)

    def test_auto_expected_reps_validation(self):
        with self.assertRaises(ValueError):
            RunConfig(min_reps=100, max_reps=1000, auto_expected_reps=99)
        with self.assertRaises(ValueError):
            RunConfig(min_reps=100, max_reps=1000, auto_expected_reps=1001)

    def test_reproducible_numpy_run(self):
        config = RunConfig(
            backend="numpy",
            batch_size=100,
            min_reps=200,
            max_reps=500,
            target_mcse=None,
            seed=42,
        )
        design = normal_mean_design(sample_sizes=[20])
        first = SimulationRunner(config).run(design).conditions[0]
        second = SimulationRunner(config).run(design).conditions[0]
        self.assertEqual(first.attempted_reps, 500)
        self.assertEqual(first.estimate_mean, second.estimate_mean)
        self.assertEqual(first.rmse, second.rmse)

    def test_adaptive_stopping(self):
        config = RunConfig(
            backend="numpy",
            batch_size=100,
            min_reps=200,
            max_reps=5000,
            target_mcse=0.01,
            seed=43,
        )
        result = SimulationRunner(config).run(normal_mean_design(sample_sizes=[100])).conditions[0]
        self.assertTrue(result.stopped_early)
        self.assertLessEqual(result.estimate_mcse, 0.01)
        self.assertIsNotNone(result.coverage)

    def test_joint_estimate_and_coverage_stopping(self):
        config = RunConfig(
            backend="numpy",
            batch_size=100,
            min_reps=100,
            max_reps=2000,
            target_mcse=1.0,
            target_coverage_mcse=0.02,
            stopping_rule="all",
            seed=44,
        )
        result = SimulationRunner(config).run(normal_mean_design(sample_sizes=[100])).conditions[0]
        self.assertTrue(result.stopped_early)
        self.assertLessEqual(result.estimate_mcse, 1.0)
        self.assertLessEqual(result.coverage_mcse, 0.02)
        self.assertIn("coverage MCSE", result.stop_reason)

    def test_coverage_target_without_standard_errors_reaches_maximum(self):
        def generator(context, condition, batch_size):
            return context.backend.normal((batch_size, 5), seed=context.seed)

        def estimator(data, context, condition):
            return BatchEstimate(estimates=context.backend.mean(data, axis=1))

        design = SimulationDesign(
            name="no-standard-errors",
            conditions=[{"n": 5}],
            generator=generator,
            estimator=estimator,
            truth=0.0,
        )
        config = RunConfig(
            backend="numpy",
            batch_size=100,
            min_reps=100,
            max_reps=300,
            target_mcse=1.0,
            target_coverage_mcse=0.02,
            stopping_rule="all",
        )
        report = SimulationRunner(config).run(design)
        self.assertEqual(report.conditions[0].attempted_reps, 300)
        self.assertFalse(report.conditions[0].stopped_early)
        self.assertTrue(any("could not be evaluated" in item for item in report.advice))

    def test_high_level_run_api(self):
        report = run(
            normal_mean_design(sample_sizes=[20]),
            backend="numpy",
            precision="float32",
            batch_size=100,
            min_reps=100,
            max_reps=100,
            target_mcse=None,
        )
        self.assertEqual(report.selected_precision, "float32")
        self.assertEqual(report.conditions[0].precision, "float32")

    def test_scheduler_accounts_for_warmup_and_projected_work(self):
        design = normal_mean_design(sample_sizes=[50, 200, 1000])
        short = RunConfig(
            backend="auto",
            batch_size=100,
            min_reps=1000,
            max_reps=100000,
            target_mcse=0.1,
            auto_probe_batch_size=100,
        )
        long = RunConfig(
            backend="auto",
            batch_size=100,
            min_reps=1000,
            max_reps=100000,
            target_mcse=None,
            auto_probe_batch_size=100,
        )
        probe_result = (
            ComputeBackend("numpy"),
            {"numpy": 0.03, "torch_xpu": 0.003},
            {"numpy": 0.001, "torch_xpu": 1.0},
            {},
        )
        with patch("montepilot.engine.probe_backends", return_value=probe_result), patch(
            "montepilot.backends.available_backends", return_value=["numpy", "torch_xpu"]
        ):
            short_backend = SimulationRunner(short)._select_backend(design)[0]
            long_backend = SimulationRunner(long)._select_backend(design)[0]
        self.assertEqual(short_backend.name, "numpy")
        self.assertEqual(long_backend.name, "torch_xpu")

    def test_json_report(self):
        config = RunConfig(backend="numpy", batch_size=50, min_reps=100, max_reps=100, target_mcse=None)
        report = SimulationRunner(config).run(normal_mean_design(sample_sizes=[10]))
        payload = json.loads(report.to_json())
        self.assertEqual(payload["selected_backend"], "numpy")
        self.assertEqual(payload["conditions"][0]["attempted_reps"], 100)

    def test_auto_backend_falls_back_to_available_backend(self):
        config = RunConfig(
            backend="auto",
            batch_size=50,
            min_reps=100,
            max_reps=100,
            target_mcse=None,
        )
        report = SimulationRunner(config).run(normal_mean_design(sample_sizes=[10]))
        self.assertIn(
            report.selected_backend,
            {"numpy", "torch_cpu", "torch_cuda", "torch_xpu"},
        )

    def test_checkpoint_resume_does_not_repeat_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            first_config = RunConfig(
                backend="numpy",
                batch_size=50,
                min_reps=100,
                max_reps=100,
                target_mcse=None,
                checkpoint_dir=directory,
                checkpoint_every_batches=1,
                seed=99,
            )
            design = normal_mean_design(sample_sizes=[15])
            first = SimulationRunner(first_config).run(design).conditions[0]
            second = SimulationRunner(first_config).run(design).conditions[0]
            self.assertEqual(first.estimate_mean, second.estimate_mean)
            self.assertEqual(second.attempted_reps, 100)

    def test_checkpoint_rejects_changed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            design = normal_mean_design(sample_sizes=[15])
            original = RunConfig(
                backend="numpy",
                batch_size=50,
                min_reps=100,
                max_reps=100,
                target_mcse=None,
                checkpoint_dir=directory,
                checkpoint_every_batches=1,
                seed=99,
            )
            SimulationRunner(original).run(design)
            changed = RunConfig(
                backend="numpy",
                batch_size=50,
                min_reps=100,
                max_reps=100,
                target_mcse=None,
                checkpoint_dir=directory,
                checkpoint_every_batches=1,
                seed=100,
            )
            with self.assertRaises(RuntimeError):
                SimulationRunner(changed).run(design)

    def test_checkpoint_rejects_changed_stopping_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            design = normal_mean_design(sample_sizes=[15])
            original = RunConfig(
                backend="numpy",
                batch_size=50,
                min_reps=100,
                max_reps=100,
                target_mcse=0.01,
                stopping_rule="all",
                checkpoint_dir=directory,
                checkpoint_every_batches=1,
            )
            SimulationRunner(original).run(design)
            changed = RunConfig(
                backend="numpy",
                batch_size=50,
                min_reps=100,
                max_reps=100,
                target_mcse=0.02,
                stopping_rule="all",
                checkpoint_dir=directory,
                checkpoint_every_batches=1,
            )
            with self.assertRaises(RuntimeError):
                SimulationRunner(changed).run(design)


if __name__ == "__main__":
    unittest.main()
