import unittest

import numpy as np

from montepilot.backends import ComputeBackend
from montepilot.design import BatchEstimate
from montepilot.validation import validate_batch


class ValidationTests(unittest.TestCase):
    def test_rejects_wrong_batch_length(self):
        with self.assertRaises(ValueError):
            validate_batch(
                BatchEstimate(estimates=np.array([1.0, 2.0])),
                ComputeBackend("numpy"),
                expected_size=3,
                finite_tolerance=0.99,
            )

    def test_nonfinite_values_are_invalid(self):
        estimates, _, valid, warnings = validate_batch(
            BatchEstimate(estimates=np.array([1.0, np.nan])),
            ComputeBackend("numpy"),
            expected_size=2,
            finite_tolerance=0.99,
        )
        self.assertEqual(estimates.size, 2)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()

