import math
import unittest

import numpy as np

from montepilot.metrics import BinaryMoments, OnlineMoments


class OnlineMomentsTests(unittest.TestCase):
    def test_chunked_update_matches_numpy(self):
        values = np.arange(1.0, 101.0)
        moments = OnlineMoments()
        moments.update(values[:17])
        moments.update(values[17:])
        self.assertEqual(moments.n, 100)
        self.assertAlmostEqual(moments.mean, float(values.mean()))
        self.assertAlmostEqual(moments.sd, float(values.std(ddof=1)))
        self.assertAlmostEqual(moments.mcse, float(values.std(ddof=1) / math.sqrt(100)))

    def test_binary_summary(self):
        moments = BinaryMoments()
        moments.update(np.array([True, False, True, True]))
        self.assertEqual(moments.n, 4)
        self.assertEqual(moments.successes, 3)
        self.assertAlmostEqual(moments.proportion, 0.75)


if __name__ == "__main__":
    unittest.main()

