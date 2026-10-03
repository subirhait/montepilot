import unittest
from unittest.mock import patch

import numpy as np

from montepilot.backends import ComputeBackend, available_backends, probe_backends


class _FakeDevice:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    xpu = _FakeDevice(True)
    cuda = _FakeDevice(False)


class BackendTests(unittest.TestCase):
    def test_xpu_device_mapping(self):
        self.assertEqual(ComputeBackend("torch_xpu").device, "xpu")

    def test_available_backends_detects_xpu(self):
        with patch("montepilot.backends.torch_installed", return_value=True), patch(
            "montepilot.backends._torch", return_value=_FakeTorch()
        ):
            self.assertEqual(available_backends(), ["numpy", "torch_cpu", "torch_xpu"])

    def test_numpy_matrix_surface(self):
        backend = ComputeBackend("numpy")
        coefficients = np.array([[[2.0, 0.0], [0.0, 4.0]]])
        values = np.array([[[6.0], [8.0]]])
        solution = backend.solve(coefficients, values)
        np.testing.assert_allclose(solution, np.array([[[3.0], [2.0]]]))
        self.assertEqual(backend.transpose_last2(coefficients).shape, (1, 2, 2))

    def test_numpy_integer_stream_is_reproducible(self):
        backend = ComputeBackend("numpy")
        first = backend.integers((5, 4), seed=91, low=0, high=10)
        second = backend.integers((5, 4), seed=91, low=0, high=10)
        np.testing.assert_array_equal(first, second)

    def test_numpy_float32_precision(self):
        backend = ComputeBackend("numpy", precision="float32")
        values = backend.normal((4, 3), seed=92, mean=0.5, sd=2.0)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(backend.asarray([1.0, 2.0]).dtype, np.float32)
        self.assertEqual(
            backend.uniform((4, 3), seed=93, low=-1.0, high=2.0).dtype,
            np.float32,
        )

    def test_numpy_float32_stream_is_reproducible(self):
        backend = ComputeBackend("numpy", precision="float32")
        first = backend.normal((5, 4), seed=94)
        second = backend.normal((5, 4), seed=94)
        np.testing.assert_array_equal(first, second)

    def test_auto_precision_preserves_numpy_float64(self):
        backend = ComputeBackend("numpy", precision="auto")
        self.assertEqual(backend.normal((2, 2), seed=93).dtype, np.float64)
        self.assertEqual(backend.dtype_name, "float64")

    def test_probe_includes_torch_cpu(self):
        observed = []

        def workload(backend):
            observed.append(backend.name)

        with patch(
            "montepilot.backends.available_backends",
            return_value=["numpy", "torch_cpu"],
        ):
            _, timings, warmups, failures = probe_backends(workload, repeats=1)
        self.assertEqual(set(timings), {"numpy", "torch_cpu"})
        self.assertEqual(set(warmups), {"numpy", "torch_cpu"})
        self.assertFalse(failures)
        self.assertIn("torch_cpu", observed)


if __name__ == "__main__":
    unittest.main()
