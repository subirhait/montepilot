"""NumPy and optional PyTorch compute backends."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch():
    if not torch_installed():
        raise RuntimeError("PyTorch is not installed; install montepilot[torch]")
    import torch

    return torch


@dataclass(frozen=True)
class ComputeBackend:
    """Small common numerical surface used by vectorized simulation designs."""

    name: str
    precision: str = "auto"

    @property
    def is_torch(self) -> bool:
        return self.name.startswith("torch_")

    @property
    def device(self) -> str:
        if self.name == "torch_cuda":
            return "cuda"
        if self.name == "torch_xpu":
            return "xpu"
        return "cpu"

    def _torch_dtype(self):
        """Return a portable default dtype for the selected torch device.

        Intel client GPUs do not consistently provide FP64 kernels. XPU uses
        FP32 so supported Arc devices can execute the complete simulation;
        CPU and CUDA retain the original FP64 behavior.
        """

        torch = _torch()
        if self.precision == "float32":
            return torch.float32
        if self.precision == "float64":
            return torch.float64
        return torch.float32 if self.name == "torch_xpu" else torch.float64

    @property
    def dtype_name(self) -> str:
        if self.precision != "auto":
            return self.precision
        return "float32" if self.name == "torch_xpu" else "float64"

    def _numpy_dtype(self):
        return np.float32 if self.dtype_name == "float32" else np.float64

    def normal(self, shape: tuple[int, ...], seed: int, mean: float = 0.0, sd: float = 1.0):
        if self.is_torch:
            torch = _torch()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            values = torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=self._torch_dtype(),
            )
            return values * sd + mean
        rng = np.random.default_rng(seed)
        dtype = self._numpy_dtype()
        values = rng.standard_normal(size=shape, dtype=dtype)
        return values * np.asarray(sd, dtype=dtype) + np.asarray(mean, dtype=dtype)

    def uniform(self, shape: tuple[int, ...], seed: int, low: float = 0.0, high: float = 1.0):
        if self.is_torch:
            torch = _torch()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            values = torch.rand(
                shape,
                generator=generator,
                device=self.device,
                dtype=self._torch_dtype(),
            )
            return values * (high - low) + low
        rng = np.random.default_rng(seed)
        dtype = self._numpy_dtype()
        values = rng.random(size=shape, dtype=dtype)
        return values * np.asarray(high - low, dtype=dtype) + np.asarray(low, dtype=dtype)

    def integers(self, shape: tuple[int, ...], seed: int, low: int, high: int):
        if self.is_torch:
            torch = _torch()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            return torch.randint(
                low,
                high,
                shape,
                generator=generator,
                device=self.device,
                dtype=torch.int64,
            )
        rng = np.random.default_rng(seed)
        return rng.integers(low=low, high=high, size=shape)

    def asarray(self, value: Any):
        if self.is_torch:
            return _torch().as_tensor(
                value,
                device=self.device,
                dtype=self._torch_dtype(),
            )
        return np.asarray(value, dtype=self._numpy_dtype())

    def mean(self, x: Any, axis: int | None = None):
        if self.is_torch:
            return x.mean() if axis is None else x.mean(dim=axis)
        return np.mean(x, axis=axis)

    def std(self, x: Any, axis: int | None = None, ddof: int = 1):
        if self.is_torch:
            return x.std(unbiased=(ddof == 1)) if axis is None else x.std(dim=axis, unbiased=(ddof == 1))
        return np.std(x, axis=axis, ddof=ddof)

    def sqrt(self, x: Any):
        if self.is_torch:
            return _torch().sqrt(x)
        return np.sqrt(x)

    def sum(self, x: Any, axis: int | None = None):
        if self.is_torch:
            return x.sum() if axis is None else x.sum(dim=axis)
        return np.sum(x, axis=axis)

    def clip(self, x: Any, low: float, high: float):
        if self.is_torch:
            return _torch().clamp(x, min=low, max=high)
        return np.clip(x, low, high)

    def sigmoid(self, x: Any):
        if self.is_torch:
            return _torch().sigmoid(x)
        return 1.0 / (1.0 + np.exp(-x))

    def matmul(self, left: Any, right: Any):
        if self.is_torch:
            return _torch().matmul(left, right)
        return np.matmul(left, right)

    def transpose_last2(self, x: Any):
        if self.is_torch:
            return x.transpose(-2, -1)
        return np.swapaxes(x, -2, -1)

    def expand_last(self, x: Any):
        if self.is_torch:
            return x.unsqueeze(-1)
        return np.expand_dims(x, axis=-1)

    def squeeze_last(self, x: Any):
        if self.is_torch:
            return x.squeeze(-1)
        return np.squeeze(x, axis=-1)

    def solve(self, coefficients: Any, values: Any):
        if self.is_torch:
            return _torch().linalg.solve(coefficients, values)
        return np.linalg.solve(coefficients, values)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.is_torch:
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.name == "torch_cuda":
            _torch().cuda.synchronize()
        elif self.name == "torch_xpu":
            _torch().xpu.synchronize()


def _device_available(torch: Any, device: str) -> bool:
    namespace = getattr(torch, device, None)
    if namespace is None:
        return False
    try:
        return bool(namespace.is_available())
    except Exception:
        return False


def available_backends() -> list[str]:
    names = ["numpy"]
    if torch_installed():
        torch = _torch()
        names.append("torch_cpu")
        if _device_available(torch, "xpu"):
            names.append("torch_xpu")
        if _device_available(torch, "cuda"):
            names.append("torch_cuda")
    return names


def resolve_backend(name: str, precision: str = "auto") -> ComputeBackend:
    if name == "auto":
        available = available_backends()
        name = next(
            (candidate for candidate in ("torch_cuda", "torch_xpu", "numpy") if candidate in available),
            "numpy",
        )
    if name not in available_backends():
        raise RuntimeError(f"Backend {name!r} is unavailable; available: {available_backends()}")
    return ComputeBackend(name, precision=precision)


def probe_backends(
    workload: Callable[[ComputeBackend], Any],
    repeats: int = 2,
    precision: str = "auto",
) -> tuple[ComputeBackend, dict[str, float], dict[str, float], dict[str, str]]:
    """Benchmark available backends and return the fastest successful option."""

    timings: dict[str, float] = {}
    warmup_timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    available = available_backends()
    candidates = ["numpy"]
    for name in ("torch_cpu", "torch_cuda", "torch_xpu"):
        if name in available:
            candidates.append(name)

    for name in candidates:
        backend = ComputeBackend(name, precision=precision)
        try:
            warmup_started = time.perf_counter()
            workload(backend)
            backend.synchronize()
            warmup_timings[name] = time.perf_counter() - warmup_started
            started = time.perf_counter()
            for _ in range(max(1, repeats)):
                workload(backend)
            backend.synchronize()
            timings[name] = (time.perf_counter() - started) / max(1, repeats)
        except Exception as exc:  # a failed optional backend must not kill auto mode
            failures[name] = f"{type(exc).__name__}: {exc}"

    if not timings:
        raise RuntimeError(f"No compute backend completed the probe: {failures}")
    winner = min(timings, key=timings.get)
    return ComputeBackend(winner, precision=precision), timings, warmup_timings, failures
