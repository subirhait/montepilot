"""Online numerical summaries used for adaptive Monte Carlo stopping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class OnlineMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        batch_n = int(x.size)
        batch_mean = float(x.mean())
        batch_m2 = float(np.sum((x - batch_mean) ** 2))
        if self.n == 0:
            self.n, self.mean, self.m2 = batch_n, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.n + batch_n
        self.mean += delta * batch_n / total
        self.m2 += batch_m2 + delta * delta * self.n * batch_n / total
        self.n = total

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def mcse(self) -> float:
        return self.sd / np.sqrt(self.n) if self.n > 1 else float("inf")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OnlineMoments":
        return cls(n=int(value["n"]), mean=float(value["mean"]), m2=float(value["m2"]))


@dataclass
class BinaryMoments:
    n: int = 0
    successes: int = 0

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values).reshape(-1)
        self.n += int(x.size)
        self.successes += int(np.sum(x.astype(bool)))

    @property
    def proportion(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def mcse(self) -> float:
        if self.n == 0:
            return float("inf")
        p = self.proportion
        return float(np.sqrt(p * (1 - p) / self.n))

