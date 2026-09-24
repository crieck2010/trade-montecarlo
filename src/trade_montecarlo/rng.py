"""Seeded random streams and samplers.

``RandomStream`` wraps ``random.Random`` with finance's workhorse
distributions: uniform, normal (Box-Muller with spare caching),
Student-t (via Marsaglia-Tsang gamma).  ``spawn()`` derives independent
child streams so parallel workers stay reproducible — same seed, same
paths, every time.
"""

from __future__ import annotations

import math
import random


class RandomStream:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._spare: float | None = None

    # ------------------------------------------------------------------
    def uniform(self) -> float:
        return self._rng.random()

    def normal(self) -> float:
        """Standard normal via Box-Muller (caches the spare draw)."""
        if self._spare is not None:
            z, self._spare = self._spare, None
            return z
        u1 = max(self._rng.random(), 1e-300)
        u2 = self._rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return r * math.cos(theta)

    def _gamma(self, shape: float) -> float:
        """Gamma(shape, rate=1) via Marsaglia-Tsang (shape >= 1)."""
        if shape < 1.0:  # boost: Gamma(k) = Gamma(k+1) * U^(1/k)
            return self._gamma(shape + 1.0) * self._rng.random() ** (1.0 / shape)
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = (1.0 + c * x) ** 3
            if v <= 0.0:
                continue
            u = self._rng.random()
            if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
                return d * v

    def student_t(self, df: float) -> float:
        """Student-t with ``df`` degrees of freedom (fat tails)."""
        if df <= 0:
            raise ValueError("df must be positive")
        z = self.normal()
        chi2 = 2.0 * self._gamma(df / 2.0)
        return z / math.sqrt(chi2 / df)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return math.exp(mean + sigma * self.normal())

    # ------------------------------------------------------------------
    def spawn(self, n: int) -> list["RandomStream"]:
        """Derive ``n`` independent child streams (deterministic)."""
        return [RandomStream(self._derive(i)) for i in range(n)]

    def _derive(self, i: int) -> int:
        h = random.Random(f"{self.seed}:{i}").getrandbits(63)
        return h
