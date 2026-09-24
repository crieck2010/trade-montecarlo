"""Price-process models: GBM, Merton jump-diffusion, Kou jump-diffusion,
GARCH(1,1), OU.

Each simulator takes a ``RandomStream`` and returns a list of paths; every
path is a list of ``n_steps + 1`` prices starting at ``s0``.  Same seed →
identical paths, always.  With ``antithetic=True`` each path is paired
with its mirror image (negated shocks), a classic variance-reduction
technique — both paths are unbiased, their average is tighter.
"""

from __future__ import annotations

import math

from .rng import RandomStream


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0:
                    raise ValueError("correlation matrix not positive-definite")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _pairs(n_paths: int, antithetic: bool) -> tuple[int, bool]:
    if antithetic and n_paths % 2:
        raise ValueError("antithetic needs an even n_paths")
    return (n_paths // 2 if antithetic else n_paths), antithetic


def simulate_gbm(stream: RandomStream, s0: float, n_paths: int, n_steps: int,
                 dt: float, mu: float, sigma: float,
                 antithetic: bool = False) -> list[list[float]]:
    """Geometric Brownian motion: dS/S = μ·dt + σ·dW."""
    if s0 <= 0 or sigma < 0 or dt <= 0:
        raise ValueError("need s0 > 0, sigma >= 0, dt > 0")
    drift, vol = (mu - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt)
    n_half, anti = _pairs(n_paths, antithetic)
    paths = []
    for _ in range(n_half):
        zs = [stream.normal() for _ in range(n_steps)]
        for sign in (1.0, -1.0) if anti else (1.0,):
            s, path = s0, [s0]
            for z in zs:
                s *= math.exp(drift + vol * sign * z)
                path.append(s)
            paths.append(path)
    return paths


def _poisson(stream: RandomStream, lam: float) -> int:
    """Knuth's method — fine for the small λ·dt of jump models."""
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= stream.uniform()
        if p <= l:
            return k - 1


def simulate_jump(stream: RandomStream, s0: float, n_paths: int, n_steps: int,
                  dt: float, mu: float, sigma: float, lam: float,
                  jump_mean: float = -0.05, jump_vol: float = 0.10,
                  antithetic: bool = False) -> list[list[float]]:
    """Merton jump-diffusion: GBM plus Poisson jumps with lognormal sizes.

    The drift carries the compensator −λ·κ (κ = E[jump multiplier − 1]) so
    E[S] still grows at μ.  With ``lam=0`` this is *exactly* GBM — jump
    draws are skipped, so the normal sequence aligns.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    kappa = math.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0
    drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt
    vol = sigma * math.sqrt(dt)
    n_half, anti = _pairs(n_paths, antithetic)
    paths = []
    for _ in range(n_half):
        zs = [stream.normal() for _ in range(n_steps)]
        jumps: list[list[float]] = []  # per step: list of jump-size normals
        if lam > 0:
            for _ in range(n_steps):
                jumps.append([stream.normal()
                              for _ in range(_poisson(stream, lam * dt))])
        else:
            jumps = [[] for _ in range(n_steps)]
        for sign in (1.0, -1.0) if anti else (1.0,):
            s, path = s0, [s0]
            for z, js in zip(zs, jumps):
                s *= math.exp(drift + vol * sign * z)
                for jz in js:
                    s *= math.exp(jump_mean + jump_vol * sign * jz)
                path.append(s)
            paths.append(path)
    return paths


def _kou_kappa(p_up: float, eta1: float, eta2: float) -> float:
    """E[jump multiplier − 1] for double-exponential jumps.

    E[e^Y] = p·η₁/(η₁−1) + (1−p)·η₂/(η₂+1), so
    κ = p·η₁/(η₁−1) + (1−p)·η₂/(η₂+1) − 1.  Needs η₁ > 1.
    """
    if eta1 <= 1.0:
        raise ValueError("eta1 must exceed 1 for a finite E[e^Y]")
    return (p_up * eta1 / (eta1 - 1.0)
            + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0)


def _kou_jump(stream: RandomStream, p_up: float, eta1: float,
              eta2: float) -> float:
    """One double-exponential jump in log-price space.

    With prob p_up: Y = +Exp(η₁); else Y = −Exp(η₂).  Uniforms are
    clamped away from 0/1 so the log never blows up.
    """
    u = min(max(stream.uniform(), 1e-12), 1.0 - 1e-12)
    if stream.uniform() < p_up:
        return -math.log(u) / eta1
    return math.log(u) / eta2


def simulate_kou(stream: RandomStream, s0: float, n_paths: int, n_steps: int,
                 dt: float, mu: float, sigma: float, lam: float,
                 p_up: float = 0.4, eta1: float = 25.0, eta2: float = 20.0,
                 antithetic: bool = False) -> list[list[float]]:
    """Kou (2002) double-exponential jump-diffusion.

    Jumps arrive as a Poisson(λ) process; each jump's log-size is
    asymmetric double-exponential: up-jumps Exp(η₁) with prob ``p_up``,
    down-jumps −Exp(η₂) otherwise.  Unlike Merton's symmetric lognormal
    jumps, Kou reproduces the equity skew — frequent small up-moves,
    rare violent down-moves — and the volatility smirk.

    The drift carries the compensator −λ·κ (see :func:`_kou_kappa`) so
    E[S] still grows at μ.  With ``lam=0`` this is *exactly* GBM — jump
    draws are skipped, so the normal sequence aligns.

    Antithetic pairing mirrors the diffusion shocks; the jump *times and
    sizes* are shared within a pair (mirroring an asymmetric jump law
    would change its distribution).  Each path stays unbiased.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if not 0.0 <= p_up <= 1.0:
        raise ValueError("p_up must be in [0, 1]")
    if eta1 <= 1.0 or eta2 <= 0.0:
        raise ValueError("need eta1 > 1 and eta2 > 0")
    kappa = _kou_kappa(p_up, eta1, eta2)
    drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt
    vol = sigma * math.sqrt(dt)
    n_half, anti = _pairs(n_paths, antithetic)
    paths = []
    for _ in range(n_half):
        zs = [stream.normal() for _ in range(n_steps)]
        jumps: list[list[float]] = [[] for _ in range(n_steps)]
        if lam > 0:
            for i in range(n_steps):
                jumps[i] = [_kou_jump(stream, p_up, eta1, eta2)
                            for _ in range(_poisson(stream, lam * dt))]
        for sign in (1.0, -1.0) if anti else (1.0,):
            s, path = s0, [s0]
            for z, js in zip(zs, jumps):
                s *= math.exp(drift + vol * sign * z)
                for y in js:
                    s *= math.exp(y)
                path.append(s)
            paths.append(path)
    return paths


def simulate_garch(stream: RandomStream, s0: float, n_paths: int, n_steps: int,
                   dt: float = 1.0, omega: float = 1e-6, alpha: float = 0.08,
                   beta: float = 0.90, innov: str = "normal",
                   df: float = 6.0, antithetic: bool = False) -> list[list[float]]:
    """GARCH(1,1) log-returns: volatility clusters, then mean-reverts.

    σ²_t = ω + α·r²_{t−1} + β·σ²_{t−1}, r_t = σ_t·z_t.  Innovations are
    normal or Student-t (``df``) for fat tails.  Variance starts at the
    unconditional level ω/(1−α−β).
    """
    if not alpha + beta < 1.0:
        raise ValueError("need alpha + beta < 1 for stationarity")
    if innov not in ("normal", "t"):
        raise ValueError("innov must be 'normal' or 't'")
    uncond = omega / (1.0 - alpha - beta)
    n_half, anti = _pairs(n_paths, antithetic)
    paths = []
    for _ in range(n_half):
        draws = [stream.normal() if innov == "normal"
                 else stream.student_t(df) for _ in range(n_steps)]
        for sign in (1.0, -1.0) if anti else (1.0,):
            s, path, var = s0, [s0], uncond
            for z in draws:
                r = math.sqrt(var) * sign * z
                s *= math.exp(r * math.sqrt(dt))
                var = omega + alpha * r * r + beta * var
                path.append(s)
            paths.append(path)
    return paths


def simulate_ou(stream: RandomStream, s0: float, n_paths: int, n_steps: int,
                dt: float, theta: float, long_mean: float, sigma: float,
                antithetic: bool = False) -> list[list[float]]:
    """Ornstein-Uhlenbeck: dx = θ(μ−x)dt + σ√dt·dW.  Mean-reverting spreads.

    A natural scenario engine for pairs-trading spreads (see trade-pairs'
    half-life: θ ≈ ln 2 / half_life).
    """
    if theta <= 0:
        raise ValueError("theta must be positive")
    n_half, anti = _pairs(n_paths, antithetic)
    paths = []
    for _ in range(n_half):
        zs = [stream.normal() for _ in range(n_steps)]
        for sign in (1.0, -1.0) if anti else (1.0,):
            x, path = s0, [s0]
            for z in zs:
                x += theta * (long_mean - x) * dt + sigma * math.sqrt(dt) * sign * z
                path.append(x)
            paths.append(path)
    return paths


def simulate_correlated_gbm(stream: RandomStream, s0: list[float],
                            mu: list[float], sigma: list[float],
                            corr: list[list[float]], n_paths: int,
                            n_steps: int, dt: float,
                            antithetic: bool = False) -> list[list[list[float]]]:
    """Multi-asset GBM with a correlation matrix.  Returns [asset][path][step]."""
    n = len(s0)
    if not (len(mu) == len(sigma) == len(corr) == n):
        raise ValueError("s0/mu/sigma/corr dimension mismatch")
    L = _cholesky(corr)
    drifts = [(m - 0.5 * s ** 2) * dt for m, s in zip(mu, sigma)]
    vols = [s * math.sqrt(dt) for s in sigma]
    n_half, anti = _pairs(n_paths, antithetic)
    out: list[list[list[float]]] = [[[] for _ in range(n_half * (2 if anti else 1))]
                                    for _ in range(n)]
    for p in range(n_half):
        shocks = [[stream.normal() for _ in range(n)] for _ in range(n_steps)]
        for sign in (1.0, -1.0) if anti else (1.0,):
            for a in range(n):
                s, path = s0[a], [s0[a]]
                for z in shocks:
                    zc = sum(L[a][k] * z[k] for k in range(a + 1))
                    s *= math.exp(drifts[a] + vols[a] * sign * zc)
                    path.append(s)
                idx = p * (2 if anti else 1) + (1 if sign < 0 else 0)
                out[a][idx] = path
    return out
