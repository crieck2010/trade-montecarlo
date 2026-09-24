"""Jump-diffusion calibration: log-returns → Merton parameters.

Truncation-based fit, stdlib-only and deterministic:

1. Estimate the diffusion vol **jump-robustly** via bipower variation
   (Barndorff-Nielsen & Shephard): jumps inflate realized variance but
   wash out of ``mean(|r_t · r_{t−1}|)``, so bipower variation isolates σ.
2. Flag jumps by truncation: any return with ``|r| > thresh·σ·√dt``
   (default 4σ — a Gaussian exceeds it with probability ≈ 6e-5, so
   almost everything flagged is a genuine jump).
3. Read λ, jump_mean, jump_vol straight off the flagged jumps.

This is the honest, inspectable estimator: you can list exactly which
returns were called jumps.  It slightly *undercounts* small jumps (they
hide inside the diffusion), so λ is a lower bound and |jump_mean| a
touch overstated — fine for scenario generation, not for trading.
Moment-matching the cumulants instead would be more efficient in
theory and far noisier in practice (the 4th cumulant of 10 years of
daily data is dominated by a handful of observations).

Interop: feed it return series from anywhere (trade-data bars,
trade-eda describe output, backtest residuals); the fitted dict drops
straight into ``simulate_jump`` / ``SimConfig``.
"""

from __future__ import annotations

import math


def _moments(xs: list[float]) -> tuple[float, float, float, float]:
    """Population mean, variance, skewness, excess kurtosis."""
    n = len(xs)
    if n < 4:
        raise ValueError("need at least 4 returns")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    if v <= 0:
        raise ValueError("returns have no variance")
    sd = math.sqrt(v)
    skew = sum((x - m) ** 3 for x in xs) / n / sd ** 3
    kurt = sum((x - m) ** 4 for x in xs) / n / sd ** 4 - 3.0
    return m, v, skew, kurt


def bipower_vol(returns: list[float], dt: float = 1 / 252) -> float:
    """Jump-robust diffusion volatility (annualized).

    (π/2)·mean(|r_t·r_{t−1}|) → σ²·dt even when jumps are present, so
    ``sqrt(BV/dt)`` estimates the diffusive σ while plain realized vol
    overstates it.  On pure GBM data BV ≈ realized vol.
    """
    if len(returns) < 2:
        raise ValueError("need at least 2 returns")
    if dt <= 0:
        raise ValueError("dt must be positive")
    n = len(returns)
    bv_step = ((math.pi / 2.0) / (n - 1)
               * sum(abs(returns[t] * returns[t - 1]) for t in range(1, n)))
    return math.sqrt(max(bv_step, 0.0) / dt)


def merton_jump_cumulants(lam_step: float, jump_mean: float,
                          jump_vol: float) -> tuple[float, float, float]:
    """Per-step cumulants (var, skew-numerator, kurt-numerator) of the
    Merton jump component: Poisson(lam_step) jumps of N(jump_mean,
    jump_vol²) added to the log-return.

    κ₂ = λ(μJ² + σJ²),  κ₃ = λ(μJ³ + 3μJσJ²),
    κ₄ = λ(μJ⁴ + 6μJ²σJ² + 3σJ⁴).
    Kept for diagnostics and tests.
    """
    mj, sj = jump_mean, jump_vol
    c2 = lam_step * (mj ** 2 + sj ** 2)
    c3 = lam_step * (mj ** 3 + 3 * mj * sj ** 2)
    c4 = lam_step * (mj ** 4 + 6 * mj ** 2 * sj ** 2 + 3 * sj ** 4)
    return c2, c3, c4


def fit_merton_jumps(returns: list[float], dt: float = 1 / 252,
                     sigma: float | None = None,
                     thresh: float = 4.0) -> dict:
    """Fit Merton jump-diffusion parameters by truncation.

    Returns ``{"lam", "jump_mean", "jump_vol", "sigma", "sigma_bipower",
    "n_jumps", "jump_share_var", ...}`` with ``lam`` annualized — ready
    to unpack into :func:`trade_montecarlo.processes.simulate_jump`.

    ``sigma`` (annualized diffusion vol) defaults to the bipower
    estimate.  Returns with ``|r| > thresh·σ·√dt`` are flagged as jumps;
    fewer than 3 flags → ``lam = 0.0`` (no jump signature; pure
    diffusion).  Deterministic.
    """
    n = len(returns)
    if n < 4:
        raise ValueError("need at least 4 returns")
    sig = bipower_vol(returns, dt) if sigma is None else float(sigma)
    if sig < 0:
        raise ValueError("sigma must be >= 0")
    bv = sig  # keep the bipower estimate for the report
    sd_step = sig * math.sqrt(dt)
    m = sum(returns) / n
    v = sum((r - m) ** 2 for r in returns) / n
    if v <= 0:
        raise ValueError("returns have no variance")
    jumps = [r for r in returns if abs(r) > thresh * sd_step]
    if len(jumps) < 3:
        return {"lam": 0.0, "jump_mean": 0.0, "jump_vol": 0.0,
                "sigma": sig, "sigma_bipower": bv,
                "n_jumps": len(jumps), "jump_share_var": 0.0,
                "n_returns": n,
                "note": "fewer than 3 jumps flagged; pure diffusion"}
    mj = sum(jumps) / len(jumps)
    sj = math.sqrt(sum((j - mj) ** 2 for j in jumps) / len(jumps))
    lam_ann = len(jumps) / (n * dt)
    jump_var_step = (len(jumps) / n) * (mj ** 2 + sj ** 2)
    return {"lam": lam_ann, "jump_mean": mj, "jump_vol": sj,
            "sigma": sig, "sigma_bipower": bv,
            "n_jumps": len(jumps),
            "jump_share_var": jump_var_step / v if v > 0 else 0.0,
            "n_returns": n}
