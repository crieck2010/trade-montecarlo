"""Monte Carlo option pricers (risk-neutral GBM).

European prices cross-check against Black-Scholes closed form (see tests);
Asian and barrier options are where Monte Carlo earns its keep — no closed
form, no problem.  Every pricer returns (price, standard_error) so you
know how much to trust the digits.
"""

from __future__ import annotations

import math

from .processes import simulate_gbm
from .rng import RandomStream


def _payoff_stats(payoffs: list[float], r: float, T: float) -> dict:
    n = len(payoffs)
    disc = math.exp(-r * T)
    price = disc * sum(payoffs) / n
    mean = sum(payoffs) / n
    var = sum((p - mean) ** 2 for p in payoffs) / max(n - 1, 1)
    se = disc * math.sqrt(var / n)
    return {"price": price, "std_error": se, "n_paths": n}


def european(s0: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", n_paths: int = 50_000, n_steps: int = 252,
             seed: int = 7, antithetic: bool = True) -> dict:
    """European call/put by risk-neutral Monte Carlo (mu = r)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    stream = RandomStream(seed)
    paths = simulate_gbm(stream, s0, n_paths, n_steps, T / n_steps, r, sigma,
                         antithetic)
    if kind == "call":
        payoffs = [max(p[-1] - K, 0.0) for p in paths]
    else:
        payoffs = [max(K - p[-1], 0.0) for p in paths]
    return {"kind": kind, **_payoff_stats(payoffs, r, T)}


def asian(s0: float, K: float, T: float, r: float, sigma: float,
          kind: str = "call", n_paths: int = 50_000, n_steps: int = 252,
          seed: int = 7, antithetic: bool = True) -> dict:
    """Arithmetic-average Asian call/put (average over all fixings)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    stream = RandomStream(seed)
    paths = simulate_gbm(stream, s0, n_paths, n_steps, T / n_steps, r, sigma,
                         antithetic)
    avgs = [sum(p[1:]) / n_steps for p in paths]
    if kind == "call":
        payoffs = [max(a - K, 0.0) for a in avgs]
    else:
        payoffs = [max(K - a, 0.0) for a in avgs]
    return {"kind": "asian-" + kind, **_payoff_stats(payoffs, r, T)}


def barrier_up_out_call(s0: float, K: float, B: float, T: float, r: float,
                        sigma: float, n_paths: int = 50_000,
                        n_steps: int = 252, seed: int = 7,
                        antithetic: bool = True) -> dict:
    """Up-and-out call: knocked out (worthless) if S ever touches B."""
    if B <= s0:
        raise ValueError("barrier B must be above s0 for up-and-out")
    stream = RandomStream(seed)
    paths = simulate_gbm(stream, s0, n_paths, n_steps, T / n_steps, r, sigma,
                         antithetic)
    payoffs = [0.0 if max(p) >= B else max(p[-1] - K, 0.0) for p in paths]
    return {"kind": "up-and-out-call", **_payoff_stats(payoffs, r, T)}
