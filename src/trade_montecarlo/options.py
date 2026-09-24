"""Monte Carlo option pricers (risk-neutral).

Vanilla Europeans cross-check against Black-Scholes closed form (see
tests); the exotics — digitals, lookbacks, barriers, geometric Asians,
American (Longstaff-Schwartz) — are where Monte Carlo earns its keep.
Every pricer returns (price, standard_error) so you know how much to
trust the digits.

Each pricer simulates risk-neutral GBM by default, but ``model="jump"``
or ``model="kou"`` prices the same payoff under jump-diffusion (drift
set to ``r`` with the jump compensator kept, so E[S_T] = s0·e^{rT} —
the standard risk-neutralized-drift approach), and ``paths=`` accepts
*any* pre-simulated path set (block bootstrap, GARCH, …) for fully
model-agnostic pricing.
"""

from __future__ import annotations

import math

from .processes import simulate_gbm, simulate_jump, simulate_kou
from .rng import RandomStream

PathSet = list[list[float]]


def _payoff_stats(payoffs: list[float], r: float, T: float) -> dict:
    n = len(payoffs)
    disc = math.exp(-r * T)
    price = disc * sum(payoffs) / n
    mean = sum(payoffs) / n
    var = sum((p - mean) ** 2 for p in payoffs) / max(n - 1, 1)
    se = disc * math.sqrt(var / n)
    return {"price": price, "std_error": se, "n_paths": n}


def _simulate(s0: float, T: float, r: float, sigma: float, n_paths: int,
              n_steps: int, seed: int, antithetic: bool,
              paths: PathSet | None, model: str,
              model_kw: dict | None) -> PathSet:
    """Paths for a pricer: caller-supplied, or simulated under ``model``."""
    if paths is not None:
        if not paths or any(len(p) != len(paths[0]) for p in paths):
            raise ValueError("paths must be non-empty with equal lengths")
        if len(paths[0]) < 2:
            raise ValueError("paths need at least 2 points")
        return paths
    stream = RandomStream(seed)
    dt = T / n_steps
    kw = dict(model_kw or {})
    if model == "gbm":
        return simulate_gbm(stream, s0, n_paths, n_steps, dt, r, sigma,
                            antithetic)
    if model == "jump":
        return simulate_jump(stream, s0, n_paths, n_steps, dt, r, sigma,
                             kw.get("lam", 0.5), kw.get("jump_mean", -0.05),
                             kw.get("jump_vol", 0.10), antithetic)
    if model == "kou":
        return simulate_kou(stream, s0, n_paths, n_steps, dt, r, sigma,
                            kw.get("lam", 0.5), kw.get("p_up", 0.4),
                            kw.get("eta1", 25.0), kw.get("eta2", 20.0),
                            antithetic)
    raise ValueError(f"unknown model {model!r}")


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def european(s0: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", n_paths: int = 50_000, n_steps: int = 252,
             seed: int = 7, antithetic: bool = True,
             paths: PathSet | None = None, model: str = "gbm",
             model_kw: dict | None = None) -> dict:
    """European call/put by risk-neutral Monte Carlo (mu = r)."""
    _check_kind(kind)
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    if kind == "call":
        payoffs = [max(p[-1] - K, 0.0) for p in sim]
    else:
        payoffs = [max(K - p[-1], 0.0) for p in sim]
    return {"kind": kind, "model": model, **_payoff_stats(payoffs, r, T)}


def digital(s0: float, K: float, T: float, r: float, sigma: float,
            kind: str = "call", cash: float = 1.0, n_paths: int = 50_000,
            n_steps: int = 252, seed: int = 7, antithetic: bool = True,
            paths: PathSet | None = None, model: str = "gbm",
            model_kw: dict | None = None) -> dict:
    """Cash-or-nothing: pays ``cash`` iff the option finishes in the money.

    The purest bet on direction — a discontinuous payoff, so Monte Carlo
    needs more paths than for smooth payoffs.  Closed form under GBM:
    cash·e^{−rT}·N(d₂) (call); validated in the test suite.
    """
    _check_kind(kind)
    if cash < 0:
        raise ValueError("cash must be >= 0")
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    if kind == "call":
        payoffs = [cash if p[-1] > K else 0.0 for p in sim]
    else:
        payoffs = [cash if p[-1] < K else 0.0 for p in sim]
    return {"kind": "digital-" + kind, "model": model,
            **_payoff_stats(payoffs, r, T)}


def asset_or_nothing(s0: float, K: float, T: float, r: float, sigma: float,
                     kind: str = "call", n_paths: int = 50_000,
                     n_steps: int = 252, seed: int = 7,
                     antithetic: bool = True, paths: PathSet | None = None,
                     model: str = "gbm", model_kw: dict | None = None) -> dict:
    """Asset-or-nothing: pays S_T iff in the money (the stock itself,
    conditional on finishing above/below K).  Closed form under GBM:
    s0·N(d₁) (call); validated in the test suite."""
    _check_kind(kind)
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    if kind == "call":
        payoffs = [p[-1] if p[-1] > K else 0.0 for p in sim]
    else:
        payoffs = [p[-1] if p[-1] < K else 0.0 for p in sim]
    return {"kind": "asset-or-nothing-" + kind, "model": model,
            **_payoff_stats(payoffs, r, T)}


def asian(s0: float, K: float, T: float, r: float, sigma: float,
          kind: str = "call", n_paths: int = 50_000, n_steps: int = 252,
          seed: int = 7, antithetic: bool = True,
          paths: PathSet | None = None, model: str = "gbm",
          model_kw: dict | None = None) -> dict:
    """Arithmetic-average Asian call/put (average over all fixings)."""
    _check_kind(kind)
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    steps = len(sim[0]) - 1
    avgs = [sum(p[1:]) / steps for p in sim]
    if kind == "call":
        payoffs = [max(a - K, 0.0) for a in avgs]
    else:
        payoffs = [max(K - a, 0.0) for a in avgs]
    return {"kind": "asian-" + kind, "model": model,
            **_payoff_stats(payoffs, r, T)}


def geometric_asian(s0: float, K: float, T: float, r: float, sigma: float,
                    kind: str = "call", n_paths: int = 50_000,
                    n_steps: int = 252, seed: int = 7,
                    antithetic: bool = True, paths: PathSet | None = None,
                    model: str = "gbm", model_kw: dict | None = None) -> dict:
    """Geometric-average Asian call/put.

    The geometric average of lognormal prices is lognormal, so a closed
    form exists (Kemna–Vorst) — used as a test cross-check and as a
    control variate for the arithmetic Asian.
    """
    _check_kind(kind)
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    steps = len(sim[0]) - 1
    gavgs = [math.exp(sum(math.log(x) for x in p[1:]) / steps) for p in sim]
    if kind == "call":
        payoffs = [max(a - K, 0.0) for a in gavgs]
    else:
        payoffs = [max(K - a, 0.0) for a in gavgs]
    return {"kind": "geometric-asian-" + kind, "model": model,
            **_payoff_stats(payoffs, r, T)}


def lookback(s0: float, T: float, r: float, sigma: float,
             kind: str = "call", strike: str = "floating",
             K: float | None = None, n_paths: int = 50_000,
             n_steps: int = 252, seed: int = 7, antithetic: bool = True,
             paths: PathSet | None = None, model: str = "gbm",
             model_kw: dict | None = None) -> dict:
    """Lookback option — the "buy at the low, sell at the high" payoff.

    Floating strike: call pays S_T − min(S), put pays max(S) − S_T (the
    strike is the realized extreme, so it always finishes in the money).
    Fixed strike: call pays max(S) − K, put pays K − min(S); ``K`` is
    required.  Path-dependent with no closed form under jumps — pure
    Monte Carlo territory.
    """
    _check_kind(kind)
    if strike not in ("floating", "fixed"):
        raise ValueError("strike must be 'floating' or 'fixed'")
    if strike == "fixed" and K is None:
        raise ValueError("fixed-strike lookback needs K")
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    payoffs = []
    for p in sim:
        hi, lo = max(p), min(p)
        if strike == "floating":
            payoffs.append(p[-1] - lo if kind == "call" else hi - p[-1])
        else:
            assert K is not None
            payoffs.append(max(hi - K, 0.0) if kind == "call"
                           else max(K - lo, 0.0))
    return {"kind": f"lookback-{strike}-{kind}", "model": model,
            **_payoff_stats(payoffs, r, T)}


_BARRIER_TYPES = ("up-in", "up-out", "down-in", "down-out")


def barrier(s0: float, K: float, B: float, T: float, r: float, sigma: float,
            kind: str = "call", barrier_type: str = "up-out",
            n_paths: int = 50_000, n_steps: int = 252, seed: int = 7,
            antithetic: bool = True, paths: PathSet | None = None,
            model: str = "gbm", model_kw: dict | None = None) -> dict:
    """Single-barrier knock-in/knock-out call/put.

    ``barrier_type`` picks the side (``up``: B > s0, ``down``: B < s0)
    and whether the option dies (``out``) or comes alive (``in``) when
    the barrier is touched.  In+out parity holds path-by-path: pricing
    both on the same paths sums to the European — checked in tests.
    """
    _check_kind(kind)
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}")
    up = barrier_type.startswith("up")
    if up and B <= s0:
        raise ValueError("up barrier B must be above s0")
    if not up and B >= s0:
        raise ValueError("down barrier B must be below s0")
    knock_in = barrier_type.endswith("in")
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    payoffs = []
    for p in sim:
        touched = (max(p) >= B) if up else (min(p) <= B)
        alive = touched if knock_in else not touched
        if not alive:
            payoffs.append(0.0)
            continue
        payoffs.append(max(p[-1] - K, 0.0) if kind == "call"
                       else max(K - p[-1], 0.0))
    return {"kind": f"{barrier_type}-{kind}", "model": model,
            **_payoff_stats(payoffs, r, T)}


def barrier_up_out_call(s0: float, K: float, B: float, T: float, r: float,
                        sigma: float, n_paths: int = 50_000,
                        n_steps: int = 252, seed: int = 7,
                        antithetic: bool = True) -> dict:
    """Up-and-out call (kept for backwards compatibility; see :func:`barrier`)."""
    res = barrier(s0, K, B, T, r, sigma, kind="call",
                  barrier_type="up-out", n_paths=n_paths, n_steps=n_steps,
                  seed=seed, antithetic=antithetic)
    res["kind"] = "up-and-out-call"
    return res


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (least-squares normal
    equations for the LSM regression)."""
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    n = 3
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(m[i][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-14:
            return [0.0, 0.0, 0.0]
        for i in range(col + 1, n):
            f = m[i][col] / m[col][col]
            for j in range(col, n + 1):
                m[i][j] -= f * m[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def american_lsm(s0: float, K: float, T: float, r: float, sigma: float,
                 kind: str = "put", n_paths: int = 50_000,
                 n_steps: int = 50, seed: int = 7, antithetic: bool = True,
                 paths: PathSet | None = None, model: str = "gbm",
                 model_kw: dict | None = None) -> dict:
    """American call/put by Longstaff-Schwartz least-squares Monte Carlo.

    Works backwards from expiry: at each exercise date, in-the-money
    paths regress their discounted future cashflows on {1, S, S²} and
    exercise when the immediate payoff beats the fitted continuation
    value.  Cross-checked against an independent Cox-Ross-Rubinstein
    binomial tree in the test suite.

    ``n_steps`` is the number of exercise dates (50 is plenty — early
    exercise is a coarse decision).  Works on any ``paths``/``model``,
    including jump-diffusion.
    """
    _check_kind(kind)
    sim = _simulate(s0, T, r, sigma, n_paths, n_steps, seed, antithetic,
                    paths, model, model_kw)
    steps = len(sim[0]) - 1
    dt = T / steps
    is_call = kind == "call"

    def exercise(s: float) -> float:
        return max(s - K, 0.0) if is_call else max(K - s, 0.0)

    # cashflow value and the step at which it is received
    cf = [exercise(p[-1]) for p in sim]
    ex_step = [steps] * len(sim)

    for i in range(steps - 1, 0, -1):
        itm = [p for p in range(len(sim)) if exercise(sim[p][i]) > 0]
        if len(itm) < 4:
            continue
        # discounted cashflows to time i, regressed on basis of S_i
        xs = [sim[p][i] / s0 for p in itm]
        ys = [cf[p] * math.exp(-r * dt * (ex_step[p] - i)) for p in itm]
        xtx = [[0.0] * 3 for _ in range(3)]
        xty = [0.0] * 3
        for x, y in zip(xs, ys):
            b = (1.0, x, x * x)
            for a in range(3):
                xty[a] += b[a] * y
                for c_ in range(3):
                    xtx[a][c_] += b[a] * b[c_]
        beta = _solve_3x3(xtx, xty)
        for p, x in zip(itm, xs):
            cont = beta[0] + beta[1] * x + beta[2] * x * x
            ex = exercise(sim[p][i])
            if ex > cont:
                cf[p] = ex
                ex_step[p] = i

    disc_cf = [c * math.exp(-r * dt * t) for c, t in zip(cf, ex_step)]
    n = len(disc_cf)
    price = sum(disc_cf) / n
    var = sum((c - price) ** 2 for c in disc_cf) / max(n - 1, 1)
    return {"kind": "american-" + kind + "-lsm", "model": model,
            "price": price, "std_error": math.sqrt(var / n),
            "n_paths": n, "n_steps": steps}
