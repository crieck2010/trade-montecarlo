"""Analytics over simulated paths and P&L.

Sign convention: P&L is positive for profit.  VaR/CVaR are reported as
*positive loss amounts*: VaR₉₅ = the loss exceeded with 5% probability.
"""

from __future__ import annotations

import math


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        raise ValueError("empty sample")
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def var(pnl: list[float], alpha: float = 0.95) -> float:
    """Value at Risk: loss exceeded with probability 1−alpha (positive)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return -_quantile(pnl, 1.0 - alpha)


def cvar(pnl: list[float], alpha: float = 0.95) -> float:
    """Conditional VaR (expected shortfall): mean loss beyond VaR (positive)."""
    threshold = _quantile(pnl, 1.0 - alpha)
    tail = [x for x in pnl if x <= threshold]
    return -sum(tail) / len(tail)


def prob_profit(pnl: list[float]) -> float:
    return sum(1 for x in pnl if x > 0) / len(pnl)


def terminal_stats(paths: list[list[float]], s0: float) -> dict:
    """Distribution of terminal P&L (vs s0) across paths."""
    pnl = [p[-1] - s0 for p in paths]
    n = len(pnl)
    mean = sum(pnl) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnl) / max(n - 1, 1))
    return {
        "mean_pnl": mean,
        "std_pnl": sd,
        "median_pnl": _quantile(pnl, 0.5),
        "p05_pnl": _quantile(pnl, 0.05),
        "p95_pnl": _quantile(pnl, 0.95),
        "prob_profit": prob_profit(pnl),
        "var_95": var(pnl),
        "cvar_95": cvar(pnl),
    }


def max_drawdown(path: list[float]) -> float:
    """Worst peak-to-trough fall as a fraction of the peak."""
    peak, worst = path[0], 0.0
    for x in path:
        peak = max(peak, x)
        worst = max(worst, (peak - x) / peak if peak > 0 else 0.0)
    return worst


def drawdown_stats(paths: list[list[float]]) -> dict:
    """Distribution of per-path maximum drawdowns."""
    dds = [max_drawdown(p) for p in paths]
    return {
        "mean_max_dd": sum(dds) / len(dds),
        "median_max_dd": _quantile(dds, 0.5),
        "p95_max_dd": _quantile(dds, 0.95),
        "worst_max_dd": max(dds),
    }


def prob_hit(paths: list[list[float]], level: float,
             direction: str = "above") -> float:
    """Fraction of paths that ever touch ``level`` (barrier analysis)."""
    if direction == "above":
        hit = [any(x >= level for x in p) for p in paths]
    elif direction == "below":
        hit = [any(x <= level for x in p) for p in paths]
    else:
        raise ValueError("direction must be 'above' or 'below'")
    return sum(hit) / len(hit)
