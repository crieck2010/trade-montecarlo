"""Interop adapters: plain-data bridges to sibling suite modules.

Sibling imports are lazy (inside functions) so trade-montecarlo stays
importable and testable on its own.
"""

from __future__ import annotations

from . import analytics
from .resample import block_bootstrap
from .rng import RandomStream
from .simulate import SimConfig, run_correlated, run_simulation


# --------------------------------------------------------------- trade-agents

def to_agent_scenarios(config: SimConfig | None = None) -> dict:
    """Scenario summary for researcher agents: the distribution, not paths.

    Gives an agent P(profit), VaR/CVaR, and drawdown stats as JSON — no
    engine, no path arrays, nothing to misuse.
    """
    result = run_simulation(config)
    return {
        "source": "trade-montecarlo",
        "model": result.config.model,
        "n_paths": len(result.paths),
        "terminal": result.summary(),
        "drawdowns": analytics.drawdown_stats(result.paths),
    }


# ----------------------------------------------------------------- trade-risk

def var_for_risk(pnl: list[float], alpha: float = 0.95,
                 capital: float = 1.0) -> dict:
    """VaR/CVaR block for trade-risk limit checks (losses as positives)."""
    return {
        "source": "trade-montecarlo",
        "alpha": alpha,
        "var": analytics.var(pnl, alpha),
        "cvar": analytics.cvar(pnl, alpha),
        "var_pct_of_capital": analytics.var(pnl, alpha) / capital if capital else 0.0,
        "prob_profit": analytics.prob_profit(pnl),
    }


def portfolio_var(weights: list[float], s0: list[float], mu: list[float],
                  sigma: list[float], corr: list[list[float]],
                  capital: float = 1_000_000.0, n_paths: int = 10_000,
                  n_steps: int = 252, T: float = 1.0, seed: int = 7,
                  alpha: float = 0.95) -> dict:
    """Simulated portfolio VaR: correlated GBM → portfolio P&L → VaR/CVaR.

    Complements trade-optimize's analytic vol with a full distribution —
    skew, fat tails (via the t-innovation GARCH single-asset path), and
    drawdowns included.
    """
    paths = run_correlated(s0, mu, sigma, corr, n_paths, n_steps, T, seed)
    shares = [w * capital / p for w, p in zip(weights, s0)]
    pnl = []
    for j in range(n_paths):
        end_val = sum(shares[a] * paths[a][j][-1] for a in range(len(s0)))
        pnl.append(end_val - capital)
    return {"weights": weights, **var_for_risk(pnl, alpha, capital)}


# ------------------------------------------------------------- trade-backtest

def bootstrap_for_backtest(returns: list[list[float]], n_paths: int = 200,
                           block_len: int = 20, seed: int = 7) -> dict:
    """Resampled return scenarios to stress a backtested strategy.

    Feed each scenario matrix through the strategy to get a distribution
    of Sharpe/max-drawdown — "was this backtest luck?" as a number.
    """
    scenarios = block_bootstrap(returns, n_paths, block_len, seed)
    return {
        "source": "trade-montecarlo",
        "method": "circular-block-bootstrap",
        "n_paths": n_paths,
        "block_len": block_len,
        "scenarios": scenarios,  # each: T×N resampled returns
    }


# -------------------------------------------------------------- trade-optimize

def ou_spread_config(half_life: float, spread_vol: float, dt: float = 1 / 252,
                     seed: int = 7) -> SimConfig:
    """OU scenario config from a trade-pairs spread: θ = ln2 / half_life.

    Lets the pairs engine's estimated half-life drive spread scenarios
    without either module importing the other.
    """
    import math
    return SimConfig(model="ou", s0=0.0, n_paths=5_000, n_steps=252, T=1.0,
                     sigma=spread_vol, theta=math.log(2) / half_life,
                     long_mean=0.0, seed=seed)


# ------------------------------------------------------------------ utilities

def spawn_streams(seed: int, n: int) -> list[RandomStream]:
    """Independent reproducible streams for parallel workers."""
    return RandomStream(seed).spawn(n)
