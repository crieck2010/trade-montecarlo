"""Monte Carlo simulation for the trade-suite.

Seeded price-process models (GBM, Merton and Kou jump-diffusion,
GARCH(1,1), Ornstein-Uhlenbeck), correlated multi-asset simulation,
circular block bootstrap, jump calibration, Monte Carlo option pricers
(European, Asian, barrier, digital, lookback, American LSM), and
distribution analytics (VaR/CVaR, drawdowns, barrier hits).
Stdlib-only and deterministic.
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import analytics, calibrate, options, processes
from .adapters import (
    bootstrap_for_backtest,
    ou_spread_config,
    portfolio_var,
    spawn_streams,
    to_agent_scenarios,
    var_for_risk,
)
from .calibrate import bipower_vol, fit_merton_jumps
from .resample import block_bootstrap
from .rng import RandomStream
from .simulate import SimConfig, SimulationResult, run_correlated, run_simulation

__all__ = [
    "RandomStream",
    "SimConfig",
    "SimulationResult",
    "analytics",
    "bipower_vol",
    "block_bootstrap",
    "bootstrap_for_backtest",
    "calibrate",
    "fit_merton_jumps",
    "options",
    "ou_spread_config",
    "portfolio_var",
    "processes",
    "run_correlated",
    "run_simulation",
    "spawn_streams",
    "to_agent_scenarios",
    "var_for_risk",
    "__version__",
]
