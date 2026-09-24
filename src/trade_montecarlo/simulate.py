"""High-level simulation runner: one config in, paths + summary out."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import analytics
from .processes import (
    simulate_correlated_gbm,
    simulate_garch,
    simulate_gbm,
    simulate_jump,
    simulate_ou,
)
from .rng import RandomStream


@dataclass
class SimConfig:
    model: str = "gbm"  # gbm | jump | garch | ou
    n_paths: int = 10_000
    n_steps: int = 252
    T: float = 1.0  # years; dt = T / n_steps
    s0: float = 100.0
    mu: float = 0.05  # GBM/jump drift (annualized)
    sigma: float = 0.20  # vol (annualized; GARCH ignores this)
    seed: int = 7
    antithetic: bool = True
    # jump-diffusion
    lam: float = 0.5
    jump_mean: float = -0.05
    jump_vol: float = 0.10
    # GARCH(1,1)
    omega: float = 2e-6
    alpha: float = 0.08
    beta: float = 0.90
    innov: str = "normal"
    df: float = 6.0
    # OU
    theta: float = 2.0
    long_mean: float = 0.0


@dataclass
class SimulationResult:
    config: SimConfig
    paths: list[list[float]] = field(default_factory=list)

    def summary(self) -> dict:
        return {"model": self.config.model,
                "n_paths": len(self.paths),
                "n_steps": self.config.n_steps,
                **analytics.terminal_stats(self.paths, self.config.s0)}

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "paths": self.paths}


def run_simulation(config: SimConfig | None = None) -> SimulationResult:
    """Run one Monte Carlo session.  Deterministic for a fixed config."""
    cfg = config or SimConfig()
    stream = RandomStream(cfg.seed)
    dt = cfg.T / cfg.n_steps
    kw = dict(antithetic=cfg.antithetic)
    if cfg.model == "gbm":
        paths = simulate_gbm(stream, cfg.s0, cfg.n_paths, cfg.n_steps, dt,
                             cfg.mu, cfg.sigma, **kw)
    elif cfg.model == "jump":
        paths = simulate_jump(stream, cfg.s0, cfg.n_paths, cfg.n_steps, dt,
                              cfg.mu, cfg.sigma, cfg.lam, cfg.jump_mean,
                              cfg.jump_vol, **kw)
    elif cfg.model == "garch":
        # GARCH(1,1) parameters are per simulation step: dt is 1 step.
        paths = simulate_garch(stream, cfg.s0, cfg.n_paths, cfg.n_steps, 1.0,
                               cfg.omega, cfg.alpha, cfg.beta,
                               cfg.innov, cfg.df, **kw)
    elif cfg.model == "ou":
        paths = simulate_ou(stream, cfg.s0, cfg.n_paths, cfg.n_steps, dt,
                            cfg.theta, cfg.long_mean, cfg.sigma, **kw)
    else:
        raise ValueError(f"unknown model {cfg.model!r}")
    return SimulationResult(config=cfg, paths=paths)


def run_correlated(s0: list[float], mu: list[float], sigma: list[float],
                   corr: list[list[float]], n_paths: int = 10_000,
                   n_steps: int = 252, T: float = 1.0, seed: int = 7,
                   antithetic: bool = True) -> list[list[list[float]]]:
    """Correlated multi-asset GBM: [asset][path][step]."""
    return simulate_correlated_gbm(RandomStream(seed), s0, mu, sigma, corr,
                                   n_paths, n_steps, T / n_steps, antithetic)
