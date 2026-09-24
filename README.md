# trade-montecarlo

Monte Carlo simulation for the trade-suite: seeded price-process models (GBM, Merton jump-diffusion, GARCH(1,1), Ornstein-Uhlenbeck), correlated multi-asset paths, circular block bootstrap, Monte Carlo option pricers, and distribution analytics (VaR/CVaR, drawdowns, barrier hits).

Stdlib-only. Deterministic: same seed → identical paths, every run, every machine.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-montecarlo.git
```

## Quick start

```python
from trade_montecarlo import SimConfig, run_simulation, options

# 10k GBM paths, antithetic variates on by default
result = run_simulation(SimConfig(model="gbm", n_paths=10_000, seed=7))
print(result.summary())
# {'model': 'gbm', 'n_paths': 10000, 'mean_pnl': ..., 'var_95': ..., ...}

# Price an up-and-out call — no closed form exists, MC handles it
res = options.barrier_up_out_call(100, 100, 130, T=1.0, r=0.03, sigma=0.20)
print(res["price"], "±", res["std_error"])
```

## CLI

```bash
trade-montecarlo simulate --model jump --paths 20000 --lam 0.8
trade-montecarlo simulate --model garch --format json
trade-montecarlo option --type barrier --s0 100 --K 100 --barrier 130
trade-montecarlo scenarios --model ou --paths 5000   # agent-ready JSON
trade-montecarlo license
trade-montecarlo update-check
```

## What's inside

| Module | Contents |
|---|---|
| `rng` | `RandomStream`: seeded uniform/normal/Student-t/lognormal samplers, `spawn()` for independent parallel streams |
| `processes` | `simulate_gbm`, `simulate_jump` (Merton), `simulate_garch` (1,1), `simulate_ou`, `simulate_correlated_gbm` |
| `simulate` | `SimConfig` + `run_simulation()` session runner, `run_correlated()` |
| `resample` | Circular block bootstrap of historical returns (no model assumed) |
| `options` | MC pricers: `european`, `asian`, `barrier_up_out_call` — each returns `(price, std_error)` |
| `analytics` | VaR, CVaR, `prob_profit`, terminal stats, max-drawdown distribution, `prob_hit` |
| `adapters` | Bridges to `trade-agents`, `trade-risk`, `trade-backtest`, `trade-optimize`/`trade-pairs` |

### Model notes

- **GBM**: the workhorse. Exact discretization, no bias.
- **Jump-diffusion** (Merton): Poisson jumps with lognormal sizes; drift carries the compensator so E[S] still grows at μ. With `lam=0` it is *exactly* GBM (tested).
- **GARCH(1,1)**: volatility clustering with normal or Student-t innovations; parameters are **per simulation step**.
- **OU**: mean-reverting spreads — pair with `trade-pairs` half-life via `adapters.ou_spread_config(half_life, spread_vol)` (θ = ln 2 / half-life).

### Interop

- `to_agent_scenarios()` → `trade-agents`: scenario summary JSON (P(profit), VaR/CVaR, drawdowns), no raw paths.
- `var_for_risk()` / `portfolio_var()` → `trade-risk`: simulated VaR/CVaR blocks for limit checks; multi-asset correlated GBM → portfolio P&L.
- `bootstrap_for_backtest()` → `trade-backtest`: resampled return scenarios to ask "was this backtest luck?"
- `ou_spread_config()` → `trade-pairs`: spread scenarios from an estimated half-life.
- All sibling imports are lazy — this package installs and imports standalone.

See `examples/montecarlo_example.py` for a full walkthrough, `docs/ARCHITECTURE.md` for design, and `docs/METHODOLOGY.md` for the math and the honest limitations.

## License

MIT. See `LICENSE`.
