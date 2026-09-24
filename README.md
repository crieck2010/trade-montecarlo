# trade-montecarlo

Monte Carlo simulation for the trade-suite: seeded price-process models (GBM, Merton and Kou jump-diffusion, GARCH(1,1), Ornstein-Uhlenbeck), jump calibration, correlated multi-asset paths, circular block bootstrap, Monte Carlo exotic option pricers (European, Asian, barrier, digital, lookback, American LSM), and distribution analytics (VaR/CVaR, drawdowns, barrier hits).

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
res = options.barrier(100, 100, 130, T=1.0, r=0.03, sigma=0.20,
                      barrier_type="up-out")
print(res["price"], "±", res["std_error"])

# Price an American put under Kou jump-diffusion, calibrated to history
from trade_montecarlo import fit_merton_jumps
fit = fit_merton_jumps(my_log_returns)          # {'lam', 'jump_mean', ...}
res = options.american_lsm(100, 100, T=1.0, r=0.03, sigma=fit["sigma"],
                           model="jump",
                           model_kw={"lam": fit["lam"],
                                     "jump_mean": fit["jump_mean"],
                                     "jump_vol": fit["jump_vol"]})
```

## CLI

```bash
trade-montecarlo simulate --model kou --paths 20000 --lam 0.8 --p-up 0.35
trade-montecarlo simulate --model garch --format json
trade-montecarlo option --type barrier --barrier-type down-in --barrier 80
trade-montecarlo option --type american --kind put --model jump --lam 1.0
trade-montecarlo option --type lookback --strike floating
trade-montecarlo calibrate --file returns.txt        # fit jump params
trade-montecarlo scenarios --model ou --paths 5000   # agent-ready JSON
trade-montecarlo license
trade-montecarlo update-check
```

## What's inside

| Module | Contents |
|---|---|
| `rng` | `RandomStream`: seeded uniform/normal/Student-t/lognormal samplers, `spawn()` for independent parallel streams |
| `processes` | `simulate_gbm`, `simulate_jump` (Merton), `simulate_kou` (double-exponential jumps), `simulate_garch` (1,1), `simulate_ou`, `simulate_correlated_gbm` |
| `simulate` | `SimConfig` + `run_simulation()` session runner, `run_correlated()` |
| `resample` | Circular block bootstrap of historical returns (no model assumed) |
| `calibrate` | `bipower_vol` (jump-robust σ) + `fit_merton_jumps` (truncation fit → dict for `simulate_jump`) |
| `options` | MC pricers: `european`, `asian`, `geometric_asian`, `digital`, `asset_or_nothing`, `lookback`, `barrier` (up/down × in/out), `american_lsm` — each returns `(price, std_error)`; every pricer takes `paths=` or `model="jump"/"kou"` |
| `analytics` | VaR, CVaR, `prob_profit`, terminal stats, max-drawdown distribution, `prob_hit` |
| `adapters` | Bridges to `trade-agents`, `trade-risk`, `trade-backtest`, `trade-optimize`/`trade-pairs` |

### Model notes

- **GBM**: the workhorse. Exact discretization, no bias.
- **Jump-diffusion** (Merton): Poisson jumps with lognormal sizes; drift carries the compensator so E[S] still grows at μ. With `lam=0` it is *exactly* GBM (tested).
- **Kou jump-diffusion**: asymmetric double-exponential jumps — separate up/down tails, reproduces the equity skew. Compensator κ = p·η₁/(η₁−1) + (1−p)·η₂/(η₂+1) − 1; `lam=0` ≡ GBM (tested).
- **Jump calibration**: bipower variation isolates diffusive σ; truncation (|r| > 4σ√dt) flags jumps and reads (λ, jump_mean, jump_vol) off them. Scenario-grade, not trading-grade.
- **GARCH(1,1)**: volatility clustering with normal or Student-t innovations; parameters are **per simulation step**.
- **OU**: mean-reverting spreads — pair with `trade-pairs` half-life via `adapters.ou_spread_config(half_life, spread_vol)` (θ = ln 2 / half-life).

### The maths

- **Kou jumps**: log-jump sizes are double-exponential, f_Y(y) = p·η₁e^{−η₁y}1_{y≥0} + (1−p)·η₂e^{η₂y}1_{y<0}. Drift −λκ keeps E[S_T] = s₀e^{μT}; skew comes free from η₁ ≠ η₂.
- **Bipower variation**: (π/2)·mean(|r_t·r_{t−1}|) → σ²dt — jumps wash out of the adjacent product, so σ is estimated without jump contamination.
- **Digitals**: call = cash·e^{−rT}·N(d₂), the discounted risk-neutral probability of finishing in the money; asset-or-nothing call = s₀·N(d₁).
- **Geometric Asian** (Kemna–Vorst): the geometric average of lognormals is lognormal — E[ln G] = ln s₀ + (r−½σ²)T(n+1)/2n, Var = σ²T(n+1)(2n+1)/6n² — then Black-Scholes on the forward.
- **American LSM**: backward induction; at each date regress discounted future cashflows on {1, S, S²} over in-the-money paths — the fitted value is E[future | S_t], the continuation value. Exercise iff payoff beats it.
- **Barrier parity**: up-in + up-out = European, path by path (asserted to 1e-9 on identical paths in the test suite).

Full derivations, why each matters, and the honest limitations live in `docs/EXOTICS.md`.

### Interop

- `to_agent_scenarios()` → `trade-agents`: scenario summary JSON (P(profit), VaR/CVaR, drawdowns), no raw paths.
- `var_for_risk()` / `portfolio_var()` → `trade-risk`: simulated VaR/CVaR blocks for limit checks; multi-asset correlated GBM → portfolio P&L.
- `bootstrap_for_backtest()` → `trade-backtest`: resampled return scenarios to ask "was this backtest luck?"
- `ou_spread_config()` → `trade-pairs`: spread scenarios from an estimated half-life.
- All sibling imports are lazy — this package installs and imports standalone.

See `examples/montecarlo_example.py` for a full walkthrough, `docs/ARCHITECTURE.md` for design, and `docs/METHODOLOGY.md` for the math and the honest limitations.

## License

MIT. See `LICENSE`.
