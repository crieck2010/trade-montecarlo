"""End-to-end demo: simulate, price, bootstrap, and summarize for an agent."""

from trade_montecarlo import SimConfig, options, run_simulation
from trade_montecarlo import analytics
from trade_montecarlo.adapters import (
    bootstrap_for_backtest,
    ou_spread_config,
    portfolio_var,
    to_agent_scenarios,
)

# 1. GBM session: distribution of 1-year P&L on $100
result = run_simulation(SimConfig(model="gbm", n_paths=10_000, seed=7))
s = result.summary()
print(f"GBM: P(profit)={s['prob_profit']:.1%}  "
      f"VaR95={s['var_95']:.2f}  CVaR95={s['cvar_95']:.2f}")

# 2. Jump-diffusion: fatter tails, worse VaR
j = run_simulation(SimConfig(model="jump", n_paths=10_000, lam=0.8, seed=7))
print(f"Jump: VaR95={j.summary()['var_95']:.2f}  "
      f"CVaR95={j.summary()['cvar_95']:.2f}")

# 3. Drawdowns and barrier hits
dd = analytics.drawdown_stats(result.paths)
print(f"Max DD: mean {dd['mean_max_dd']:.1%}, p95 {dd['p95_max_dd']:.1%}")
print(f"P(touch 130) = {analytics.prob_hit(result.paths, 130, 'above'):.1%}")

# 4. Price an up-and-out call (no closed form — MC territory)
b = options.barrier_up_out_call(100, 100, 130, T=1.0, r=0.03, sigma=0.20)
print(f"Up-and-out call: {b['price']:.4f} ± {b['std_error']:.4f}")

# 5. Portfolio VaR for trade-risk: 60/40 correlated GBM
pv = portfolio_var([0.6, 0.4], [100.0, 50.0], [0.05, 0.05], [0.2, 0.3],
                   [[1.0, 0.3], [0.3, 1.0]], capital=100_000.0,
                   n_paths=5000, n_steps=63, seed=7)
print(f"Portfolio VaR95={pv['var']:,.0f}  CVaR95={pv['cvar']:,.0f}")

# 6. Pairs-trading spread scenarios from a half-life estimate
cfg = ou_spread_config(half_life=21.0, spread_vol=0.02)
print(f"OU spread config: theta={cfg.theta:.4f}")

# 7. Block-bootstrap a return history for backtest stress
hist = [[0.001, -0.0005]] * 252
bb = bootstrap_for_backtest(hist, n_paths=50, block_len=20, seed=7)
print(f"Bootstrap: {bb['n_paths']} scenarios of {len(bb['scenarios'][0])} bars")

# 8. Agent-ready scenario summary
sc = to_agent_scenarios(SimConfig(model="gbm", n_paths=2000, seed=7))
print(f"Agent summary keys: {sorted(sc)}")
