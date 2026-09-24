"""Exotic pricing under jump-diffusion — trade-montecarlo v0.2.0.

Calibrate Merton jumps to (synthetic) history, then price a small book
of exotics under the fitted jump model vs plain GBM.  Stdlib-only.
"""

from __future__ import annotations

import math

from trade_montecarlo import (
    RandomStream,
    fit_merton_jumps,
    options,
    processes,
)

s0, K, T, r = 100.0, 100.0, 1.0, 0.03

# --- 1. synthetic "history": Merton jumps, 10 years of daily returns ---
hist = processes.simulate_jump(RandomStream(99), s0, 1, 2_520, 1 / 252,
                               0.05, 0.18, 3.0, -0.08, 0.12,
                               antithetic=False)[0]
rets = [math.log(hist[i + 1] / hist[i]) for i in range(2_520)]

# --- 2. calibrate ---
fit = fit_merton_jumps(rets)
print(f"calibrated: lam={fit['lam']:.2f}/yr  jump_mean={fit['jump_mean']:+.3f} "
      f"jump_vol={fit['jump_vol']:.3f}  sigma={fit['sigma']:.3f} "
      f"({fit['n_jumps']} jumps flagged, "
      f"jump share of var {fit['jump_share_var']:.0%})")

mkw = {"lam": fit["lam"], "jump_mean": fit["jump_mean"],
       "jump_vol": fit["jump_vol"]}
N = 30_000

# --- 3. price the book under GBM vs fitted jumps ---
book = [
    ("european put ", lambda md, kw: options.european(s0, K, T, r, fit["sigma"],
          kind="put", n_paths=N, seed=7, model=md, model_kw=kw)),
    ("american put ", lambda md, kw: options.american_lsm(s0, K, T, r, fit["sigma"],
          kind="put", n_paths=N, n_steps=50, seed=7, model=md, model_kw=kw)),
    ("digital put  ", lambda md, kw: options.digital(s0, K, T, r, fit["sigma"],
          kind="put", n_paths=N, seed=7, model=md, model_kw=kw)),
    ("down-in put  ", lambda md, kw: options.barrier(s0, K, 80.0, T, r, fit["sigma"],
          kind="put", barrier_type="down-in", n_paths=N, seed=7,
          model=md, model_kw=kw)),
    ("lookback put ", lambda md, kw: options.lookback(s0, T, r, fit["sigma"],
          kind="put", n_paths=N, seed=7, model=md, model_kw=kw)),
]
print(f"\n{'payoff':<14}{'GBM':>10}{'jumps':>10}   (n={N})")
for name, price in book:
    gbm = price("gbm", None)
    jmp = price("jump", mkw)
    print(f"{name}{gbm['price']:>10.4f}{jmp['price']:>10.4f}   "
          f"± {jmp['std_error']:.4f}")

print("\nJumps raise every downside payoff — that is the skew you calibrated.")
