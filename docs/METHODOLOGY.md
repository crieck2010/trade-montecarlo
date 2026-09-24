# Methodology — trade-montecarlo

## Models

**Geometric Brownian motion.** dS/S = μ·dt + σ·dW, discretized exactly:
S_{t+dt} = S_t·exp((μ − σ²/2)dt + σ√dt·Z). No discretization bias.

**Merton jump-diffusion.** Between jumps the price follows GBM; jumps
arrive as a Poisson process with intensity λ and multiply the price by a
lognormal factor. The drift carries the compensator −λκ with
κ = E[jump multiplier − 1] = exp(jump_mean + jump_vol²/2) − 1, so the
mean growth rate stays μ. Jumps add the fat tails GBM misses.

**GARCH(1,1).** σ²_t = ω + α·r²_{t−1} + β·σ²_{t−1}, r_t = σ_t·z_t, with
z_t standard normal or Student-t. Captures volatility clustering:
large moves beget large moves. Parameters are **per simulation step** —
calibrate ω, α, β to your step size (defaults suit daily steps).
Requires α + β < 1 (stationarity); variance starts at the unconditional
level ω/(1−α−β).

**Ornstein-Uhlenbeck.** dx = θ(μ−x)dt + σ√dt·dW. Mean-reverting; the
natural scenario engine for pairs-trading spreads, with
θ ≈ ln 2 / half_life linking it to trade-pairs' estimates.

**Correlated multi-asset GBM.** Correlated normals via Cholesky of the
correlation matrix, applied per step per path.

## Variance reduction

Antithetic variates (default on): each path is paired with its mirror
image. For monotone payoffs this roughly halves the variance — the same
accuracy with ~half the paths.

## Option pricing

Risk-neutral valuation: simulate under μ = r, discount payoffs at r.
European prices are validated against Black-Scholes closed form in the
test suite (independent implementation). Asian (arithmetic average) and
up-and-out barrier options have no closed form — that is precisely where
Monte Carlo earns its keep. Every pricer reports a standard error:
price ± 1.96·SE is the honest 95% interval.

## Analytics conventions

P&L is positive for profit. **VaR and CVaR are reported as positive loss
amounts**: VaR₉₅ is the loss exceeded with 5% probability; CVaR₉₅
(expected shortfall) is the mean loss given VaR is breached.

## Block bootstrap

When you distrust parametric models, resample history: cut the T×N
return matrix into overlapping circular blocks and concatenate random
blocks into new paths. Within-block autocorrelation and cross-asset
correlation survive; no distributional assumptions enter. This is the
honest way to stress a backtest ("was this Sharpe luck?").

## Honest limitations

- **Model risk dominates.** GBM understates tails; jump-diffusion and
  GARCH patch this but are still stylized. The bootstrap avoids
  distributional assumptions but assumes the future resembles the past.
- **GARCH parameters are per-step** — mixing step sizes without
  recalibrating silently rescales volatility.
- **Correlated simulation is GBM-only** in v0.1.0; multi-asset
  jump/GARCH correlation is future work.
- **Pure Python is slow** next to NumPy: 50k×252 option pricings take
  under a minute here; a vectorized backend would be 50–100× faster.
  The API is designed to accept one without changing callers.
- Monte Carlo error shrinks as 1/√n — halving the standard error costs
  4× the paths. Always read the reported SE before trusting digits.
- Barrier options use discrete (per-step) monitoring; continuous
  monitoring would knock out slightly more often.
