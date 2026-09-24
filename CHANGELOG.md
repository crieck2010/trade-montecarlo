# Changelog

## v0.1.0 — 2026-09-24

Initial release.

- `rng`: `RandomStream` with seeded uniform/normal (Box-Muller)/
  Student-t (Marsaglia-Tsang gamma)/lognormal samplers; `spawn()` for
  independent reproducible parallel streams.
- `processes`: GBM, Merton jump-diffusion (compensated drift; exactly
  GBM when λ=0), GARCH(1,1) with normal/t innovations, Ornstein-Uhlenbeck,
  correlated multi-asset GBM via Cholesky. Antithetic variates in every model.
- `simulate`: `SimConfig` + `run_simulation()` session runner,
  `SimulationResult` with summary/`to_dict()`, `run_correlated()`.
- `resample`: circular block bootstrap of historical return matrices.
- `options`: Monte Carlo pricers for European, arithmetic Asian, and
  up-and-out barrier options — each returns price + standard error.
- `analytics`: VaR/CVaR (losses as positives), prob-profit, terminal
  stats, max-drawdown distribution, barrier hit probability.
- `adapters`: `to_agent_scenarios` (trade-agents), `var_for_risk` +
  `portfolio_var` (trade-risk), `bootstrap_for_backtest` (trade-backtest),
  `ou_spread_config` (trade-pairs half-life → OU).
- CLI: `simulate`, `option`, `scenarios`, `license`, `update-check`.
- 31 tests, including Black-Scholes cross-check (independent
  implementation in the test file), put-call parity, and structural
  jump(λ=0)≡GBM equality.
