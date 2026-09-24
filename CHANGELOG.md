# Changelog

## v0.2.0 — 2026-09-24

Jump-diffusion upgrade + exotic pricer suite.

- `processes`: new `simulate_kou` — Kou (2002) double-exponential
  jump-diffusion with asymmetric up/down jumps (equity skew), drift
  compensator −λκ, `lam=0` *exactly* GBM (tested); antithetic pairing
  mirrors diffusion shocks, shares jump times/sizes within a pair.
  `SimConfig` gains `model="kou"` + `p_up`/`eta1`/`eta2`.
- New `calibrate` module: `bipower_vol` (jump-robust diffusive σ via
  Barndorff-Nielsen & Shephard bipower variation) and
  `fit_merton_jumps` (truncation fit: |r| > 4σ√dt flags → λ/jump_mean/
  jump_vol; <3 flags → pure diffusion). Scenario-grade, documented
  honestly — λ is a lower bound since small jumps hide.
- `options`: new `digital` (cash-or-nothing), `asset_or_nothing`,
  `lookback` (floating/fixed strike), generalized `barrier`
  (up/down × in/out; `barrier_up_out_call` kept as a back-compat
  wrapper), `geometric_asian`, and `american_lsm` (Longstaff-Schwartz
  least-squares Monte Carlo, stdlib 3×3 solver). Every pricer now takes
  `paths=` (price on any path set) or `model="jump"/"kou"` + `model_kw`
  (risk-neutralized drift under jump-diffusion), and reports `model`.
- CLI: `simulate --model kou` (+ `--p-up/--eta1/--eta2`); `option
  --type` gains digital/asset-or-nothing/lookback/american/barrier/
  geometric-asian, `--model gbm|jump|kou`, `--barrier-type`,
  `--strike`, `--cash`; new `calibrate` subcommand (`--returns` /
  `--file`).
- Docs: new `docs/EXOTICS.md` (what-you-learn / why-it-matters / the
  maths for every addition + honest limitations); METHODOLOGY,
  ARCHITECTURE, README updated; new `examples/exotic_example.py`.
- 56 tests (31 → 56): closed-form cross-checks (digital N(d₂),
  asset-or-nothing N(d₁), geometric Asian Kemna–Vorst, American LSM vs
  independent CRR binomial), barrier in+out parity to 1e-9 on identical
  paths, Kou structural/drift/skew tests, calibration recovery on
  synthetic jumps, CLI coverage. Backwards compatible.

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
