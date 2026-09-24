# Architecture — trade-montecarlo

## Layering

```
CLI (cli.py) ──thin──▶ session API (simulate.py)
                                │
session API ──▶ processes.py (model zoo) ──▶ rng.py (RandomStream)
            ├─▶ resample.py (block bootstrap) ─▶ rng.py
            ├─▶ options.py (MC pricers) ──▶ processes (gbm/jump/kou) or caller paths
            ├─▶ calibrate.py (bipower vol + truncation jump fit; pure functions)
            ├─▶ analytics.py (pure functions over paths / P&L lists)
            └─▶ adapters.py (sibling bridges; imports lazy)
licensing.py stands alone (license-key + update-check hooks).
```

No UI-framework imports anywhere. `__init__.py` re-exports the session
API; submodules stay importable on their own.

## Key decisions

**Determinism as a feature.** Every stochastic draw flows through
`RandomStream`, seeded once per session. Tests assert exact path equality
across runs (`test_gbm_deterministic`, `test_jump_with_zero_intensity_is_gbm`).
Reproducibility is what makes MC results debuggable and backtestable.

**Antithetic variates at the model level.** Each model pairs every path
with its mirror (negated shocks). Implemented inside the model loops —
not at the stream level — because Box-Muller's spare cache makes
stream-level negation incorrect. Both paths are unbiased; their average
has lower variance. Verified by `test_antithetic_deterministic_and_unbiased`.

**Jump-diffusion shares the GBM code path.** With `lam=0`, jump draws are
skipped entirely, so the normal sequence aligns and the output is
*bit-identical* to GBM. This is a structural guarantee, not a statistical
one, and it's tested as such.

**Plain-data adapters.** Adapters exchange dicts/lists, never engine
objects. `to_agent_scenarios` hands agents a JSON summary (distributions,
not paths); `var_for_risk` hands trade-risk a VaR/CVaR block;
`bootstrap_for_backtest` hands trade-backtest scenario matrices;
`ou_spread_config` turns a trade-pairs half-life into an OU config
without either module importing the other.

**Pricing validation without coupling.** `options.european` is
cross-checked against an independent Black-Scholes implementation that
lives *in the test file*, so the package never depends on
trade-data-options at runtime while still proving the pricer is correct.
The same pattern covers digitals (N(d₂)), asset-or-nothing (N(d₁)),
geometric Asians (Kemna–Vorst), American LSM (independent CRR binomial
tree in the test file), and barrier in+out parity on identical paths.

**Pricers are process-agnostic.** Every pricer accepts `paths=` — any
pre-simulated path set — or `model="jump"/"kou"` to simulate under
jump-diffusion internally. Pricing logic never assumes GBM; the model
is a parameter, not a coupling.

## Scaling

- Pure-Python loops: ~2.5M price steps (10k paths × 252 steps) run in a
  few seconds; 50k-path option pricings in under a minute. Fine for
  research and paper trading.
- `RandomStream.spawn(n)` derives independent child streams for
  multiprocessing workers — each worker simulates a shard with a
  reproducible seed. No shared state, no locks.
- Paths are plain `list[list[float]]`: trivially serializable (JSON in
  `SimulationResult.to_dict()`), shardable, and streamable. For very
  large runs, process paths in chunks rather than holding all in memory.
- The natural growth path is a NumPy vectorized backend behind the same
  `SimConfig`/`SimulationResult` API; callers would not change.

## Conventions (suite-wide)

MIT license · semver + CHANGELOG · `license` / `update-check` CLI hooks ·
`master` branch · stdlib-only runtime · lazy sibling imports.
