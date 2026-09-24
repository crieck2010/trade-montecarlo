# Exotic options & jump-diffusion — trade-montecarlo v0.2.0

What v0.2.0 adds: the **Kou double-exponential jump-diffusion** process,
a **truncation-based jump calibrator**, and a Monte Carlo **exotic
pricer suite** (digitals, asset-or-nothing, lookbacks, all four single
barriers, geometric Asians, American via Longstaff-Schwartz). Every
pricer accepts `paths=` (price on *any* simulated paths) or
`model="jump"/"kou"` (price the same payoff under jump-diffusion).

---

## 1. Kou double-exponential jump-diffusion

### What you learn
Whether your scenario tails are **asymmetric** — frequent small
up-jumps, rare violent down-jumps — instead of Merton's symmetric
lognormal jumps. Kou reproduces the equity skew: markets melt up and
crash down.

### Why it matters
Merton's jumps are symmetric in log-space; real equity jumps are not.
An option pricer or VaR engine fed symmetric jumps understates
downside-tail risk and misprices skew-sensitive payoffs (puts, down
barriers). Kou's two-sided exponential law has separate up/down
parameters, so the model can lean bearish.

### The maths
Between jumps the log-price follows Brownian motion; jumps arrive as a
Poisson(λ) process. Each jump's log-size Y is double-exponential:

> f_Y(y) = p·η₁·e^{−η₁y}·1_{y≥0} + (1−p)·η₂·e^{η₂y}·1_{y<0}

With probability p the jump is up, Exp(η₁); otherwise down, −Exp(η₂).
Mean up-jump = 1/η₁, mean down-jump = 1/η₂. The drift carries the
compensator −λκ with

> κ = E[e^Y − 1] = p·η₁/(η₁−1) + (1−p)·η₂/(η₂+1) − 1,  requiring η₁ > 1

so E[S_T] = s₀·e^{μT} exactly as in Merton. With λ = 0 the simulator
is *exactly* GBM (no jump draws consumed — tested). Antithetic pairing
mirrors the diffusion shocks; jump times and sizes are *shared* within
a pair (mirroring an asymmetric law would change its distribution) —
each path stays unbiased.

---

## 2. Jump calibration (bipower variation + truncation)

### What you learn
The jump parameters (λ, jump_mean, jump_vol) implied by a return
series — how often jumps arrive and how big they are — plus a
jump-robust estimate of the diffusive σ.

### Why it matters
Simulating jumps with guessed parameters is storytelling. Calibration
grounds the scenarios in history: feed it trade-data bars or
backtest residuals and the fitted dict drops straight into
`simulate_jump` / `SimConfig`. The bipower σ matters on its own —
plain realized vol overstates diffusive vol whenever jumps are
present, which quietly inflates every GBM scenario you run.

### The maths
**Bipower variation** (Barndorff-Nielsen & Shephard): for independent
Gaussian returns, E|r_t·r_{t−1}| = σ²dt·(2/π), so
(π/2)·mean(|r_t·r_{t−1}|) → σ²dt. Jumps are isolated events — they
inflate Σr_t² but wash out of the adjacent product — so this estimates
the *diffusive* σ even on jumpy data. On pure GBM it ≈ realized vol
(tested).

**Truncation:** flag any return with |r| > 4σ√dt as a jump (a Gaussian
exceeds 4σ with probability ≈ 6·10⁻⁵, so flags are almost all genuine).
Then λ = (#flags)/(n·dt), jump_mean = mean(flags),
jump_vol = std(flags). Fewer than 3 flags → λ = 0 (pure diffusion).
Small jumps hide inside the diffusion, so λ is a *lower bound* and
|jump_mean| is slightly overstated — documented, not hidden.

---

## 3. Digitals & asset-or-nothing

### What you learn
The price of a pure directional bet: cash-or-nothing pays a fixed
`cash` iff S_T crosses K; asset-or-nothing pays S_T itself iff it
crosses. No participation, no smoothness — just the event.

### Why it matters
Digitals are the building blocks: spreads of digitals approximate any
payoff, and their closed forms make them the sharpest test of a Monte
Carlo engine (discontinuous payoffs converge slowly — if your engine
prices these right, it prices everything right).

### The maths
Call payoff: cash·1_{S_T>K}. Under GBM, price = cash·e^{−rT}·N(d₂) —
the discounted risk-neutral probability of finishing in the money.
Asset-or-nothing call: S_T·1_{S_T>K}, price = s₀·N(d₁) (no discount on
the asset leg — it *is* the numeraire asset). Both are cross-checked
against closed form in the test suite. Digital put+call parity:
C + P = cash·e^{−rT} (tested on identical paths).

---

## 4. Lookbacks

### What you learn
The value of perfect hindsight: a floating-strike call pays
S_T − min(S) ("buy at the low"), a floating put pays max(S) − S_T
("sell at the high"). Fixed-strike variants pay max(S) − K / K − min(S).

### Why it matters
Lookbacks are the most expensive common exotic — they always finish in
the money (floating strike), so they isolate the price of *extremes*
rather than direction. If your desk ever prices guaranteed-minimum
products, this is the engine.

### The maths
No closed form under jumps (and none worth using under GBM for the
floating strike in discrete monitoring) — the payoff depends on the
whole path's running min/max, so Monte Carlo is the natural pricer.
Note the floating call always pays ≥ 0, hence costs more than any
European with K ≥ min; the test suite checks positivity and the
fixed-strike K requirement rather than a formula.

---

## 5. Barrier options (up/down × in/out)

### What you learn
How much a knock-out *rebate of risk* is worth: an up-and-out call dies
if S ever touches B, so it costs less than the European — you are
explicitly selling the "melt-up" paths. Knock-ins are the mirror: they
only come alive on a touch.

### Why it matters
Barriers are the cheapest way to express "I want upside, but not if it
moons first" — popular in FX and structured products. In+out parity
(up-in + up-out = European, path by path) is both a pricing identity
and the sharpest test: the suite prices both on the *same* paths and
asserts the sum equals the European to 1e-9.

### The maths
Monitored discretely on the simulated path: touched = (max ≥ B) for
up, (min ≤ B) for down; knock-out pays only if never touched, knock-in
only if touched. Under GBM with continuous monitoring closed forms
exist (Reiner–Rubinstein), but discrete monitoring is what actually
trades — Monte Carlo prices what trades.

---

## 6. Geometric Asian

### What you learn
The price of averaging: payoff on the *geometric* average of fixings
instead of the terminal price. Averaging kills volatility, so Asians
are cheaper than Europeans — popular in commodity and FX hedging where
the exposure itself is an average.

### Why it matters
Two roles: a product in its own right, and a **control variate** for
the arithmetic Asian (price the geometric by closed form, use its
simulation error to correct the arithmetic — same paths, correlated
errors).

### The maths
The geometric average of lognormals is lognormal (Kemna–Vorst). With n
fixings at t_i = i·T/n: E[ln G] = ln s₀ + (r−½σ²)·T(n+1)/(2n),
Var(ln G) = σ²T(n+1)(2n+1)/(6n²). Then it is Black-Scholes on the
forward F = exp(E[ln G] + Var/2). Cross-checked against this closed
form in the test suite.

---

## 7. American options (Longstaff–Schwartz)

### What you learn
The value of early exercise: an American put is worth more than its
European twin because you can exercise into a crash instead of waiting.

### Why it matters
Every listed US equity option is American. LSM is the standard Monte
Carlo method — it turns the optimal-stopping problem into a sequence
of regressions, and it works on *any* path set, including
jump-diffusion (where trees and PDEs struggle).

### The maths
Backward induction from expiry: at each exercise date, take the
in-the-money paths and regress their discounted future cashflows on
{1, S, S²} (ordinary least squares via normal equations, 3×3 Gaussian
elimination — stdlib only). The fitted value is the **continuation
value**; exercise iff immediate payoff > continuation. The regression
estimates E[discounted future | S_t] — the conditional expectation
that defines the optimal stopping rule. Cross-checked against an
independent Cox-Ross-Rubinstein binomial tree in the tests; American
put ≥ European put and American call (no dividends) ≈ European call
are asserted as bounds.

---

## Pricing under jump-diffusion

Pass `model="jump"` / `model="kou"` (plus `model_kw` with the jump
parameters) to any pricer, or `fit_merton_jumps` your history first and
feed the result in. The drift is set to r with the jump compensator
kept, so E[S_T] = s₀·e^{rT} — the standard risk-neutralized-drift
approach. Caveat, stated plainly: true risk-neutral jump pricing also
transforms the jump intensity/size law (Esscher transform); we keep the
physical jump law and only neutralize the drift. For scenario pricing
and relative-value work this is the market-standard shortcut; for
trading listed options against it, know what you skipped.

## Honest limitations

- **Kou/Merton are still stylized.** Real jumps cluster and have
  intraday structure; these models don't.
- **Calibration is scenario-grade, not trading-grade.** Truncation
  misses small jumps; 10 years of daily data hold only ~30 jumps at
  λ = 3/yr — the 4th cumulant is noise-dominated, which is why we
  don't moment-match.
- **LSM is biased low** (finite basis, finite paths) — the true
  American value is ≥ the LSM estimate. More paths, tighter SE.
- **Discrete monitoring** throughout: barrier/lookback/Asian fixings
  are the simulated steps. Finer steps → closer to continuous, slower
  to run.
