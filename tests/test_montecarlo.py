"""Tests for trade-montecarlo."""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import pytest

from trade_montecarlo import (
    RandomStream,
    SimConfig,
    block_bootstrap,
    run_correlated,
    run_simulation,
)
from trade_montecarlo import analytics, options, processes
from trade_montecarlo.adapters import (
    bootstrap_for_backtest,
    ou_spread_config,
    portfolio_var,
    to_agent_scenarios,
    var_for_risk,
)


def bs_call(s0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes closed form (independent reimplementation for tests)."""
    N = NormalDist().cdf
    d1 = (math.log(s0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * N(d1) - K * math.exp(-r * T) * N(d2)


# --------------------------------------------------------------------- rng

def test_stream_deterministic():
    a, b = RandomStream(7), RandomStream(7)
    assert [a.normal() for _ in range(100)] == [b.normal() for _ in range(100)]


def test_spawn_independent_but_reproducible():
    kids1 = RandomStream(7).spawn(4)
    kids2 = RandomStream(7).spawn(4)
    seqs1 = [[k.normal() for _ in range(10)] for k in kids1]
    seqs2 = [[k.normal() for _ in range(10)] for k in kids2]
    assert seqs1 == seqs2  # reproducible
    assert len({tuple(s) for s in seqs1}) == 4  # independent


def test_normal_moments():
    s = RandomStream(7)
    xs = [s.normal() for _ in range(20_000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.05
    assert abs(var - 1.0) < 0.05


def test_student_t_moments():
    s = RandomStream(11)
    xs = [s.student_t(6.0) for _ in range(20_000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.1
    assert abs(var - 6.0 / 4.0) < 0.2  # df/(df-2)


# --------------------------------------------------------------- processes

def test_gbm_deterministic():
    kw = dict(s0=100.0, n_paths=50, n_steps=63, dt=1 / 252, mu=0.05, sigma=0.2)
    p1 = processes.simulate_gbm(RandomStream(7), **kw)
    p2 = processes.simulate_gbm(RandomStream(7), **kw)
    assert p1 == p2


def test_gbm_expected_terminal_value():
    paths = processes.simulate_gbm(RandomStream(7), 100.0, 20_000, 252,
                                   1 / 252, 0.05, 0.2)
    mean_terminal = sum(p[-1] for p in paths) / len(paths)
    assert mean_terminal == pytest.approx(100 * math.exp(0.05), rel=0.03)


def test_gbm_lognormal_skew():
    paths = processes.simulate_gbm(RandomStream(7), 100.0, 20_000, 252,
                                   1 / 252, 0.05, 0.2)
    terms = sorted(p[-1] for p in paths)
    median = terms[len(terms) // 2]
    mean = sum(terms) / len(terms)
    assert median < mean  # lognormal is right-skewed


def test_jump_with_zero_intensity_is_gbm():
    kw = dict(s0=100.0, n_paths=100, n_steps=63, dt=1 / 252, mu=0.05, sigma=0.2)
    gbm = processes.simulate_gbm(RandomStream(7), **kw)
    jmp = processes.simulate_jump(RandomStream(7), lam=0.0, **kw)
    assert gbm == jmp


def test_jump_adds_kurtosis():
    gbm = processes.simulate_gbm(RandomStream(7), 100.0, 5000, 252, 1 / 252,
                                 0.05, 0.2)
    jmp = processes.simulate_jump(RandomStream(7), 100.0, 5000, 252, 1 / 252,
                                  0.05, 0.2, lam=1.0, jump_mean=-0.02,
                                  jump_vol=0.15)

    def kurt(paths):
        rs = [math.log(p[-1] / p[0]) for p in paths]
        m = sum(rs) / len(rs)
        v = sum((r - m) ** 2 for r in rs) / len(rs)
        return sum((r - m) ** 4 for r in rs) / len(rs) / v ** 2

    assert kurt(jmp) > kurt(gbm)


def test_ou_mean_reverts():
    paths = processes.simulate_ou(RandomStream(7), 0.0, 200, 5000, 1 / 252,
                                  theta=5.0, long_mean=3.0, sigma=0.5)
    tail = [x for p in paths for x in p[2500:]]
    assert abs(sum(tail) / len(tail) - 3.0) < 0.15


def test_garch_unconditional_variance():
    paths = processes.simulate_garch(RandomStream(7), 100.0, 4, 20_000,
                                     omega=2e-6, alpha=0.08, beta=0.90)
    r2 = []
    for p in paths:
        r2.extend(math.log(p[i + 1] / p[i]) ** 2 for i in range(len(p) - 1))
    assert abs(sum(r2) / len(r2) - 1e-4) < 3e-5  # ω/(1−α−β)


def test_correlated_gbm_hits_target_corr():
    paths = run_correlated([100.0, 100.0], [0.05, 0.05], [0.2, 0.3],
                           [[1.0, 0.6], [0.6, 1.0]],
                           n_paths=3000, n_steps=63, seed=7)
    r1, r2 = [], []
    for j in range(3000):
        for i in range(63):
            r1.append(math.log(paths[0][j][i + 1] / paths[0][j][i]))
            r2.append(math.log(paths[1][j][i + 1] / paths[1][j][i]))
    m1, m2 = sum(r1) / len(r1), sum(r2) / len(r2)
    cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2)) / len(r1)
    v1 = sum((a - m1) ** 2 for a in r1) / len(r1)
    v2 = sum((b - m2) ** 2 for b in r2) / len(r2)
    assert abs(cov / math.sqrt(v1 * v2) - 0.6) < 0.05


def test_antithetic_deterministic_and_unbiased():
    kw = dict(s0=100.0, n_paths=2000, n_steps=63, dt=1 / 252, mu=0.05, sigma=0.2)
    a1 = processes.simulate_gbm(RandomStream(7), antithetic=True, **kw)
    a2 = processes.simulate_gbm(RandomStream(7), antithetic=True, **kw)
    assert a1 == a2
    plain = processes.simulate_gbm(RandomStream(7), **kw)
    m_anti = sum(p[-1] for p in a1) / len(a1)
    m_plain = sum(p[-1] for p in plain) / len(plain)
    assert abs(m_anti - m_plain) < 2.0  # both unbiased for the same mean


# ----------------------------------------------------------------- options

OPT_KW = dict(s0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.20,
              n_paths=10_000, n_steps=63, seed=7)


def test_european_call_matches_black_scholes():
    res = options.european(**OPT_KW)
    assert abs(res["price"] - bs_call(100, 100, 1.0, 0.03, 0.20)) < 3 * res["std_error"]
    assert res["std_error"] > 0


def test_put_call_parity():
    c = options.european(kind="call", **OPT_KW)
    p = options.european(kind="put", **OPT_KW)
    fwd = 100 - 100 * math.exp(-0.03)
    assert abs((c["price"] - p["price"]) - fwd) < 1.0


def test_asian_cheaper_than_european():
    e = options.european(**OPT_KW)
    a = options.asian(**OPT_KW)
    assert a["price"] < e["price"]


def test_barrier_up_out_cheaper_than_european():
    e = options.european(**OPT_KW)
    b = options.barrier_up_out_call(B=130.0, **OPT_KW)
    assert b["price"] < e["price"]
    assert b["price"] > 0


# --------------------------------------------------------------- analytics

def test_var_cvar_known_values():
    pnl = [float(x) for x in range(-100, 100)]
    assert analytics.var(pnl, 0.95) == pytest.approx(90.05)
    assert analytics.cvar(pnl, 0.95) >= analytics.var(pnl, 0.95) - 1e-9


def test_var_sign_convention():
    # all profits → VaR is negative (a "loss" that is actually a gain)
    assert analytics.var([1.0, 2.0, 3.0, 4.0, 5.0]) < 0


def test_max_drawdown():
    assert analytics.max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)
    assert analytics.max_drawdown([100, 110, 120]) == pytest.approx(0.0)


def test_terminal_stats_keys():
    paths = processes.simulate_gbm(RandomStream(7), 100.0, 500, 63, 1 / 252,
                                   0.05, 0.2)
    s = analytics.terminal_stats(paths, 100.0)
    assert s["prob_profit"] > 0.4  # positive drift → majority profitable
    assert s["p05_pnl"] < s["median_pnl"] < s["p95_pnl"]
    json.dumps(s)


def test_prob_hit():
    paths = [[100, 110, 105], [100, 99, 98]]
    assert analytics.prob_hit(paths, 108, "above") == pytest.approx(0.5)
    assert analytics.prob_hit(paths, 99, "below") == pytest.approx(0.5)


# --------------------------------------------------------------- simulate

def test_run_simulation_deterministic_and_summary():
    cfg = SimConfig(model="jump", n_paths=500, n_steps=63, seed=7)
    r1, r2 = run_simulation(cfg), run_simulation(cfg)
    assert r1.paths == r2.paths
    s = r1.summary()
    assert s["model"] == "jump" and s["n_paths"] == 500
    json.dumps(r1.to_dict())


def test_run_simulation_bad_model():
    with pytest.raises(ValueError):
        run_simulation(SimConfig(model="nope"))


# --------------------------------------------------------------- resample

def test_block_bootstrap_shape_and_content():
    rets = [[i * 0.01, -i * 0.01] for i in range(50)]
    paths = block_bootstrap(rets, n_paths=10, block_len=7, seed=7)
    assert len(paths) == 10 and all(len(p) == 50 for p in paths)
    original = {tuple(r) for r in rets}
    assert all(tuple(row) in original for p in paths for row in p)


def test_block_bootstrap_deterministic():
    rets = [[0.01, -0.02], [0.03, 0.01]] * 25
    assert block_bootstrap(rets, 5, seed=7) == block_bootstrap(rets, 5, seed=7)


# ---------------------------------------------------------------- adapters

def test_to_agent_scenarios():
    out = to_agent_scenarios(SimConfig(model="gbm", n_paths=500, n_steps=63, seed=7))
    assert out["source"] == "trade-montecarlo"
    assert "var_95" in out["terminal"]
    assert "mean_max_dd" in out["drawdowns"]
    json.dumps(out)


def test_var_for_risk():
    pnl = [float(x) for x in range(-100, 100)]
    out = var_for_risk(pnl, alpha=0.95, capital=1000.0)
    assert out["var"] == pytest.approx(90.05)
    assert out["var_pct_of_capital"] == pytest.approx(0.09005)
    assert out["source"] == "trade-montecarlo"


def test_portfolio_var_runs():
    out = portfolio_var([0.6, 0.4], [100.0, 50.0], [0.05, 0.05], [0.2, 0.3],
                        [[1.0, 0.3], [0.3, 1.0]], capital=100_000.0,
                        n_paths=2000, n_steps=63, seed=7)
    assert out["var"] > 0 and out["cvar"] >= out["var"]


def test_bootstrap_for_backtest():
    rets = [[0.01, -0.01]] * 60
    out = bootstrap_for_backtest(rets, n_paths=8, block_len=10, seed=7)
    assert out["method"] == "circular-block-bootstrap"
    assert len(out["scenarios"]) == 8
    assert all(len(s) == 60 for s in out["scenarios"])


def test_ou_spread_config():
    cfg = ou_spread_config(half_life=21.0, spread_vol=0.02)
    assert cfg.model == "ou"
    assert cfg.theta == pytest.approx(math.log(2) / 21.0)


# ------------------------------------------------------- v0.2.0: Kou jumps

def test_kou_lam_zero_is_gbm():
    """Kou with lam=0 must be *exactly* GBM (no jump draws consumed)."""
    kw = dict(s0=100.0, n_paths=100, n_steps=60, dt=1 / 252, mu=0.05,
              sigma=0.2)
    a = processes.simulate_kou(RandomStream(11), lam=0.0, **kw)
    b = processes.simulate_gbm(RandomStream(11), **kw)
    assert a == b


def test_kou_kappa_closed_form():
    p_up, eta1, eta2 = 0.4, 25.0, 20.0
    expect = p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1
    assert processes._kou_kappa(p_up, eta1, eta2) == pytest.approx(expect)
    with pytest.raises(ValueError):
        processes._kou_kappa(0.4, 1.0, 20.0)  # eta1 must exceed 1
    with pytest.raises(ValueError):
        processes.simulate_kou(RandomStream(1), 100, 10, 10, 0.01, 0.05, 0.2,
                               0.5, p_up=1.5)


def test_kou_drift_compensated():
    """E[S_T] = s0·e^{μT}: the −λκ compensator keeps the mean growth at μ."""
    paths = processes.simulate_kou(RandomStream(21), 100.0, 20_000, 252,
                                   1 / 252, 0.05, 0.2, 0.8,
                                   p_up=0.4, eta1=25.0, eta2=20.0)
    mean = sum(p[-1] for p in paths) / len(paths)
    assert mean == pytest.approx(100 * math.exp(0.05), abs=1.5)


def test_kou_asymmetric_down_skew():
    """Mostly down-jumps → negatively skewed log-returns."""
    paths = processes.simulate_kou(RandomStream(22), 100.0, 8_000, 252,
                                   1 / 252, 0.05, 0.2, 1.0,
                                   p_up=0.2, eta1=25.0, eta2=8.0,
                                   antithetic=False)
    rets = [math.log(p[i + 1] / p[i]) for p in paths for i in range(252)]
    n = len(rets)
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / n)
    skew = sum((x - m) ** 3 for x in rets) / n / sd ** 3
    assert skew < -0.05


def test_kou_antithetic_unbiased_pair():
    paths = processes.simulate_kou(RandomStream(23), 100.0, 2_000, 60,
                                   1 / 252, 0.05, 0.2, 0.5, antithetic=True)
    assert len(paths) == 2_000
    assert all(p[0] == 100.0 and len(p) == 61 for p in paths)
    mean = sum(p[-1] for p in paths) / len(paths)
    assert mean == pytest.approx(100 * math.exp(0.05 * 60 / 252), rel=0.05)


def test_simconfig_kou_model():
    res = run_simulation(SimConfig(model="kou", n_paths=500, n_steps=60,
                                   lam=0.7, p_up=0.4, eta1=25.0, eta2=20.0))
    assert res.summary()["model"] == "kou"
    assert len(res.paths) == 500


# -------------------------------------------------- v0.2.0: calibration

def _gbm_returns(seed: int, n: int, mu: float, sigma: float,
                 dt: float) -> list[float]:
    s = RandomStream(seed)
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    return [drift + vol * s.normal() for _ in range(n)]


def test_bipower_vol_matches_realized_on_gbm():
    from trade_montecarlo.calibrate import bipower_vol
    rets = _gbm_returns(31, 5_000, 0.05, 0.20, 1 / 252)
    bv = bipower_vol(rets)
    rv = math.sqrt(sum(r * r for r in rets))  # sqrt(252·mean(r²))·... per-year
    rv = math.sqrt(sum(r * r for r in rets) / len(rets) * 252)
    assert bv == pytest.approx(0.20, abs=0.02)
    assert bv == pytest.approx(rv, rel=0.05)


def test_bipower_vol_robust_to_jumps():
    """Jumps inflate realized vol; bipower variation sees through them."""
    from trade_montecarlo.calibrate import bipower_vol
    paths = processes.simulate_jump(RandomStream(32), 100.0, 1, 2_520,
                                    1 / 252, 0.05, 0.18, 3.0, -0.08, 0.12,
                                    antithetic=False)
    rets = [math.log(paths[0][i + 1] / paths[0][i]) for i in range(2_520)]
    bv = bipower_vol(rets)
    rv = math.sqrt(sum(r * r for r in rets) / len(rets) * 252)
    assert bv == pytest.approx(0.18, abs=0.03)
    assert rv > bv + 0.03  # realized vol is jump-inflated


def test_fit_merton_recovers_jump_params():
    from trade_montecarlo.calibrate import fit_merton_jumps
    paths = processes.simulate_jump(RandomStream(99), 100.0, 1, 2_520,
                                    1 / 252, 0.05, 0.18, 3.0, -0.08, 0.12,
                                    antithetic=False)
    rets = [math.log(paths[0][i + 1] / paths[0][i]) for i in range(2_520)]
    fit = fit_merton_jumps(rets)
    # truncation undercounts small jumps: lam is a lower bound, sizes are
    # roughly right — documented behavior, not a bug
    assert 0.5 < fit["lam"] < 5.0
    assert fit["jump_mean"] == pytest.approx(-0.08, abs=0.08)
    assert fit["jump_vol"] == pytest.approx(0.12, abs=0.06)
    assert fit["sigma"] == pytest.approx(0.18, abs=0.04)
    assert fit["jump_share_var"] > 0.3
    assert fit["n_jumps"] >= 3


def test_fit_pure_gbm_gives_zero_lam():
    from trade_montecarlo.calibrate import fit_merton_jumps
    rets = _gbm_returns(34, 2_520, 0.05, 0.20, 1 / 252)
    fit = fit_merton_jumps(rets)
    assert fit["lam"] == 0.0  # no jump signature → pure diffusion


# ------------------------------------------------------ v0.2.0: exotics

def _norm_cdf(x: float) -> float:
    return NormalDist().cdf(x)


def test_digital_call_matches_closed_form():
    s0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.20
    d2 = (math.log(s0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    expect = math.exp(-r * T) * _norm_cdf(d2)
    got = options.digital(s0, K, T, r, sigma, n_paths=60_000, seed=7)
    assert got["price"] == pytest.approx(expect, abs=3 * got["std_error"] + 1e-3)
    assert got["kind"] == "digital-call"


def test_digital_put_call_parity():
    c = options.digital(100, 100, 1.0, 0.03, 0.2, kind="call",
                        n_paths=40_000, seed=7)
    p = options.digital(100, 100, 1.0, 0.03, 0.2, kind="put",
                        n_paths=40_000, seed=7)
    # same seed → same paths: call + put = discounted cash exactly-ish
    assert c["price"] + p["price"] == pytest.approx(math.exp(-0.03), abs=0.02)


def test_asset_or_nothing_matches_closed_form():
    s0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    d1 = (math.log(s0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    expect = s0 * _norm_cdf(d1)
    got = options.asset_or_nothing(s0, K, T, r, sigma, n_paths=60_000, seed=7)
    assert got["price"] == pytest.approx(expect, abs=3 * got["std_error"] + 0.05)


def test_geometric_asian_matches_kemna_vorst():
    """Kemna–Vorst closed form (independent reimplementation)."""
    s0, K, T, r, sigma, n = 100.0, 100.0, 1.0, 0.03, 0.20, 252
    dt = T / n
    m = math.log(s0) + (r - 0.5 * sigma ** 2) * T * (n + 1) / (2 * n)
    v_g = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n ** 2)
    F = math.exp(m + v_g / 2)
    d1 = (math.log(F / K) + v_g / 2) / math.sqrt(v_g)
    d2 = d1 - math.sqrt(v_g)
    expect = math.exp(-r * T) * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    got = options.geometric_asian(s0, K, T, r, sigma, n_paths=60_000,
                                  n_steps=n, seed=7)
    assert got["price"] == pytest.approx(expect, abs=3 * got["std_error"] + 0.02)


def test_barrier_in_out_parity():
    """Up-in + up-out = European on the SAME paths (path-by-path parity)."""
    paths = processes.simulate_gbm(RandomStream(41), 100.0, 30_000, 252,
                                   1 / 252, 0.03, 0.2, antithetic=True)
    kw = dict(s0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2, paths=paths)
    up_in = options.barrier(B=130.0, barrier_type="up-in", **kw)
    up_out = options.barrier(B=130.0, barrier_type="up-out", **kw)
    euro = options.european(s0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2,
                            paths=paths)
    assert up_in["price"] + up_out["price"] == pytest.approx(euro["price"],
                                                            abs=1e-9)
    kwd = dict(s0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2, paths=paths)
    dn_in = options.barrier(B=80.0, barrier_type="down-in", **kwd)
    dn_out = options.barrier(B=80.0, barrier_type="down-out", **kwd)
    assert dn_in["price"] + dn_out["price"] == pytest.approx(euro["price"],
                                                            abs=1e-9)


def test_barrier_up_out_backcompat():
    old = options.barrier_up_out_call(100, 100, 130, 1.0, 0.03, 0.2,
                                      n_paths=20_000, seed=7)
    new = options.barrier(100, 100, 130, 1.0, 0.03, 0.2, kind="call",
                          barrier_type="up-out", n_paths=20_000, seed=7)
    assert old["kind"] == "up-and-out-call"
    assert old["price"] == pytest.approx(new["price"], abs=1e-9)
    with pytest.raises(ValueError):
        options.barrier(100, 100, 90, 1.0, 0.03, 0.2, barrier_type="up-out")


def test_lookback_floating_positive_and_fixed_needs_K():
    c = options.lookback(100, 1.0, 0.03, 0.2, kind="call", n_paths=20_000,
                         seed=7)
    p = options.lookback(100, 1.0, 0.03, 0.2, kind="put", n_paths=20_000,
                         seed=7)
    assert c["price"] > 0 and p["price"] > 0
    assert c["kind"] == "lookback-floating-call"
    with pytest.raises(ValueError):
        options.lookback(100, 1.0, 0.03, 0.2, strike="fixed")
    f = options.lookback(100, 1.0, 0.03, 0.2, kind="call", strike="fixed",
                         K=100.0, n_paths=20_000, seed=7)
    assert f["price"] > 0


def _crr_american(s0: float, K: float, T: float, r: float, sigma: float,
                  kind: str, n: int = 500) -> float:
    """Cox-Ross-Rubinstein binomial, independent implementation for tests."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    q = (math.exp(r * dt) - d) / (u - d)
    ex = (lambda s: max(s - K, 0.0)) if kind == "call" else (lambda s: max(K - s, 0.0))
    vals = [ex(s0 * u ** j * d ** (n - j)) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        new = []
        for j in range(i + 1):
            s = s0 * u ** j * d ** (i - j)
            hold = disc * (q * vals[j + 1] + (1 - q) * vals[j])
            new.append(max(ex(s), hold))
        vals = new
    return vals[0]


def test_american_lsm_vs_binomial():
    crr = _crr_american(100, 100, 1.0, 0.05, 0.2, "put")
    lsm = options.american_lsm(100, 100, 1.0, 0.05, 0.2, kind="put",
                               n_paths=30_000, n_steps=50, seed=7)
    assert lsm["price"] == pytest.approx(crr, abs=4 * lsm["std_error"] + 0.05)


def test_american_put_bounds():
    paths = processes.simulate_gbm(RandomStream(43), 100.0, 30_000, 50,
                                   1 / 50, 0.05, 0.2, antithetic=True)
    kw = dict(s0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, paths=paths)
    am = options.american_lsm(kind="put", **kw)
    eu = options.european(kind="put", **kw)
    assert am["price"] >= eu["price"] - 1e-9  # early exercise never hurts
    amc = options.american_lsm(kind="call", **kw)
    euc = options.european(kind="call", **kw)
    # American call, no dividends ≈ European call
    assert amc["price"] == pytest.approx(euc["price"], abs=0.15)


def test_exotic_under_jump_models():
    kw = dict(n_paths=20_000, n_steps=252, seed=7)
    for model, mkw in (("jump", {"lam": 1.0, "jump_mean": -0.08,
                                 "jump_vol": 0.12}),
                       ("kou", {"lam": 1.0, "p_up": 0.3,
                                "eta1": 25.0, "eta2": 15.0})):
        d = options.digital(100, 100, 1.0, 0.03, 0.2, model=model,
                            model_kw=mkw, **kw)
        assert d["model"] == model and 0 < d["price"] < 1.0
        lk = options.lookback(100, 1.0, 0.03, 0.2, model=model,
                              model_kw=mkw, **kw)
        assert lk["price"] > 0
    with pytest.raises(ValueError):
        options.european(100, 100, 1.0, 0.03, 0.2, model="nope")


def test_paths_passthrough_used_verbatim():
    paths = processes.simulate_gbm(RandomStream(44), 100.0, 5_000, 100,
                                   0.01, 0.03, 0.2, antithetic=False)
    res = options.european(100, 100, 1.0, 0.03, 0.2, paths=paths)
    manual = math.exp(-0.03) * sum(max(p[-1] - 100, 0.0) for p in paths) / 5_000
    assert res["price"] == pytest.approx(manual, abs=1e-12)
    with pytest.raises(ValueError):
        options.european(100, 100, 1.0, 0.03, 0.2, paths=[[100.0]])


# ------------------------------------------------------------ v0.2.0: CLI

def test_cli_option_new_types(capsys):
    from trade_montecarlo.cli import main
    assert main(["option", "--type", "digital", "--paths", "2000"]) == 0
    out = capsys.readouterr().out
    assert "digital-call" in out
    assert main(["option", "--type", "american", "--kind", "put",
                 "--paths", "2000", "--steps", "12"]) == 0
    assert "american-put-lsm" in capsys.readouterr().out
    assert main(["option", "--type", "barrier", "--barrier-type", "down-in",
                 "--barrier", "80", "--paths", "2000"]) == 0
    assert "down-in-call" in capsys.readouterr().out
    assert main(["option", "--type", "lookback", "--paths", "2000"]) == 0
    assert "lookback-floating-call" in capsys.readouterr().out
    assert main(["option", "--type", "geometric-asian",
                 "--paths", "2000"]) == 0
    assert "geometric-asian-call" in capsys.readouterr().out


def test_cli_option_under_kou(capsys):
    from trade_montecarlo.cli import main
    assert main(["option", "--type", "european", "--model", "kou",
                 "--paths", "2000", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "kou"


def test_cli_simulate_kou(capsys):
    from trade_montecarlo.cli import main
    assert main(["simulate", "--model", "kou", "--paths", "500",
                 "--steps", "60", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["model"] == "kou"


def test_cli_calibrate(capsys):
    from trade_montecarlo.cli import main
    rets = ",".join(f"{r:.6f}" for r in _gbm_returns(45, 500, 0.05, 0.2, 1 / 252))
    assert main(["calibrate", f"--returns={rets}", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lam"] == 0.0  # pure GBM → no jump signature
    assert data["sigma"] == pytest.approx(0.2, abs=0.05)
