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
