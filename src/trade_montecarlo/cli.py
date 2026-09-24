"""Command-line interface for trade-montecarlo."""

from __future__ import annotations

import argparse
import json

from . import __version__, analytics
from .adapters import to_agent_scenarios
from .calibrate import fit_merton_jumps
from .licensing import check_license, check_update
from .options import (
    american_lsm,
    asian,
    asset_or_nothing,
    barrier,
    barrier_up_out_call,
    digital,
    european,
    geometric_asian,
    lookback,
)
from .simulate import SimConfig, run_simulation


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = SimConfig(model=args.model, n_paths=args.paths, n_steps=args.steps,
                    T=args.T, s0=args.s0, mu=args.mu, sigma=args.sigma,
                    seed=args.seed, antithetic=not args.no_antithetic,
                    lam=args.lam, theta=args.theta, long_mean=args.long_mean,
                    p_up=args.p_up, eta1=args.eta1, eta2=args.eta2)
    result = run_simulation(cfg)
    out = {"summary": result.summary(),
           "drawdowns": analytics.drawdown_stats(result.paths)}
    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        s = out["summary"]
        d = out["drawdowns"]
        print(f"{args.model}: {s['n_paths']} paths x {s['n_steps']} steps, "
              f"s0={args.s0}")
        print(f"mean P&L {s['mean_pnl']:+.2f}  std {s['std_pnl']:.2f}  "
              f"P(profit) {s['prob_profit']:.1%}")
        print(f"VaR95 {s['var_95']:.2f}  CVaR95 {s['cvar_95']:.2f}  "
              f"median P&L {s['median_pnl']:+.2f}")
        print(f"max drawdown: mean {d['mean_max_dd']:.1%}, "
              f"p95 {d['p95_max_dd']:.1%}, worst {d['worst_max_dd']:.1%}")
    return 0


def cmd_option(args: argparse.Namespace) -> int:
    steps = args.steps
    if steps is None:  # sensible default per payoff
        steps = 50 if args.type == "american" else 252
    kw = dict(s0=args.s0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
              n_paths=args.paths, seed=args.seed, model=args.model,
              model_kw={"lam": args.lam, "jump_mean": args.jump_mean,
                        "jump_vol": args.jump_vol, "p_up": args.p_up,
                        "eta1": args.eta1, "eta2": args.eta2})
    if args.type == "european":
        res = european(kind=args.kind, n_steps=steps, **kw)
    elif args.type == "asian":
        res = asian(kind=args.kind, n_steps=steps, **kw)
    elif args.type == "geometric-asian":
        res = geometric_asian(kind=args.kind, n_steps=steps, **kw)
    elif args.type == "digital":
        res = digital(kind=args.kind, cash=args.cash, n_steps=steps, **kw)
    elif args.type == "asset-or-nothing":
        res = asset_or_nothing(kind=args.kind, n_steps=steps, **kw)
    elif args.type == "lookback":
        lk = dict(kw)
        lk.pop("K")
        res = lookback(strike=args.strike, K=args.K, n_steps=steps, **lk)
    elif args.type == "american":
        res = american_lsm(kind=args.kind, n_steps=steps, **kw)
    elif args.type == "barrier":
        res = barrier(B=args.barrier, kind=args.kind,
                      barrier_type=args.barrier_type, n_steps=steps, **kw)
    else:  # legacy "up-out" alias
        res = barrier_up_out_call(B=args.barrier, n_steps=steps,
                                  s0=args.s0, K=args.K, T=args.T, r=args.r,
                                  sigma=args.sigma, n_paths=args.paths,
                                  seed=args.seed)
    if args.format == "json":
        print(json.dumps(res, indent=2))
    else:
        print(f"{res['kind']}: price {res['price']:.4f} "
              f"± {res['std_error']:.4f} ({res['n_paths']} paths, "
              f"model={res['model']})")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.file:
        with open(args.file) as f:
            rets = [float(line.strip()) for line in f if line.strip()]
    else:
        rets = [float(x) for x in args.returns.split(",") if x.strip()]
    fit = fit_merton_jumps(rets, dt=args.dt, sigma=args.sigma)
    if args.format == "json":
        print(json.dumps(fit, indent=2))
    else:
        print(f"fitted Merton jumps on {fit['n_returns']} returns "
              f"(dt={args.dt:g}):")
        print(f"  lam={fit['lam']:.3f}/yr  jump_mean={fit['jump_mean']:+.4f}  "
              f"jump_vol={fit['jump_vol']:.4f}")
        print(f"  sigma={fit['sigma']:.4f} (bipower {fit['sigma_bipower']:.4f})"
              f"  jump share of var: {fit['jump_share_var']:.1%}")
        if "note" in fit:
            print(f"  note: {fit['note']}")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    cfg = SimConfig(model=args.model, n_paths=args.paths, seed=args.seed)
    print(json.dumps(to_agent_scenarios(cfg), indent=2))
    return 0


def cmd_license(_args: argparse.Namespace) -> int:
    print(json.dumps(check_license(), indent=2))
    return 0


def cmd_update_check(_args: argparse.Namespace) -> int:
    print(json.dumps(check_update(), indent=2))
    return 0


def _add_sim_args(p: argparse.ArgumentParser, steps_default: int | None = 252,
                  paths_default: int = 10_000) -> None:
    p.add_argument("--paths", type=int, default=paths_default)
    p.add_argument("--steps", type=int, default=steps_default)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--p-up", type=float, default=0.4)
    p.add_argument("--eta1", type=float, default=25.0)
    p.add_argument("--eta2", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trade-montecarlo",
                                 description="Monte Carlo simulation for trading")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="run a Monte Carlo session")
    p.add_argument("--model", choices=["gbm", "jump", "kou", "garch", "ou"],
                   default="gbm")
    p.add_argument("--theta", type=float, default=2.0)
    p.add_argument("--long-mean", type=float, default=0.0)
    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--format", choices=["text", "json"], default="text")
    _add_sim_args(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("option", help="price an option by Monte Carlo")
    p.add_argument("--type",
                   choices=["european", "asian", "geometric-asian", "digital",
                            "asset-or-nothing", "lookback", "american",
                            "barrier", "up-out"],
                   default="european")
    p.add_argument("--kind", choices=["call", "put"], default="call")
    p.add_argument("--model", choices=["gbm", "jump", "kou"], default="gbm",
                   help="price process: GBM or jump-diffusion")
    p.add_argument("--barrier", type=float, default=130.0)
    p.add_argument("--barrier-type",
                   choices=["up-in", "up-out", "down-in", "down-out"],
                   default="up-out")
    p.add_argument("--strike", choices=["floating", "fixed"],
                   default="floating", help="lookback strike style")
    p.add_argument("--cash", type=float, default=1.0,
                   help="digital cash payout")
    p.add_argument("--r", type=float, default=0.03)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--jump-mean", type=float, default=-0.05)
    p.add_argument("--jump-vol", type=float, default=0.10)
    p.add_argument("--format", choices=["text", "json"], default="text")
    _add_sim_args(p, steps_default=None, paths_default=50_000)
    p.set_defaults(func=cmd_option)

    p = sub.add_parser("calibrate",
                       help="fit Merton jump params to log-returns")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--returns",
                     help="comma-separated log-returns, e.g. 0.01,-0.02,...")
    src.add_argument("--file", help="file with one log-return per line")
    p.add_argument("--dt", type=float, default=1 / 252,
                   help="year-fraction per return (default: daily)")
    p.add_argument("--sigma", type=float, default=None,
                   help="diffusion vol override (default: bipower estimate)")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("scenarios", help="agent-ready scenario JSON")
    p.add_argument("--model", choices=["gbm", "jump", "kou", "garch", "ou"],
                   default="gbm")
    p.add_argument("--paths", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("license", help="check the license-key hook")
    p.set_defaults(func=cmd_license)
    p = sub.add_parser("update-check", help="check for a newer release")
    p.set_defaults(func=cmd_update_check)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
