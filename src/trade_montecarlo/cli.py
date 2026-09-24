"""Command-line interface for trade-montecarlo."""

from __future__ import annotations

import argparse
import json

from . import __version__, analytics
from .adapters import to_agent_scenarios
from .licensing import check_license, check_update
from .options import asian, barrier_up_out_call, european
from .simulate import SimConfig, run_simulation


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = SimConfig(model=args.model, n_paths=args.paths, n_steps=args.steps,
                    T=args.T, s0=args.s0, mu=args.mu, sigma=args.sigma,
                    seed=args.seed, antithetic=not args.no_antithetic,
                    lam=args.lam, theta=args.theta, long_mean=args.long_mean)
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
    kw = dict(s0=args.s0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
              n_paths=args.paths, seed=args.seed)
    if args.type == "european":
        res = european(kind=args.kind, n_steps=args.steps, **kw)
    elif args.type == "asian":
        res = asian(kind=args.kind, n_steps=args.steps, **kw)
    else:
        res = barrier_up_out_call(B=args.barrier, n_steps=args.steps, **kw)
    if args.format == "json":
        print(json.dumps(res, indent=2))
    else:
        print(f"{res['kind']}: price {res['price']:.4f} "
              f"± {res['std_error']:.4f} ({res['n_paths']} paths)")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trade-montecarlo",
                                 description="Monte Carlo simulation for trading")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="run a Monte Carlo session")
    p.add_argument("--model", choices=["gbm", "jump", "garch", "ou"], default="gbm")
    p.add_argument("--paths", type=int, default=10_000)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--theta", type=float, default=2.0)
    p.add_argument("--long-mean", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("option", help="price an option by Monte Carlo")
    p.add_argument("--type", choices=["european", "asian", "barrier"], default="european")
    p.add_argument("--kind", choices=["call", "put"], default="call")
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--barrier", type=float, default=130.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.03)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--paths", type=int, default=50_000)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_option)

    p = sub.add_parser("scenarios", help="agent-ready scenario JSON")
    p.add_argument("--model", choices=["gbm", "jump", "garch", "ou"], default="gbm")
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
