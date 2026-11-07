"""The funnel shared by the futures and Solana labs:
honest backtest -> unseen data -> 2x costs -> coin flip -> (futures) prop sizing."""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from .core.coinflip import coin_flip_test
from .core.engine import Bars, backtest
from .core.stats import by_year, summarize


def run_period(dfs: list[pd.DataFrame], strategy, params: dict, inst, flips: int, sampler: str,
               keep_sets: int = 0, seed: int = 7) -> dict:
    """Backtest one period (one or many instruments/pools) and run its coin-flip test."""
    datasets, datasets_2x = [], []
    inst2 = inst.scaled_costs(2.0)
    for df in dfs:
        if len(df) < 2:
            continue
        bars = Bars(df)
        sig = strategy.generate_signals(df, params)
        datasets.append((bars, backtest(bars, sig, inst)))
        datasets_2x.append(backtest(bars, sig, inst2))
    trades = _cat([t for _, t in datasets])
    trades_2x = _cat(datasets_2x)
    start = min(df.index.min() for df in dfs if len(df))
    end = max(df.index.max() for df in dfs if len(df))
    cf = coin_flip_test(datasets, inst, runs=flips, sampler=sampler, seed=seed, keep_sets=keep_sets)
    return {"trades": trades, "summary": summarize(trades, start, end),
            "summary_2x": summarize(trades_2x, start, end), "coinflip": cf, "start": start, "end": end}


def _cat(parts):
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["pnl", "gross", "entry_time", "exit_time", "mae_usd", "mfe_usd"])


def side_by_side(periods: dict) -> pd.DataFrame:
    rows = {}
    for name, r in periods.items():
        s = dict(r["summary"])
        s["net_at_2x_costs_$"] = r["summary_2x"]["net_$"]
        cf = r["coinflip"]
        s["coinflip_beat_%"] = cf["beat_%_of_random"]
        s["coinflip_p_value"] = cf["p_value"]
        s["coinflip_random_median_$"] = cf["random_median_$"]
        rows[f"{name} ({r['start']:%Y-%m-%d} to {r['end']:%Y-%m-%d})"] = s
    return pd.DataFrame(rows).map(_fmt)


def _fmt(x):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "-"
        if np.isinf(x):
            return "inf"
        return f"{x:,.0f}" if float(x).is_integer() and abs(x) >= 100 else f"{x:,.2f}"
    return x


def checklist(periods: dict, min_beat: float, min_trades: int) -> list[tuple[str, bool]]:
    is_, oos = periods["IN-SAMPLE"], periods["OUT-OF-SAMPLE"]
    return [
        ("Profitable after costs on the years it was built on", is_["summary"]["net_$"] > 0),
        ("Profitable after costs on years it has NEVER seen", oos["summary"]["net_$"] > 0),
        ("Still profitable at 2x costs on unseen years", oos["summary_2x"]["net_$"] > 0),
        (f"Beats >= {min_beat:.0f}% of coin flips (built-on years)", is_["coinflip"]["beat_%_of_random"] >= min_beat),
        (f"Beats >= {min_beat:.0f}% of coin flips (unseen years)", oos["coinflip"]["beat_%_of_random"] >= min_beat),
        (f"At least {min_trades} trades on unseen years", oos["summary"]["trades"] >= min_trades),
    ]


def multiple_testing_note(stats: dict, family: str) -> str:
    n = stats["distinct_in_family"]
    return (f"Research log: this is variation #{n} of '{family}' "
            f"({stats['distinct_variations_total']} distinct variations across all strategies). "
            f"Even with zero real edge, about 1 in 20 variations beats 95% of coin flips by luck - "
            f"so after {n} tries, ~{n * 0.05:.1f} lucky 'winners' are expected. "
            "Treat a lone winner among many tries with suspicion; the paper-trading stage is the real judge.")


def write_report(name: str, sections: list[tuple[str, str]], out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dt.datetime.now():%Y%m%d-%H%M%S}_{name}.md")
    with open(path, "w") as f:
        for title, body in sections:
            f.write(f"## {title}\n\n{body}\n\n")
    return path


def block(obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return "```\n" + obj.to_string() + "\n```"
    return "```\n" + str(obj) + "\n```"


def metrics_for_log(periods: dict) -> dict:
    out = {}
    for name, r in periods.items():
        out[name] = {**r["summary"], "net_2x": r["summary_2x"]["net_$"],
                     "coinflip_beat_%": r["coinflip"]["beat_%_of_random"]}
    return json.loads(json.dumps(out, default=lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)))
