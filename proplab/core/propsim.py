"""Prop firm evaluation simulator.

Starts a fresh evaluation on EVERY day in history and plays your trades
forward under the firm's exact rules, at each size (1, 2, 3... micros).
Then does the same with coin-flip trades, so you can see whether your
strategy beats luck AT THE SIZE YOU WILL TRADE.

Intraday risk uses each trade's worst open loss (MAE) and best open profit
(MFE). For intraday trailing drawdowns we assume the best point came BEFORE
the worst point inside a trade - the pessimistic order.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import yaml

from .engine import NY


@dataclass
class PropRules:
    name: str = "example"
    account_size: float = 50_000
    profit_target: float = 3_000
    max_drawdown: float = 2_000
    drawdown_type: str = "eod_trailing"       # eod_trailing | intraday_trailing | static
    drawdown_lock_at: float | None = 50_000   # trailing floor stops rising here (None = never locks)
    daily_loss_limit: float | None = None
    daily_loss_action: str = "stop_day"       # stop_day | fail
    consistency_max_day_pct: float | None = None   # best day must be <= X% of total profit to pass
    min_trading_days: int = 1
    max_micros: int = 50
    max_days: int = 60                        # attempts not finished by then count as "timeout"
    eval_fee: float = 100.0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def _floor(rules: PropRules, hw: float) -> float:
    if rules.drawdown_type == "static":
        return rules.account_size - rules.max_drawdown
    f = hw - rules.max_drawdown
    return min(f, rules.drawdown_lock_at) if rules.drawdown_lock_at is not None else f


def run_eval(day_trades, start: int, rules: PropRules, qty: float):
    """Returns ('pass'|'fail'|'timeout', trading days used)."""
    bal = hw = rules.account_size
    floor = _floor(rules, hw)
    intraday = rules.drawdown_type == "intraday_trailing"
    best_day, traded_days, days = 0.0, 0, 0
    for d in range(start, min(start + rules.max_days, len(day_trades))):
        days += 1
        day_pnl = 0.0
        trades = day_trades[d]
        for p, mae, mfe in trades:
            p, mae, mfe = p * qty, mae * qty, mfe * qty
            if intraday:
                hw = max(hw, bal + mfe)
                floor = _floor(rules, hw)
            hit_dll = rules.daily_loss_limit is not None and day_pnl + mae <= -rules.daily_loss_limit
            worst = (-rules.daily_loss_limit - day_pnl) if hit_dll else mae
            if bal + worst <= floor:
                return "fail", days
            if hit_dll:
                if rules.daily_loss_action == "fail":
                    return "fail", days
                bal += worst
                day_pnl += worst
                break
            bal += p
            day_pnl += p
            if intraday:
                hw = max(hw, bal)
                floor = _floor(rules, hw)
        if trades:
            traded_days += 1
        if rules.drawdown_type == "eod_trailing":
            hw = max(hw, bal)
            floor = _floor(rules, hw)
        if bal <= floor:
            return "fail", days
        best_day = max(best_day, day_pnl)
        profit = bal - rules.account_size
        consistent = (rules.consistency_max_day_pct is None
                      or best_day <= rules.consistency_max_day_pct / 100 * profit)
        if profit >= rules.profit_target and traded_days >= rules.min_trading_days and consistent:
            return "pass", days
    return "timeout", days


def bucket_by_day(trades: pd.DataFrame, session_days) -> list:
    pos = {d: i for i, d in enumerate(session_days)}
    out = [[] for _ in session_days]
    if len(trades) == 0:
        return out
    t = trades.sort_values("entry_time")
    days = t["entry_time"].dt.tz_convert(NY).dt.date
    for d, p, mae, mfe in zip(days, t["pnl"], t["mae_usd"], t["mfe_usd"]):
        if d in pos:
            out[pos[d]].append((float(p), float(mae), float(mfe)))
    return out


def simulate(trades, session_days, rules: PropRules, qty: float, start_step: int = 1) -> dict:
    day_trades = bucket_by_day(trades, session_days)
    last_start = len(day_trades) - rules.max_days if len(day_trades) > rules.max_days else len(day_trades)
    res, days_to_pass = {"pass": 0, "fail": 0, "timeout": 0}, []
    for s in range(0, max(last_start, 1), start_step):
        outcome, used = run_eval(day_trades, s, rules, qty)
        res[outcome] += 1
        if outcome == "pass":
            days_to_pass.append(used)
    n = sum(res.values())
    pr = res["pass"] / n if n else 0.0
    return {"attempts": n, "pass_%": round(100 * pr, 1), "fail_%": round(100 * res["fail"] / n, 1) if n else 0,
            "timeout_%": round(100 * res["timeout"] / n, 1) if n else 0,
            "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else np.nan,
            "cost_per_pass_$": round(rules.eval_fee / pr, 0) if pr > 0 else np.inf}


def compare_sizes(trades, random_sets, session_days, rules: PropRules, sizes, start_step: int = 1) -> pd.DataFrame:
    rows = []
    for q in sizes:
        if q > rules.max_micros:
            continue
        r = simulate(trades, session_days, rules, q, start_step)
        rand = [simulate(rs, session_days, rules, q, max(start_step, 3))["pass_%"] for rs in random_sets]
        r["coinflip_pass_%"] = round(float(np.mean(rand)), 1) if rand else np.nan
        r["edge_vs_coinflip_pts"] = round(r["pass_%"] - r["coinflip_pass_%"], 1) if rand else np.nan
        rows.append({"micros": q, **r})
    return pd.DataFrame(rows).set_index("micros") if rows else pd.DataFrame()
