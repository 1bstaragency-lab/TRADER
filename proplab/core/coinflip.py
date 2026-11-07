"""The coin-flip test.

Your strategy vs. hundreds of copies of itself that keep the SAME number of
trades, the SAME stop and target distances and the SAME holding window - but
enter at random times in a random direction. If random entries with your stop
make as much money, your entry rules are not the reason you made money.

Samplers:
  time_of_day - futures: random day, same minute of the day as the real trade
  anywhere    - crypto: random bar in the same coin's history
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Bars, simulate_signals


def random_signals(bars: Bars, trades: pd.DataFrame, rng, sampler: str, can_short: bool):
    k = len(trades)
    old = trades["sig_idx"].to_numpy(np.int64)
    hold_ns = trades["planned_exit_ns"].to_numpy(np.int64) - bars.ns[old]
    if sampler == "time_of_day":
        days, grid, minute = bars.session_grid()
        m = minute[old]
        new = np.full(k, -1, dtype=np.int64)
        todo = np.arange(k)
        for _ in range(50):
            if len(todo) == 0:
                break
            cand = grid[rng.integers(len(days), size=len(todo)), m[todo]]
            got = cand >= 0
            new[todo[got]] = cand[got]
            todo = todo[~got]
    elif sampler == "anywhere":
        new = rng.integers(0, max(bars.n - 1, 1), size=k)
    else:
        raise ValueError(sampler)
    keep = new >= 0
    new, old, hold_ns = new[keep], old[keep], hold_ns[keep]
    scale = bars.c[new] / bars.c[old]      # same stop size relative to price
    side = rng.choice(np.array([-1, 1]), size=len(new)) if can_short else np.ones(len(new), dtype=np.int64)
    return (new, side, trades["stop_dist"].to_numpy(float)[keep] * scale,
            trades["target_dist"].to_numpy(float)[keep] * scale, bars.ns[new] + hold_ns)


def coin_flip_test(datasets, inst, runs: int = 500, sampler: str = "time_of_day",
                   seed: int = 7, keep_sets: int = 0) -> dict:
    """datasets: list of (Bars, real_trades). Returns how the real result ranks vs random entries."""
    rng = np.random.default_rng(seed)
    real = float(sum(t["pnl"].sum() for _, t in datasets))
    nets, kept = np.empty(runs), []
    for r in range(runs):
        total, sets = 0.0, []
        for bars, t in datasets:
            if len(t) == 0:
                continue
            rt = simulate_signals(bars, *random_signals(bars, t, rng, sampler, inst.can_short), inst)
            total += float(rt["pnl"].sum())
            if r < keep_sets:
                sets.append(rt)
        nets[r] = total
        if r < keep_sets:
            kept.append(pd.concat(sets, ignore_index=True) if sets else pd.DataFrame())
    return {
        "real_net_$": round(real, 2),
        "random_median_$": round(float(np.median(nets)), 2),
        "random_95th_pct_$": round(float(np.percentile(nets, 95)), 2),
        "beat_%_of_random": round(100 * float(np.mean(nets < real)), 1),
        "p_value": round((1 + int(np.sum(nets >= real))) / (runs + 1), 4),
        "runs": runs,
        "random_sets": kept,
    }
