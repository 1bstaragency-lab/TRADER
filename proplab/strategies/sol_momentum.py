"""Memecoin volume breakout (Solana) - a common public idea, used as a starting
point for the funnel. Not a recommendation.

Rules (on 1-minute bars of one pool):
  * Close breaks above the highest high of the previous `lookback` bars
  * AND this bar's volume is > vol_mult x the average of the previous `lookback` bars
  * Long only. Stop = stop_pct below, target = target_pct above, out after max_hold_min.

Who loses money to it (if it works): late sellers and slow bots that sell
into the start of a momentum burst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "lookback": 30,
    "vol_mult": 3.0,
    "stop_pct": 0.12,
    "target_pct": 0.25,
    "max_hold_min": 90,
    "min_bars": 60,        # ignore the first hour of a pool's life (launch chaos)
}


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    lb = int(p["lookback"])
    prev_high = df["high"].shift(1).rolling(lb).max()
    prev_vol = df["volume"].shift(1).rolling(lb).mean()
    cond = (df["close"] > prev_high) & (df["volume"] > p["vol_mult"] * prev_vol) & (prev_vol > 0)
    cond &= np.arange(len(df)) >= int(p["min_bars"])
    idx = np.flatnonzero(cond.to_numpy())
    close = df["close"].to_numpy(float)[idx]
    return pd.DataFrame({
        "idx": idx, "side": 1,
        "stop_dist": close * p["stop_pct"], "target_dist": close * p["target_pct"],
        "exit_time": df.index[idx] + pd.Timedelta(minutes=int(p["max_hold_min"])),
    })
