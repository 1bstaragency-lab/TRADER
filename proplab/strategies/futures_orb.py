"""Opening Range Breakout (NQ/MNQ) - a well-known public strategy, used here
as the first thing to push through the funnel. Not a recommendation.

Rules:
  * Opening range = high/low of the first `or_minutes` of the cash session (9:30 ET).
  * After the range is complete and before `last_entry`, the first 1-minute
    CLOSE above the range high -> long; below the range low -> short.
  * Stop = other side of the range. Target = rr x stop distance.
  * One trade per day. Flat by `exit_time`.
  * Skip days whose range is too small or too large, and any date in skip_dates_file
    (e.g. CPI / FOMC / NFP days - one date per line, YYYY-MM-DD).

Who loses money to it (if it works): traders fading the open who get squeezed
when the early range breaks with momentum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"

DEFAULT_PARAMS = {
    "or_start": "09:30",
    "or_minutes": 15,
    "last_entry": "11:00",
    "exit_time": "15:55",
    "rr": 2.0,
    "min_range_pts": 10.0,
    "max_range_pts": 150.0,
    "direction": "both",          # both | long | short
    "skip_dates_file": None,
}


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    naive = df.index.tz_convert(NY).tz_localize(None)
    mins = np.asarray(naive.hour * 60 + naive.minute)
    day = naive.normalize()
    or_s = _hm(p["or_start"])
    or_e = or_s + int(p["or_minutes"])
    in_or = (mins >= or_s) & (mins < or_e)

    high, low, close = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    f = pd.DataFrame({"day": day, "h": np.where(in_or, high, np.nan), "l": np.where(in_or, low, np.nan),
                      "n_or": in_or.astype(int)})
    g = f.groupby("day", sort=False)
    # running max/min of the range bars seen SO FAR today (never looks ahead)
    or_hi = g["h"].cummax().groupby(f["day"]).ffill().to_numpy()
    or_lo = g["l"].cummin().groupby(f["day"]).ffill().to_numpy()
    n_or = g["n_or"].cumsum().to_numpy()

    eligible = (mins >= or_e) & (mins < _hm(p["last_entry"])) & (n_or >= 0.8 * int(p["or_minutes"]))
    rng = or_hi - or_lo
    eligible &= (rng >= p["min_range_pts"]) & (rng <= p["max_range_pts"])
    if p.get("skip_dates_file"):
        with open(p["skip_dates_file"]) as fh:
            skip = {pd.Timestamp(x.strip()) for x in fh if x.strip()}
        eligible &= ~np.asarray(day.isin(list(skip)))

    long_sig = eligible & (close > or_hi)
    short_sig = eligible & (close < or_lo)
    if p["direction"] == "long":
        short_sig[:] = False
    elif p["direction"] == "short":
        long_sig[:] = False
    cand = pd.Series(long_sig | short_sig)
    first = cand & (cand.groupby(f["day"]).cumsum() == 1)
    idx = np.flatnonzero(first.to_numpy())
    if len(idx) == 0:
        return pd.DataFrame(columns=["idx", "side", "stop_dist", "target_dist", "exit_time"])

    side = np.where(long_sig[idx], 1, -1)
    stop_dist = np.where(side == 1, close[idx] - or_lo[idx], or_hi[idx] - close[idx])
    xm = _hm(p["exit_time"])
    exit_time = (day[idx] + pd.to_timedelta(xm, unit="min")).tz_localize(NY)
    return pd.DataFrame({"idx": idx, "side": side, "stop_dist": stop_dist,
                         "target_dist": p["rr"] * stop_dist, "exit_time": exit_time})
