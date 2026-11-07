"""The backtester. Built so it cannot cheat.

Rules it enforces (nothing in a strategy can turn them off):
  1. A signal is computed from a bar that has CLOSED. The order fills at the
     OPEN of the NEXT bar, plus slippage.
  2. Stops and targets are checked against each later bar's high/low. If both
     could have been hit inside the same bar, we assume the STOP hit first.
  3. If price gaps through your stop, you get filled at the (worse) open.
  4. A limit target only fills if price traded THROUGH it (futures: by 1 tick).
  5. Commission and slippage are charged on every fill (see instruments.py).
  6. Time exit at the close of the last bar at or before the planned exit time.

The same fill rules are re-implemented bar-by-bar in `PaperPosition` (used by
the live forward tester), and tests/test_engine.py proves both give identical
results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import ENTRY, KIND_NAMES, STOP, TARGET, TIME

NY = "America/New_York"
SIGNAL_COLUMNS = ["idx", "side", "stop_dist", "target_dist", "exit_time"]


def to_ns(x) -> np.ndarray:
    """Timezone-aware timestamps -> int64 nanoseconds since epoch (UTC)."""
    idx = pd.DatetimeIndex(x)
    if idx.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    return idx.tz_convert("UTC").as_unit("ns").asi8


class Bars:
    """OHLCV bars as fast numpy arrays. Index = bar OPEN time, timezone-aware."""

    def __init__(self, df: pd.DataFrame):
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"bars need a '{col}' column")
        self.df = df
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.ns = to_ns(df.index)
        self.n = len(df)
        if self.n > 1 and np.any(np.diff(self.ns) <= 0):
            raise ValueError("bars must be sorted and have no duplicate timestamps")
        self._grid = None

    def exit_by(self, exit_ns) -> np.ndarray:
        """Index of the last bar whose time is <= the planned exit time."""
        return np.searchsorted(self.ns, np.asarray(exit_ns, dtype=np.int64), side="right") - 1

    def session_grid(self):
        """(days, grid[day, minute_of_day] -> bar index or -1, minute_of_day per bar) in New York time."""
        if self._grid is None:
            local = self.df.index.tz_convert(NY).tz_localize(None)
            day_key = local.normalize().as_unit("ns").asi8
            days, day_id = np.unique(day_key, return_inverse=True)
            minute = np.asarray(local.hour * 60 + local.minute, dtype=np.int64)
            grid = np.full((len(days), 1440), -1, dtype=np.int64)
            grid[day_id, minute] = np.arange(self.n)
            self._grid = (days, grid, minute)
        return self._grid


def empty_trades() -> pd.DataFrame:
    cols = ["sig_idx", "entry_idx", "exit_idx", "side", "stop_dist", "target_dist", "planned_exit_ns",
            "raw_entry", "entry", "raw_exit", "exit", "kind", "pnl", "gross", "mae_usd", "mfe_usd",
            "entry_time", "exit_time"]
    return pd.DataFrame(columns=cols)


def simulate_signals(bars: Bars, idx, side, stop_dist, target_dist, exit_ns, inst, chunk: int = 2000) -> pd.DataFrame:
    """Simulate every signal independently (ignores overlap). One row per valid trade."""
    idx = np.asarray(idx, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    sd = np.asarray(stop_dist, dtype=float)
    td = np.asarray(target_dist, dtype=float)
    exit_ns = np.asarray(exit_ns, dtype=np.int64)
    exit_by = bars.exit_by(exit_ns)
    e = idx + 1
    ok = (idx >= 0) & (e < bars.n) & (e <= exit_by) & (sd > 0) & (td > 0) & np.isin(side, (-1, 1))
    if not inst.can_short:
        ok &= side == 1
    sel = np.flatnonzero(ok)
    parts = [_sim_chunk(bars, sel[i:i + chunk], idx, side, sd, td, exit_by, exit_ns, inst)
             for i in range(0, len(sel), chunk)]
    if not parts:
        return empty_trades()
    return pd.concat(parts, ignore_index=True)


def _sim_chunk(bars, s, idx, side, sd, td, exit_by, exit_ns, inst) -> pd.DataFrame:
    i0, sside, sd_, td_, xb = idx[s], side[s], sd[s], td[s], exit_by[s]
    e = i0 + 1
    hold = xb - e + 1
    width = int(hold.max())
    offs = np.arange(width)
    M = np.minimum(e[:, None] + offs, bars.n - 1)
    win = offs[None, :] < hold[:, None]
    lo, hi = bars.l[M], bars.h[M]

    raw_entry = bars.o[e]
    entry = raw_entry + sside * inst.slip(raw_entry, ENTRY)       # buy higher / sell lower
    stop = entry - sside * sd_
    tgt = entry + sside * td_
    thr = inst.through(entry)
    is_long = sside == 1
    L = is_long[:, None]

    stop_hit = np.where(L, lo <= stop[:, None], hi >= stop[:, None]) & win
    tgt_hit = np.where(L, hi >= (tgt + thr)[:, None], lo <= (tgt - thr)[:, None]) & win
    big = width + 1
    fs = np.where(stop_hit.any(1), stop_hit.argmax(1), big)
    ft = np.where(tgt_hit.any(1), tgt_hit.argmax(1), big)
    is_stop = (fs < big) & (fs <= ft)                  # same bar -> stop wins
    is_tgt = (~is_stop) & (ft < big)
    off = np.where(is_stop, fs, np.where(is_tgt, ft, hold - 1))
    j = e + off

    oj = bars.o[j]
    gapped = np.where(is_long, oj <= stop, oj >= stop)
    raw_exit = np.where(is_stop, np.where(gapped, oj, stop), np.where(is_tgt, tgt, bars.c[j]))
    kind = np.where(is_stop, STOP, np.where(is_tgt, TARGET, TIME))
    exit_px = raw_exit - sside * inst.slip(raw_exit, kind)      # exit order is the opposite side

    upto = offs[None, :] <= off[:, None]
    lo_min = np.where(upto, lo, np.inf).min(1)
    hi_max = np.where(upto, hi, -np.inf).max(1)
    mae_px = np.where(is_long, lo_min - entry, entry - hi_max)
    mfe_px = np.where(is_long, hi_max - entry, entry - lo_min)
    pnl = inst.pnl(sside, entry, exit_px)

    index = bars.df.index
    return pd.DataFrame({
        "sig_idx": i0, "entry_idx": e, "exit_idx": j, "side": sside,
        "stop_dist": sd_, "target_dist": td_, "planned_exit_ns": exit_ns[s],
        "raw_entry": raw_entry, "entry": entry, "raw_exit": raw_exit, "exit": exit_px,
        "kind": pd.Series(kind).map(KIND_NAMES).to_numpy(),
        "pnl": pnl, "gross": inst.gross(sside, raw_entry, raw_exit),
        "mae_usd": np.minimum(inst.to_dollars(mae_px, entry), pnl),
        "mfe_usd": np.maximum(inst.to_dollars(mfe_px, entry), pnl),
        "entry_time": index[e], "exit_time": index[j],
    })


def backtest(bars: Bars, signals: pd.DataFrame, inst) -> pd.DataFrame:
    """Run a strategy's signals. One position at a time: signals while in a trade are ignored."""
    if signals is None or len(signals) == 0:
        return empty_trades()
    sig = signals.sort_values("idx", kind="stable")
    t = simulate_signals(bars, sig["idx"], sig["side"], sig["stop_dist"], sig["target_dist"],
                         to_ns(sig["exit_time"]), inst)
    keep, busy = [], -1
    for k, (si, xi) in enumerate(zip(t["sig_idx"].to_numpy(), t["exit_idx"].to_numpy())):
        if si >= busy:          # signal on the exit bar's close -> enters on the next bar: allowed
            keep.append(k)
            busy = xi
    return t.iloc[keep].reset_index(drop=True)


class PaperPosition:
    """Bar-by-bar version of the exact same fill rules, for forward testing live."""

    def __init__(self, side, raw_entry, stop_dist, target_dist, exit_ns, inst):
        self.side, self.inst, self.exit_ns = int(side), inst, int(exit_ns)
        self.raw_entry = float(raw_entry)
        self.entry = float(raw_entry + side * inst.slip(raw_entry, ENTRY))
        self.stop = self.entry - side * stop_dist
        self.target = self.entry + side * target_dist
        self.thr = float(inst.through(self.entry))
        self.last_close = None

    def _done(self, kind, raw):
        px = float(raw - self.side * self.inst.slip(raw, kind))
        return {"kind": KIND_NAMES[kind], "raw_exit": float(raw), "exit": px,
                "pnl": float(self.inst.pnl(self.side, self.entry, px))}

    def on_bar(self, ns, o, h, l, c):
        """Feed one CLOSED bar. Returns an exit dict, or None if still open."""
        if ns > self.exit_ns:                    # we already passed the exit time
            return self._done(TIME, self.last_close)
        if self.side == 1:
            if l <= self.stop:
                return self._done(STOP, o if o <= self.stop else self.stop)
            if h >= self.target + self.thr:
                return self._done(TARGET, self.target)
        else:
            if h >= self.stop:
                return self._done(STOP, o if o >= self.stop else self.stop)
            if l <= self.target - self.thr:
                return self._done(TARGET, self.target)
        if ns == self.exit_ns:
            return self._done(TIME, c)
        self.last_close = c
        return None


def check_no_lookahead(strategy_fn, df: pd.DataFrame, params=None, n_cuts: int = 8, seed: int = 0):
    """Proves a strategy does not peek at the future.

    Runs the strategy on the full data and on truncated copies. Any signal at
    or before the cut must be identical either way - if hiding the future
    changes a past signal, the strategy was using the future.
    """
    full = strategy_fn(df, params)
    if len(df) < 50:
        return True
    rng = np.random.default_rng(seed)
    cuts = sorted(set(rng.integers(len(df) // 10, len(df) - 1, size=n_cuts).tolist()))
    cols = ["idx", "side", "stop_dist", "target_dist"]
    for k in cuts:
        part = strategy_fn(df.iloc[:k + 1], params)
        a = full[full["idx"] <= k][cols].reset_index(drop=True).round(9)
        b = part[part["idx"] <= k][cols].reset_index(drop=True).round(9)
        if len(a) != len(b) or not a.astype(float).equals(b.astype(float)):
            raise AssertionError(
                f"LOOKAHEAD BUG: signals up to bar {k} ({df.index[k]}) change when future bars are hidden. "
                "The strategy is using information from the future. Fix it before trusting any result.")
    return True
