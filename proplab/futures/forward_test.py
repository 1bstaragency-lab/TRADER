"""Futures forward tester ("the dealer"). PAPER ONLY - it never places an order.

Watches 1-minute NQ bars as they close, runs the SAME strategy code and the
SAME fill rules as the backtester (engine.PaperPosition), and logs every
trade the strategy would have taken today, with Discord alerts.

Feeds:
  ReplayFeed        - replays a parquet/csv file; use it to test the plumbing
  DatabentoLiveFeed - live CME bars from Databento (needs DATABENTO_API_KEY
                      and a live-data subscription). VERIFY ON FIRST RUN: this
                      code follows Databento's documented Live API but could not
                      be tested from the build environment.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import pandas as pd

from .. import notify
from ..core.engine import NY, PaperPosition, to_ns
from ..core.risk import DayState, RiskLimits, check_order, contracts_for_risk


@dataclass
class Bar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class ReplayFeed:
    def __init__(self, df: pd.DataFrame, start=None):
        self.df = df if start is None else df[df.index >= pd.Timestamp(start, tz=NY)]

    def __iter__(self):
        for ts, r in self.df.iterrows():
            yield Bar(ts, r["open"], r["high"], r["low"], r["close"], r.get("volume", 0.0))


class DatabentoLiveFeed:
    def __init__(self, symbol: str = "NQ.v.0", dataset: str = "GLBX.MDP3", stop_at: str = "16:05"):
        self.symbol, self.dataset, self.stop_at = symbol, dataset, stop_at

    def __iter__(self):
        import databento as db
        client = db.Live(key=os.environ["DATABENTO_API_KEY"])
        client.subscribe(dataset=self.dataset, schema="ohlcv-1m", stype_in="continuous", symbols=[self.symbol])
        h, m = map(int, self.stop_at.split(":"))
        for rec in client:
            if not isinstance(rec, db.OHLCVMsg):
                continue
            ts = pd.Timestamp(rec.ts_event, unit="ns", tz="UTC").tz_convert(NY)
            px = 1e-9   # Databento prices are fixed-point integers (1e-9 units)
            yield Bar(ts, rec.open * px, rec.high * px, rec.low * px, rec.close * px, float(rec.volume))
            if (ts.hour, ts.minute) >= (h, m):
                client.stop()
                return


class ForwardTester:
    def __init__(self, strategy, params: dict, inst, history: pd.DataFrame | None = None,
                 out_dir: str = "forward_test/futures", qty: int = 1, keep_days: int = 5, label: str = "dealer",
                 risk: RiskLimits | None = None, size_by_risk: bool = False):
        """risk: optional RiskLimits gate - every entry must PASS it.
        size_by_risk: size each trade from risk.max_loss_per_trade_usd instead of a fixed qty."""
        self.strategy, self.params, self.inst, self.qty, self.label = strategy, params, inst, qty, label
        self.risk, self.size_by_risk, self.day = risk, size_by_risk, DayState()
        self.blocked: list[dict] = []
        self.df = history.copy() if history is not None else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz=NY))
        self.keep_days = keep_days
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.trades_path = os.path.join(out_dir, "trades.csv")
        self.bars_path = os.path.join(out_dir, "bars.csv")
        self.cur_qty = qty
        self.pending = None      # signal waiting for the next bar's open
        self.pos = None
        self.pos_meta = None
        self.trades: list[dict] = []

    def _append_bar(self, b: Bar):
        row = pd.DataFrame({"open": [b.open], "high": [b.high], "low": [b.low], "close": [b.close],
                            "volume": [b.volume]}, index=pd.DatetimeIndex([b.ts]).tz_convert(NY))
        self.df = pd.concat([self.df, row]) if len(self.df) else row
        cutoff = b.ts - pd.Timedelta(days=self.keep_days)
        self.df = self.df[self.df.index >= cutoff]
        new = not os.path.exists(self.bars_path)
        row.to_csv(self.bars_path, mode="a", header=new, index_label="timestamp")

    def on_bar(self, b: Bar):
        if len(self.df) and b.ts <= self.df.index[-1]:
            return                              # duplicate / out-of-order bar
        self._append_bar(b)
        ns = int(to_ns([b.ts])[0])

        if self.pending is not None and self.pos is None:
            s = self.pending
            self.pending = None
            exit_ns = int(to_ns([s["exit_time"]])[0])
            if ns <= exit_ns and self._risk_ok(b, s):
                self.pos = PaperPosition(s["side"], b.open, s["stop_dist"], s["target_dist"], exit_ns, self.inst)
                self.pos_meta = {"entry_time": b.ts, "side": int(s["side"]), "qty": self.cur_qty}
                notify.send(f"[{self.label}] PAPER ENTRY {'LONG' if s['side'] == 1 else 'SHORT'} {self.cur_qty} @ "
                            f"{self.pos.entry:.2f} stop {self.pos.stop:.2f} target {self.pos.target:.2f} ({b.ts:%H:%M})")

        if self.pos is not None:
            out = self.pos.on_bar(ns, b.open, b.high, b.low, b.close)
            if out is not None:
                self._close(b, out)

        sig = self.strategy.generate_signals(self.df, self.params)
        if len(sig):
            last = sig[sig["idx"] == len(self.df) - 1]
            if len(last) and self.pos is None and self.pending is None:
                self.pending = last.iloc[0].to_dict()

    def _risk_ok(self, b: Bar, s: dict) -> bool:
        inst = self.inst
        qty = self.qty
        if self.risk is not None and self.size_by_risk:
            qty = contracts_for_risk(self.risk.max_loss_per_trade_usd, s["stop_dist"], inst, self.risk.max_contracts)
        self.cur_qty = qty
        if self.risk is None:
            return True
        cost = 2 * inst.commission_per_side * qty
        loss = (s["stop_dist"] + inst.tick * (inst.slip_ticks[0] + inst.slip_ticks[1])) * inst.point_value * qty + cost
        reward = s["target_dist"] * inst.point_value * qty - cost
        ok, why = check_order(self.risk, self.day, b.ts, qty, loss, reward)
        if not ok:
            self.blocked.append({"time": b.ts.isoformat(), "reasons": why})
            notify.send(f"[{self.label}] RISK BLOCK ({b.ts:%H:%M}): " + "; ".join(why))
            return False
        self.day.trades += 1
        self.day.open_positions += 1
        return True

    def _close(self, b: Bar, out: dict):
        row = {"entry_time": self.pos_meta["entry_time"].isoformat(), "exit_time": b.ts.isoformat(),
               "side": self.pos_meta["side"], "qty": self.pos_meta["qty"], "entry": round(self.pos.entry, 2),
               "exit": round(out["exit"], 2), "kind": out["kind"], "pnl_$": round(out["pnl"] * self.pos_meta["qty"], 2)}
        if self.risk is not None:
            self.day.on_exit(row["pnl_$"], self.risk)
        self.trades.append(row)
        new = not os.path.exists(self.trades_path)
        with open(self.trades_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        notify.send(f"[{self.label}] PAPER EXIT ({out['kind']}) @ {out['exit']:.2f}  P&L ${row['pnl_$']:+.2f}")
        self.pos = None
        self.pos_meta = None

    def run(self, feed):
        notify.send(f"[{self.label}] forward test started - PAPER ONLY")
        try:
            for b in feed:
                self.on_bar(b)
        except Exception as exc:
            notify.send(f"[{self.label}] PROBLEM: {exc!r}")
            raise
        day_pnl = sum(t["pnl_$"] for t in self.trades)
        notify.send(f"[{self.label}] session done: {len(self.trades)} trades, P&L ${day_pnl:+.2f}")
