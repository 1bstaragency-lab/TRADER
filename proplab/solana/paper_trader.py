"""Solana memecoin PAPER trader. Places no orders and needs no wallet.

Every minute:
  1. (every few minutes) find new/trending pools -> safety screen -> watchlist
  2. pull the latest CLOSED 1-minute bars for each watched pool
  3. run the SAME strategy code the backtester uses
  4. a signal on the newest closed bar -> paper BUY at a live Jupiter quote
     (the real amount of tokens your $ would get, price impact included)
  5. open positions are valued at a live SELL quote (what you could actually
     get out), and exit on stop / target / time

Results go to forward_test/solana/trades.csv and Discord. Run it for weeks,
then compare its win rate and average trade with the backtest. If paper is much
worse, the backtest costs were too optimistic: raise them and re-test.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import pandas as pd

from .. import notify
from ..strategies import get
from . import jupiter, safety
from .geckoterminal import Gecko


@dataclass
class Position:
    pool: str
    name: str
    mint: str
    decimals: int
    tokens_raw: int
    cost_usd: float
    entry_time: str
    stop_frac: float
    target_frac: float
    exit_time: str
    entry_impact_pct: float


class SolanaPaperTrader:
    def __init__(self, cfg: dict, out_dir: str = "forward_test/solana"):
        self.cfg = cfg
        self.strategy = get(cfg.get("strategy", "sol_momentum"))
        self.params = {**self.strategy.DEFAULT_PARAMS, **(cfg.get("params") or {})}
        self.gecko = Gecko()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.state_path = os.path.join(out_dir, "state.json")
        self.trades_path = os.path.join(out_dir, "trades.csv")
        self.screened: dict[str, dict] = {}     # pool -> {"ok":..., "reasons":..., "at":...}
        self.watch: dict[str, dict] = {}
        self.positions: dict[str, Position] = {}
        self.acted: set[str] = set()            # (pool, signal time) already traded
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                s = json.load(f)
            self.positions = {k: Position(**v) for k, v in s.get("positions", {}).items()}
            self.acted = set(s.get("acted", []))

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump({"positions": {k: asdict(v) for k, v in self.positions.items()},
                       "acted": sorted(self.acted)[-5000:]}, f, indent=1)

    def _record(self, row: dict):
        new = not os.path.exists(self.trades_path)
        with open(self.trades_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---------- steps ----------
    def discover(self):
        c = self.cfg
        cands = []
        for kind in c.get("discover_from", ["trending_pools", "new_pools"]):
            try:
                cands += self.gecko.pools(kind)
            except Exception as exc:
                notify.send(f"[solana-paper] pool list error ({kind}): {exc}")
        quote_ok = {jupiter.SOL, jupiter.USDC}
        for p in cands:
            if not p["pool"] or p["pool"] in self.screened or p["quote_mint"] not in quote_ok:
                continue
            rep = safety.screen(p, c.get("safety"))
            self.screened[p["pool"]] = {"ok": rep.ok, "reasons": rep.reasons, "facts": rep.facts}
            if rep.ok:
                p["decimals"] = int(rep.facts["decimals"])
                self.watch[p["pool"]] = p
                notify.send(f"[solana-paper] watching {p['name']} (liq ${p['liquidity_usd']:,.0f}, "
                            f"round-trip loss {rep.facts.get('roundtrip_loss_pct')}%)")
        # keep the most active pools only
        top = sorted(self.watch.values(), key=lambda p: -p["volume_24h_usd"])[: c.get("watchlist_size", 12)]
        self.watch = {p["pool"]: p for p in top}

    def closed_bars(self, pool: str) -> pd.DataFrame:
        df = self.gecko.ohlcv(pool, limit=max(200, int(self.params.get("lookback", 30)) + 100))
        now = pd.Timestamp.now(tz="UTC")
        return df[df.index + pd.Timedelta(minutes=1) <= now]     # drop the still-forming bar

    def check_signals(self):
        c = self.cfg
        for pool, p in list(self.watch.items()):
            if pool in self.positions or len(self.positions) >= c.get("max_open_positions", 3):
                continue
            try:
                df = self.closed_bars(pool)
            except Exception as exc:
                notify.send(f"[solana-paper] bars error {p['name']}: {exc}")
                continue
            if len(df) < 2:
                continue
            sig = self.strategy.generate_signals(df, self.params)
            sig = sig[sig["idx"] == len(df) - 1]
            if sig.empty:
                continue
            s = sig.iloc[0]
            key = f"{pool}@{df.index[-1].isoformat()}"
            if key in self.acted:
                continue
            self.acted.add(key)
            self.open(p, s, float(df["close"].iloc[-1]))

    def open(self, p: dict, s, ref_close: float):
        usd = self.cfg.get("notional_usd", 25)
        try:
            q = jupiter.buy_quote_usd(p["base_mint"], usd, p["decimals"], self.cfg.get("slippage_bps", 300))
        except Exception as exc:
            notify.send(f"[solana-paper] buy quote failed {p['name']}: {exc}")
            return
        cost = usd + self.cfg.get("tx_cost_usd", 0.05)
        pos = Position(pool=p["pool"], name=p["name"], mint=p["base_mint"], decimals=p["decimals"],
                       tokens_raw=q["out_amount"], cost_usd=cost,
                       entry_time=pd.Timestamp.now(tz="UTC").isoformat(),
                       stop_frac=float(s["stop_dist"]) / ref_close, target_frac=float(s["target_dist"]) / ref_close,
                       exit_time=pd.Timestamp(s["exit_time"]).isoformat(), entry_impact_pct=q["price_impact_pct"])
        self.positions[p["pool"]] = pos
        self._save()
        notify.send(f"[solana-paper] BUY {p['name']} ${usd} -> {q['tokens']:.4g} tokens "
                    f"(impact {q['price_impact_pct']:.2f}%, stop -{pos.stop_frac:.0%}, target +{pos.target_frac:.0%})")

    def manage(self):
        now = pd.Timestamp.now(tz="UTC")
        for pool, pos in list(self.positions.items()):
            try:
                q = jupiter.sell_quote_tokens(pos.mint, pos.tokens_raw, self.cfg.get("slippage_bps", 300))
            except Exception as exc:
                notify.send(f"[solana-paper] PROBLEM: cannot quote a sell for {pos.name}: {exc}")
                continue
            value = q["usd"] - self.cfg.get("tx_cost_usd", 0.05)
            ret = value / pos.cost_usd - 1
            kind = ("stop" if ret <= -pos.stop_frac else "target" if ret >= pos.target_frac
                    else "time" if now >= pd.Timestamp(pos.exit_time) else None)
            if kind is None:
                continue
            row = {"pool": pool, "name": pos.name, "entry_time": pos.entry_time, "exit_time": now.isoformat(),
                   "kind": kind, "cost_usd": round(pos.cost_usd, 4), "exit_value_usd": round(value, 4),
                   "pnl_usd": round(value - pos.cost_usd, 4), "return_%": round(100 * ret, 2),
                   "entry_impact_%": round(pos.entry_impact_pct, 3), "exit_impact_%": round(q["price_impact_pct"], 3)}
            self._record(row)
            del self.positions[pool]
            self._save()
            notify.send(f"[solana-paper] SELL {pos.name} ({kind}) {ret:+.1%} = ${row['pnl_usd']:+.2f}")

    def run(self):
        c = self.cfg
        notify.send(f"[solana-paper] started - PAPER ONLY, strategy {c.get('strategy', 'sol_momentum')}")
        last_discover = 0.0
        while True:
            start = time.time()
            try:
                if start - last_discover > 60 * c.get("discover_every_min", 10):
                    self.discover()
                    last_discover = start
                self.manage()
                self.check_signals()
            except Exception as exc:
                notify.send(f"[solana-paper] PROBLEM: {exc!r}")
            time.sleep(max(5.0, c.get("poll_seconds", 60) - (time.time() - start)))
