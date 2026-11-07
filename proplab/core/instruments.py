"""What a trade costs, per instrument.

Every fill goes through `fill()`, which always moves the price AGAINST you
(you buy higher, you sell lower). There is no code path that fills for free.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Fill kinds. Each kind can carry a different amount of slippage.
ENTRY, STOP, TARGET, TIME = 0, 1, 2, 3
KIND_NAMES = {ENTRY: "entry", STOP: "stop", TARGET: "target", TIME: "time"}


@dataclass(frozen=True)
class Futures:
    """A futures contract such as MNQ or NQ. Prices are in index points."""

    name: str = "MNQ"
    tick: float = 0.25
    point_value: float = 2.0            # $ per 1.00 point per contract (MNQ = $2, NQ = $20)
    commission_per_side: float = 0.62   # all-in $ per contract per side (broker + exchange + NFA)
    # Slippage in ticks for: entry (market), stop (market), target (limit), time exit (market)
    slip_ticks: tuple = (1.0, 2.0, 0.0, 1.0)
    can_short: bool = True

    def slip(self, price, kind):
        return np.asarray(self.slip_ticks, dtype=float)[kind] * self.tick + 0.0 * np.asarray(price)

    def through(self, price):
        """A limit target only counts as filled if price trades THROUGH it by a tick."""
        return np.full_like(np.asarray(price, dtype=float), self.tick)

    def pnl(self, side, entry_fill, exit_fill, qty=1.0):
        return side * (exit_fill - entry_fill) * self.point_value * qty - 2 * self.commission_per_side * qty

    def gross(self, side, raw_entry, raw_exit, qty=1.0):
        """P&L with zero costs - used only to show how much the costs ate."""
        return side * (raw_exit - raw_entry) * self.point_value * qty

    def to_dollars(self, price_move, entry_price, qty=1.0):
        return price_move * self.point_value * qty

    def scaled_costs(self, mult: float) -> "Futures":
        return replace(self, commission_per_side=self.commission_per_side * mult,
                       slip_ticks=tuple(s * mult for s in self.slip_ticks))


@dataclass(frozen=True)
class Spot:
    """A spot token bought with a fixed dollar amount (e.g. a Solana memecoin). Long only."""

    name: str = "solana-memecoin"
    notional_usd: float = 100.0
    fee_frac: float = 0.0025            # pool/LP fee per swap (0.25%; many memecoin pools charge 1%+)
    # Slippage as a fraction of price for entry, stop, target, time exit
    slip_frac: tuple = (0.01, 0.02, 0.01, 0.01)
    fixed_usd_per_tx: float = 0.05      # network + priority fee + tip, per swap
    can_short: bool = False

    def slip(self, price, kind):
        return np.asarray(self.slip_frac, dtype=float)[kind] * np.asarray(price, dtype=float)

    def through(self, price):
        return np.zeros_like(np.asarray(price, dtype=float))

    def pnl(self, side, entry_fill, exit_fill, qty=1.0):
        notional = self.notional_usd * qty
        units = notional * (1 - self.fee_frac) / entry_fill
        proceeds = units * exit_fill * (1 - self.fee_frac)
        return proceeds - notional - 2 * self.fixed_usd_per_tx * qty

    def gross(self, side, raw_entry, raw_exit, qty=1.0):
        return self.notional_usd * qty * (raw_exit / raw_entry - 1)

    def to_dollars(self, price_move, entry_price, qty=1.0):
        return self.notional_usd * qty * price_move / entry_price

    def scaled_costs(self, mult: float) -> "Spot":
        return replace(self, fee_frac=self.fee_frac * mult, fixed_usd_per_tx=self.fixed_usd_per_tx * mult,
                       slip_frac=tuple(s * mult for s in self.slip_frac))


def load_instrument(cfg: dict):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if "slip_ticks" in cfg:
        s = cfg["slip_ticks"]
        cfg["slip_ticks"] = (s["entry"], s["stop"], s["target"], s["time"])
    if "slip_frac" in cfg:
        s = cfg["slip_frac"]
        cfg["slip_frac"] = (s["entry"], s["stop"], s["target"], s["time"])
    return Futures(**cfg) if kind == "futures" else Spot(**cfg)
