"""Risk gate: every order gets PASS or BLOCK before it can happen.

Idea borrowed from the per-order risk check in Jakeschincariol/day-trader-skill
(MIT), re-implemented here for futures and Solana. The paper traders already
use it; the live trader (phase 2) must use it too, with no bypass.

Also: position sizing from "how much am I willing to lose on this trade".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import pandas as pd
import yaml


@dataclass
class RiskLimits:
    max_loss_per_trade_usd: float = 200.0
    daily_loss_limit_usd: float = 500.0      # set this BELOW your prop firm's daily limit
    max_trades_per_day: int = 3
    max_contracts: int = 5
    max_open_positions: int = 1
    entry_window: tuple = ("09:30", "15:30")  # New York time
    flat_by: str = "15:55"
    min_reward_risk: float = 1.0
    timezone: str = "America/New_York"

    @classmethod
    def from_yaml(cls, path: str) -> "RiskLimits":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        if "entry_window" in raw:
            raw["entry_window"] = tuple(raw["entry_window"])
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class DayState:
    day: object = None
    realized_pnl: float = 0.0
    trades: int = 0
    open_positions: int = 0
    halted: bool = False

    def roll(self, ts: pd.Timestamp, tz: str):
        d = ts.tz_convert(tz).date()
        if d != self.day:
            self.day, self.realized_pnl, self.trades, self.halted = d, 0.0, 0, False

    def on_exit(self, pnl: float, limits: RiskLimits):
        self.realized_pnl += pnl
        self.open_positions = max(0, self.open_positions - 1)
        if self.realized_pnl <= -limits.daily_loss_limit_usd:
            self.halted = True


def contracts_for_risk(max_loss_usd: float, stop_dist_pts: float, inst, cap: int) -> int:
    """How many contracts keep the loss at the stop (incl. costs and stop slippage) under max_loss_usd."""
    per = (stop_dist_pts + inst.tick * (inst.slip_ticks[0] + inst.slip_ticks[1])) * inst.point_value \
        + 2 * inst.commission_per_side
    return max(0, min(cap, math.floor(max_loss_usd / per))) if per > 0 else 0


def check_order(limits: RiskLimits, state: DayState, ts: pd.Timestamp, qty: int,
                loss_at_stop_usd: float, reward_usd: float) -> tuple[bool, list[str]]:
    """Returns (PASS?, reasons). Every reason is a rule that BLOCKED the order."""
    state.roll(ts, limits.timezone)
    local = ts.tz_convert(limits.timezone)
    hm = f"{local.hour:02d}:{local.minute:02d}"
    why = []
    if state.halted or state.realized_pnl <= -limits.daily_loss_limit_usd:
        why.append(f"daily loss limit hit (${state.realized_pnl:,.2f})")
    if state.trades >= limits.max_trades_per_day:
        why.append(f"max {limits.max_trades_per_day} trades/day reached")
    if state.open_positions >= limits.max_open_positions:
        why.append("already at max open positions")
    if qty <= 0:
        why.append("size is 0: the stop is too wide for the max loss per trade")
    if qty > limits.max_contracts:
        why.append(f"size {qty} > max {limits.max_contracts}")
    if loss_at_stop_usd > limits.max_loss_per_trade_usd + 1e-9:
        why.append(f"loss at stop ${loss_at_stop_usd:,.2f} > max ${limits.max_loss_per_trade_usd:,.2f}")
    if state.realized_pnl - loss_at_stop_usd < -limits.daily_loss_limit_usd - 1e-9:
        why.append("a full stop-out would break the daily loss limit")
    if not (limits.entry_window[0] <= hm < limits.entry_window[1]) or hm >= limits.flat_by:
        why.append(f"{hm} is outside the entry window {limits.entry_window[0]}-{limits.entry_window[1]}")
    if loss_at_stop_usd > 0 and reward_usd / loss_at_stop_usd < limits.min_reward_risk:
        why.append(f"reward/risk {reward_usd / loss_at_stop_usd:.2f} < {limits.min_reward_risk}")
    return (not why), why
