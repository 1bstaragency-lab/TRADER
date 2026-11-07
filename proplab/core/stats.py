"""Plain-English performance numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import NY


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    return float((eq - np.maximum.accumulate(eq)).min())


def summarize(trades: pd.DataFrame, start=None, end=None) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net_$": 0.0, "win_rate_%": np.nan, "profit_factor": np.nan,
                "avg_trade_$": np.nan, "max_drawdown_$": 0.0, "costs_ate_%_of_gross": np.nan,
                "trades_per_year": 0.0, "sharpe": np.nan}
    pnl = trades["pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross = trades["gross"].to_numpy(float).sum()
    start = pd.Timestamp(start) if start is not None else trades["entry_time"].min()
    end = pd.Timestamp(end) if end is not None else trades["entry_time"].max()
    years = max((end - start).days / 365.25, 1 / 365.25)
    days = trades["entry_time"].dt.tz_convert(NY).dt.normalize()
    daily = pd.Series(pnl).groupby(days.to_numpy()).sum()
    all_days = pd.bdate_range(start.tz_convert(NY).normalize().tz_localize(None),
                              end.tz_convert(NY).normalize().tz_localize(None))
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
    daily = daily.groupby(level=0).sum().reindex(all_days, fill_value=0.0)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan
    return {
        "trades": n,
        "net_$": round(float(pnl.sum()), 2),
        "win_rate_%": round(100 * len(wins) / n, 1),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else np.inf,
        "avg_trade_$": round(float(pnl.mean()), 2),
        "avg_win_$": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_$": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "max_drawdown_$": round(max_drawdown(pnl), 2),
        "gross_no_costs_$": round(float(gross), 2),
        "costs_ate_%_of_gross": round(100 * (gross - pnl.sum()) / gross, 1) if gross > 0 else np.nan,
        "trades_per_year": round(n / years, 1),
        "sharpe": round(sharpe, 2),
    }


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    y = trades["entry_time"].dt.tz_convert(NY).dt.year
    g = trades.groupby(y.to_numpy())["pnl"]
    return pd.DataFrame({"trades": g.size(), "net_$": g.sum().round(2),
                         "win_rate_%": (g.apply(lambda s: (s > 0).mean()) * 100).round(1)})
