"""Load bars from Parquet or CSV into the one format every tool uses:
index = timezone-aware bar OPEN time; columns open, high, low, close, volume."""
from __future__ import annotations

import pandas as pd


def load_bars(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        col = next(c for c in ("timestamp", "ts_event", "time", "datetime", "date") if c in df.columns)
        df = df.set_index(pd.to_datetime(df.pop(col), utc=True))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)
    df.index.name = "timestamp"
    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
