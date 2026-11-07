"""Fake data for testing the TOOLS (not strategies) before real data arrives.

  futures : random-walk 1-minute NQ-like bars, cash session only
  --plant-edge : secretly adds a small push in the direction of the opening-range
                 breakout, so the lab SHOULD detect an edge. Pure noise -> it should not.
  solana  : fake memecoin pools with pump-and-dump shapes

  python scripts/make_synthetic_data.py futures --years 6 --out data/synthetic_nq.parquet
  python scripts/make_synthetic_data.py futures --years 6 --plant-edge --out data/synthetic_nq_edge.parquet
  python scripts/make_synthetic_data.py solana --pools 30 --out data/solana_synthetic
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def futures(years: int, plant_edge: bool, seed: int, start: str = "2015-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=int(252 * years))
    per_day = 390
    n = len(days) * per_day
    sigma = 0.012 / np.sqrt(per_day)
    rets = rng.normal(0, sigma, size=(len(days), per_day))
    rets[:, 0] += rng.normal(0, 0.006, size=len(days))           # overnight gap
    if plant_edge:
        or_move = rets[:, 1:15].sum(1)
        push = np.sign(or_move)[:, None] * sigma * 0.15
        rets[:, 16:120] += push
    log_p = np.log(4000) + np.cumsum(rets.ravel())
    close = np.exp(log_p)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wig = np.abs(rng.normal(0, sigma * 0.6, size=(2, n)))
    high = np.maximum(open_, close) * np.exp(wig[0])
    low = np.minimum(open_, close) * np.exp(-wig[1])
    tick = 0.25
    r = lambda a: np.round(a / tick) * tick
    idx = (pd.DatetimeIndex(np.repeat(days.values, per_day))
           + pd.to_timedelta(np.tile(np.arange(per_day) + 570, len(days)), unit="min")).tz_localize("America/New_York")
    df = pd.DataFrame({"open": r(open_), "high": r(high), "low": r(low), "close": r(close),
                       "volume": rng.integers(50, 2000, size=n).astype(float)}, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "timestamp"
    return df


def solana(pools: int, out: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    t0 = pd.Timestamp("2026-06-01", tz="UTC")
    for k in range(pools):
        n = int(rng.integers(1500, 4000))
        start = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 60)))
        rets = rng.normal(0, 0.02, n)
        for _ in range(int(rng.integers(1, 4))):                     # pumps then dumps
            a = int(rng.integers(60, n - 200))
            rets[a:a + 20] += 0.02
            rets[a + 20:a + 80] -= 0.008
        close = 1e-5 * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wig = np.abs(rng.normal(0, 0.01, size=(2, n)))
        vol = rng.lognormal(6, 1, n) * (1 + 10 * (np.abs(rets) > 0.03))
        df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * np.exp(wig[0]),
                           "low": np.minimum(open_, close) * np.exp(-wig[1]), "close": close, "volume": vol},
                          index=pd.date_range(start, periods=n, freq="min"))
        df.index.name = "timestamp"
        pool = f"SYNTH{k:04d}"
        df.to_parquet(os.path.join(out, f"{pool}.parquet"))
        with open(os.path.join(out, f"{pool}.json"), "w") as f:
            json.dump({"pool": pool, "name": f"FAKE{k}/SOL", "synthetic": True}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["futures", "solana"])
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--pools", type=int, default=30)
    ap.add_argument("--plant-edge", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "futures":
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        futures(a.years, a.plant_edge, a.seed).to_parquet(a.out)
    else:
        solana(a.pools, a.out, a.seed)
    print(f"wrote {a.out}")
