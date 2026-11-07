"""Download 1-minute NQ futures bars from Databento. Shows the PRICE FIRST and
asks you to type 'yes' before spending anything.

  1. Make an account at databento.com and create an API key.
  2. Put it in .env as DATABENTO_API_KEY=...   (never paste keys into code or chat)
  3. python scripts/download_databento.py --start 2016-01-01 --end 2026-11-01

Continuous contract NQ.v.0 = always the front month by volume. Prices are NOT
back-adjusted, so there is a price jump at each quarterly roll; intraday
strategies that are flat by the close are barely affected.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NQ.v.0")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/nq_1m.parquet")
    a = ap.parse_args()

    import databento as db
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY missing - add it to .env")
    client = db.Historical(key)
    req = dict(dataset="GLBX.MDP3", symbols=[a.symbol], stype_in="continuous", schema="ohlcv-1m",
               start=a.start, end=a.end)
    cost = client.metadata.get_cost(**req)
    print(f"Databento says this download costs ${cost:,.2f} ({a.symbol}, {a.start} -> {a.end}).")
    if input("Type 'yes' to buy and download: ").strip().lower() != "yes":
        raise SystemExit("cancelled - nothing was charged")
    df = client.timeseries.get_range(**req).to_df()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "timestamp"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out)
    print(f"saved {len(df):,} bars to {a.out}  ({df.index.min()} -> {df.index.max()})")


if __name__ == "__main__":
    main()
