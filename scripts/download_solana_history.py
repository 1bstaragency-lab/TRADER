"""Build a Solana memecoin dataset from GeckoTerminal (free, no key).

  python scripts/download_solana_history.py --pools 40 --bars 10000

Saves data/solana/<pool>.parquet (1-minute bars) + <pool>.json (pool info).
Takes a while: the free API allows ~30 requests per minute.
Tip: mix sources (--sources pools,trending_pools,new_pools) to reduce survivorship bias,
and re-run every few days so coins that later die stay in your dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proplab.solana.geckoterminal import Gecko  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/solana")
    ap.add_argument("--pools", type=int, default=40)
    ap.add_argument("--bars", type=int, default=10_000)
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--sources", default="pools,trending_pools,new_pools")
    ap.add_argument("--min-liquidity", type=float, default=20_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = Gecko()
    seen, cands = set(), []
    for src in a.sources.split(","):
        for page in range(1, a.pages + 1):
            for p in g.pools(src, page):
                if p["pool"] and p["pool"] not in seen and p["liquidity_usd"] >= a.min_liquidity:
                    seen.add(p["pool"])
                    cands.append(p)
    print(f"{len(cands)} candidate pools; downloading up to {a.pools}")
    for p in cands[: a.pools]:
        path = os.path.join(a.out, f"{p['pool']}.parquet")
        df = g.ohlcv_history(p["pool"], a.bars)
        if os.path.exists(path):                      # keep older bars we already have
            import pandas as pd
            old = pd.read_parquet(path)
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        if len(df) < 200:
            continue
        df.to_parquet(path)
        with open(path[:-8] + ".json", "w") as f:
            json.dump({**p, "created_at": str(p["created_at"])}, f, indent=1)
        print(f"  {p['name']:<30} {len(df):>6} bars  liq ${p['liquidity_usd']:,.0f}")


if __name__ == "__main__":
    main()
