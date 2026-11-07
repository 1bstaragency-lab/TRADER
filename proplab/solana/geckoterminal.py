"""GeckoTerminal public API (free, no key, ~30 requests/minute).
Pool lists and 1-minute OHLCV for Solana DEX pools.
Docs: https://www.geckoterminal.com/dex-api
"""
from __future__ import annotations

import time

import pandas as pd
import requests

BASE = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json;version=20230302", "User-Agent": "prop-lab/0.1"}


def _mint(rel: dict, key: str) -> str | None:
    rid = (((rel or {}).get(key) or {}).get("data") or {}).get("id") or ""
    return rid.split("_", 1)[1] if "_" in rid else None


def parse_pools(payload: dict) -> list[dict]:
    out = []
    for item in payload.get("data", []):
        at, rel = item.get("attributes", {}), item.get("relationships", {})
        out.append({
            "pool": at.get("address"),
            "name": at.get("name"),
            "base_mint": _mint(rel, "base_token"),
            "quote_mint": _mint(rel, "quote_token"),
            "dex": (((rel.get("dex") or {}).get("data")) or {}).get("id"),
            "liquidity_usd": float(at.get("reserve_in_usd") or 0),
            "price_usd": float(at.get("base_token_price_usd") or 0),
            "volume_24h_usd": float(((at.get("volume_usd") or {}).get("h24")) or 0),
            "created_at": pd.Timestamp(at["pool_created_at"]) if at.get("pool_created_at") else None,
        })
    return out


def parse_ohlcv(payload: dict) -> pd.DataFrame:
    rows = ((payload.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    df.index.name = "timestamp"
    return df.astype(float).sort_index()


class Gecko:
    def __init__(self, min_interval_s: float = 2.2):
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(5):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.s.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:                  # rate limited: back off
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GeckoTerminal kept rate-limiting {path}")

    def pools(self, kind: str = "new_pools", page: int = 1) -> list[dict]:
        """kind: new_pools | trending_pools | pools (top by volume)"""
        return parse_pools(self._get(f"/networks/solana/{kind}", {"page": page}))

    def ohlcv(self, pool: str, limit: int = 1000, before_ts: int | None = None) -> pd.DataFrame:
        params = {"aggregate": 1, "limit": min(limit, 1000), "currency": "usd", "token": "base"}
        if before_ts:
            params["before_timestamp"] = int(before_ts)
        return parse_ohlcv(self._get(f"/networks/solana/pools/{pool}/ohlcv/minute", params))

    def ohlcv_history(self, pool: str, max_bars: int = 20_000) -> pd.DataFrame:
        parts, before = [], None
        while sum(len(p) for p in parts) < max_bars:
            df = self.ohlcv(pool, 1000, before)
            if df.empty:
                break
            parts.append(df)
            before = int(df.index.min().timestamp())
            if len(df) < 1000:
                break
        if not parts:
            return parse_ohlcv({})
        out = pd.concat(parts).sort_index()
        return out[~out.index.duplicated(keep="last")]
