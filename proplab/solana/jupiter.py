"""Jupiter swap quotes - the REAL price you would get for a given size,
including price impact. Used for paper fills and the round-trip safety check.
No order is ever sent from this module.

Endpoint is configurable because Jupiter moves it from time to time:
  JUPITER_QUOTE_URL (default https://lite-api.jup.ag/swap/v1/quote)
"""
from __future__ import annotations

import os

import requests

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL = "So11111111111111111111111111111111111111112"
USDC_DECIMALS = 6


def parse_quote(q: dict) -> dict:
    return {
        "in_amount": int(q["inAmount"]),
        "out_amount": int(q["outAmount"]),
        "price_impact_pct": float(q.get("priceImpactPct") or 0) * 100,
        "route": [((leg.get("swapInfo") or {}).get("label")) for leg in q.get("routePlan", [])],
    }


def quote(input_mint: str, output_mint: str, amount_raw: int, slippage_bps: int = 100) -> dict:
    url = os.getenv("JUPITER_QUOTE_URL", "https://lite-api.jup.ag/swap/v1/quote")
    headers = {"x-api-key": os.environ["JUPITER_API_KEY"]} if os.getenv("JUPITER_API_KEY") else {}
    r = requests.get(url, params={"inputMint": input_mint, "outputMint": output_mint,
                                  "amount": int(amount_raw), "slippageBps": slippage_bps},
                     headers=headers, timeout=15)
    r.raise_for_status()
    return parse_quote(r.json())


def buy_quote_usd(mint: str, usd: float, decimals: int, slippage_bps: int = 100) -> dict:
    """How many tokens $usd of USDC buys right now."""
    q = quote(USDC, mint, int(usd * 10 ** USDC_DECIMALS), slippage_bps)
    q["tokens"] = q["out_amount"] / 10 ** decimals
    q["effective_price"] = usd / q["tokens"] if q["tokens"] else float("inf")
    return q


def sell_quote_tokens(mint: str, tokens_raw: int, slippage_bps: int = 100) -> dict:
    """How many USDC you would get for selling these tokens right now."""
    q = quote(mint, USDC, tokens_raw, slippage_bps)
    q["usd"] = q["out_amount"] / 10 ** USDC_DECIMALS
    return q
