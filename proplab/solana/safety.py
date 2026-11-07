"""Rug / honeypot screening for Solana tokens. A token must pass EVERY check
before the bot will even watch it.

This lowers the odds of trading an obvious scam. It does NOT make memecoins
safe: developers can still dump, liquidity can still be pulled from pools
whose LP is not burned/locked, and new scam patterns appear constantly.

Uses Solana JSON-RPC. The public endpoint is heavily rate-limited - set
SOLANA_RPC_URL to your own (Helius, Triton, QuickNode...) for real use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import requests

from . import jupiter

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
# Token-2022 extensions that let the issuer tax, block, seize or freeze your tokens
DANGEROUS_EXTENSIONS = {"transferFeeConfig", "transferHook", "permanentDelegate", "nonTransferable",
                        "defaultAccountState", "pausableConfig"}

DEFAULT_RULES = {
    "require_mint_authority_revoked": True,      # nobody can print more supply
    "require_freeze_authority_revoked": True,    # nobody can freeze your tokens (classic honeypot)
    "min_liquidity_usd": 50_000,
    "min_pool_age_min": 30,
    "max_top10_excl_largest_pct": 30,            # largest holder assumed to be the pool itself
    "max_roundtrip_loss_pct": 6.0,               # buy then immediately sell, quoted - catches taxes/thin pools
    "roundtrip_test_usd": 50,
}


@dataclass
class SafetyReport:
    mint: str
    ok: bool = True
    reasons: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def fail(self, why: str):
        self.ok = False
        self.reasons.append(why)


def rpc(method: str, params: list):
    url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    r = requests.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=20)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"RPC {method}: {body['error']}")
    return body["result"]


def check_mint(report: SafetyReport, account: dict, largest: list, rules: dict) -> None:
    """Pure logic, separated from network calls so it can be unit-tested."""
    if not account or not account.get("value"):
        report.fail("mint account not found")
        return
    v = account["value"]
    owner = v.get("owner")
    info = (((v.get("data") or {}).get("parsed") or {}).get("info")) or {}
    report.facts.update({"program": "token-2022" if owner == TOKEN_2022 else "token" if owner == TOKEN_PROGRAM else owner,
                         "mint_authority": info.get("mintAuthority"), "freeze_authority": info.get("freezeAuthority"),
                         "decimals": info.get("decimals"), "supply_raw": info.get("supply")})
    if owner not in (TOKEN_PROGRAM, TOKEN_2022):
        report.fail(f"not an SPL token (owner {owner})")
    if rules["require_mint_authority_revoked"] and info.get("mintAuthority"):
        report.fail("mint authority NOT revoked - supply can be inflated")
    if rules["require_freeze_authority_revoked"] and info.get("freezeAuthority"):
        report.fail("freeze authority NOT revoked - your tokens can be frozen (honeypot risk)")
    exts = {e.get("extension") for e in info.get("extensions", []) or []}
    bad = exts & DANGEROUS_EXTENSIONS
    if bad:
        report.fail(f"dangerous Token-2022 extensions: {sorted(bad)}")

    supply = int(info.get("supply") or 0)
    amounts = sorted((int(a.get("amount") or 0) for a in (largest or {}).get("value", [])), reverse=True)
    if supply > 0 and amounts:
        top10 = 100 * sum(amounts[:10]) / supply
        top10_ex = 100 * sum(amounts[1:11]) / supply
        report.facts.update({"top10_pct": round(top10, 1), "top10_excl_largest_pct": round(top10_ex, 1)})
        if top10_ex > rules["max_top10_excl_largest_pct"]:
            report.fail(f"holders too concentrated: top 10 (excluding largest) own {top10_ex:.0f}%")


def check_pool(report: SafetyReport, pool: dict, rules: dict, now=None) -> None:
    now = now or pd.Timestamp.now(tz="UTC")
    liq = pool.get("liquidity_usd") or 0
    report.facts["liquidity_usd"] = liq
    if liq < rules["min_liquidity_usd"]:
        report.fail(f"liquidity ${liq:,.0f} < ${rules['min_liquidity_usd']:,.0f}")
    created = pool.get("created_at")
    if created is not None:
        age = (now - pd.Timestamp(created)).total_seconds() / 60
        report.facts["pool_age_min"] = round(age)
        if age < rules["min_pool_age_min"]:
            report.fail(f"pool only {age:.0f} min old")


def check_roundtrip(report: SafetyReport, mint: str, decimals: int, rules: dict) -> None:
    usd = rules["roundtrip_test_usd"]
    try:
        buy = jupiter.buy_quote_usd(mint, usd, decimals)
        sell = jupiter.sell_quote_tokens(mint, buy["out_amount"])
    except Exception as exc:
        report.fail(f"could not quote a round trip (maybe unsellable): {exc}")
        return
    loss = 100 * (1 - sell["usd"] / usd)
    report.facts["roundtrip_loss_pct"] = round(loss, 2)
    if loss > rules["max_roundtrip_loss_pct"]:
        report.fail(f"buy-then-sell loses {loss:.1f}% instantly (tax or thin liquidity)")


def screen(pool: dict, rules: dict | None = None) -> SafetyReport:
    rules = {**DEFAULT_RULES, **(rules or {})}
    mint = pool["base_mint"]
    rep = SafetyReport(mint=mint)
    check_pool(rep, pool, rules)
    if not rep.ok:                       # cheap checks first; skip RPC calls for obvious rejects
        return rep
    try:
        acct = rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        largest = rpc("getTokenLargestAccounts", [mint])
    except Exception as exc:
        rep.fail(f"RPC error: {exc}")
        return rep
    check_mint(rep, acct, largest, rules)
    if rep.ok:
        check_roundtrip(rep, mint, int(rep.facts["decimals"]), rules)
    return rep
