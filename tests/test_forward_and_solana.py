import pandas as pd
import pytest

from proplab.core.engine import Bars, backtest
from proplab.core.instruments import Futures
from proplab.futures.forward_test import ForwardTester, ReplayFeed
from proplab.solana import safety
from proplab.solana.geckoterminal import parse_ohlcv, parse_pools
from proplab.solana.jupiter import parse_quote
from proplab.strategies import futures_orb
from scripts.make_synthetic_data import futures as make_futures


def test_forward_tester_reproduces_backtest_exactly(tmp_path, monkeypatch):
    """Paper trading on replayed bars must produce the same trades as the backtest."""
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    df = make_futures(years=1, plant_edge=True, seed=5).iloc[: 390 * 40]
    inst = Futures()
    bt = backtest(Bars(df), futures_orb.generate_signals(df), inst)
    ft = ForwardTester(futures_orb, futures_orb.DEFAULT_PARAMS, inst, out_dir=str(tmp_path), keep_days=2)
    ft.run(ReplayFeed(df))
    assert len(ft.trades) == len(bt) > 10
    for paper, (_, real) in zip(ft.trades, bt.iterrows()):
        assert paper["kind"] == real["kind"]
        assert paper["pnl_$"] == pytest.approx(real["pnl"], abs=0.01)


def test_gecko_parsers():
    pools = parse_pools({"data": [{
        "attributes": {"address": "POOL1", "name": "BONKX / SOL", "reserve_in_usd": "123456.7",
                       "base_token_price_usd": "0.0001", "volume_usd": {"h24": "999"},
                       "pool_created_at": "2026-11-01T00:00:00Z"},
        "relationships": {"base_token": {"data": {"id": "solana_MINT1"}},
                          "quote_token": {"data": {"id": "solana_So11111111111111111111111111111111111111112"}},
                          "dex": {"data": {"id": "raydium"}}}}]})
    assert pools[0]["base_mint"] == "MINT1" and pools[0]["liquidity_usd"] == pytest.approx(123456.7)
    df = parse_ohlcv({"data": {"attributes": {"ohlcv_list": [[1700000060, 2, 3, 1, 2.5, 10],
                                                             [1700000000, 1, 2, 1, 2, 5]]}}})
    assert list(df["close"]) == [2.0, 2.5] and str(df.index.tz) == "UTC"


def test_jupiter_parser():
    q = parse_quote({"inAmount": "1000000", "outAmount": "52345", "priceImpactPct": "0.0123",
                     "routePlan": [{"swapInfo": {"label": "Raydium"}}]})
    assert q["out_amount"] == 52345 and q["price_impact_pct"] == pytest.approx(1.23)


def _acct(mint_auth=None, freeze=None, owner=safety.TOKEN_PROGRAM, ext=None):
    info = {"mintAuthority": mint_auth, "freezeAuthority": freeze, "decimals": 6, "supply": "1000000"}
    if ext:
        info["extensions"] = [{"extension": e} for e in ext]
    return {"value": {"owner": owner, "data": {"parsed": {"info": info}}}}


def test_safety_rejects_classic_scams():
    rules = safety.DEFAULT_RULES
    spread = {"value": [{"amount": "300000"}] + [{"amount": "10000"}] * 15}
    good = safety.SafetyReport("M")
    safety.check_mint(good, _acct(), spread, rules)
    assert good.ok, good.reasons

    for acct, why in [(_acct(mint_auth="DEV"), "mint authority"), (_acct(freeze="DEV"), "freeze"),
                      (_acct(owner=safety.TOKEN_2022, ext=["transferFeeConfig"]), "Token-2022")]:
        r = safety.SafetyReport("M")
        safety.check_mint(r, acct, spread, rules)
        assert not r.ok and any(why in x for x in r.reasons)

    whales = {"value": [{"amount": "300000"}] + [{"amount": "60000"}] * 10}
    r = safety.SafetyReport("M")
    safety.check_mint(r, _acct(), whales, rules)
    assert not r.ok and "concentrated" in r.reasons[0]

    r = safety.SafetyReport("M")
    safety.check_pool(r, {"liquidity_usd": 5000, "created_at": pd.Timestamp.now(tz="UTC")}, rules)
    assert not r.ok and len(r.reasons) == 2
