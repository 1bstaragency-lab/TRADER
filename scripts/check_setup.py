"""Checks that everything is installed and connected. Safe: places no orders,
spends no money (the Databento check only asks for a PRICE, it buys nothing).

  python scripts/check_setup.py            # everything
  python scripts/check_setup.py --no-discord

Each line is PASS, FAIL (with how to fix it) or SKIP (not configured yet).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS: list[tuple[str, str, str]] = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[{status:<4}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def check(name, fix):
    def deco(fn):
        def run(*a, **k):
            try:
                out = fn(*a, **k)
                if out is None:
                    return
                status, detail = out
                record(name, status, detail)
            except Exception as exc:
                record(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}  -> FIX: {fix}")
                if os.getenv("CHECK_DEBUG"):
                    traceback.print_exc()
        return run
    return deco


@check("Python version", "install Python 3.11 or newer")
def python_version():
    ok = sys.version_info >= (3, 10)
    return ("PASS" if ok else "FAIL"), sys.version.split()[0]


@check("Python packages", "run: pip install -r requirements.txt")
def packages():
    missing = []
    for mod in ("numpy", "pandas", "pyarrow", "yaml", "requests", "dotenv"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError("missing " + ", ".join(missing))
    return "PASS", "numpy, pandas, pyarrow, pyyaml, requests, python-dotenv"


@check(".env file", "run: cp .env.example .env   then fill in your keys")
def env_file():
    if not os.path.exists(".env"):
        raise FileNotFoundError("no .env in this folder (run this from the repo root)")
    keys = ["DATABENTO_API_KEY", "DISCORD_WEBHOOK_URL", "SOLANA_RPC_URL"]
    have = [k for k in keys if os.getenv(k)]
    return "PASS", "set: " + (", ".join(have) if have else "none yet")


@check("Backtest engine", "run: pytest   and send the output to Claude")
def engine():
    import pandas as pd

    from proplab.core.engine import Bars, backtest
    from proplab.core.instruments import Futures
    idx = pd.date_range("2024-01-02 09:30", periods=5, freq="min", tz="America/New_York")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    sig = pd.DataFrame({"idx": [0], "side": [1], "stop_dist": [10.0], "target_dist": [10.0], "exit_time": [idx[3]]})
    pnl = backtest(Bars(df), sig, Futures())["pnl"].iloc[0]
    if abs(pnl + 2.24) > 1e-9:
        raise AssertionError(f"flat trade should lose exactly $2.24 in costs, got {pnl}")
    return "PASS", "a flat trade loses exactly its costs"


@check("Discord alerts", "in Discord: Server Settings > Integrations > Webhooks > New Webhook > Copy URL, put it in .env")
def discord(enabled: bool):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return "SKIP", "DISCORD_WEBHOOK_URL not set"
    if not enabled:
        return "SKIP", "--no-discord"
    import requests
    r = requests.post(url, json={"content": "prop-lab setup check: Discord alerts work."}, timeout=10)
    r.raise_for_status()
    return "PASS", "test message sent - check your channel"


@check("GeckoTerminal (Solana prices)", "check your internet / firewall; the free API allows ~30 requests/min, wait a minute and retry")
def gecko():
    from proplab.solana.geckoterminal import Gecko
    g = Gecko()
    pools = [p for p in g.pools("trending_pools") if p["pool"]]
    if not pools:
        raise RuntimeError("no pools returned")
    bars = g.ohlcv(pools[0]["pool"], limit=50)
    if bars.empty:
        raise RuntimeError(f"no 1-minute bars for {pools[0]['name']}")
    return "PASS", f"{len(pools)} trending pools; {len(bars)} bars for {pools[0]['name']}"


@check("Solana RPC (token safety checks)", "set SOLANA_RPC_URL in .env to a private RPC (free tiers: Helius, QuickNode, Alchemy)")
def solana_rpc():
    from proplab.solana import safety
    from proplab.solana.jupiter import USDC
    acct = safety.rpc("getAccountInfo", [USDC, {"encoding": "jsonParsed"}])
    dec = acct["value"]["data"]["parsed"]["info"]["decimals"]
    if dec != 6:
        raise AssertionError(f"USDC decimals should be 6, got {dec}")
    public = "api.mainnet-beta.solana.com" in os.getenv("SOLANA_RPC_URL", "api.mainnet-beta.solana.com")
    return "PASS", "connected" + (" (public RPC - fine for testing, use a private one for the bot)" if public else "")


@check("Jupiter quotes (paper fills)", "Jupiter may have moved its endpoint: set JUPITER_QUOTE_URL in .env (see dev.jup.ag docs)")
def jup():
    from proplab.solana.jupiter import SOL, USDC, quote
    q = quote(USDC, SOL, 10_000_000)           # quote for 10 USDC -> SOL; nothing is sent
    sol = q["out_amount"] / 1e9
    if sol <= 0:
        raise AssertionError("quote returned 0")
    return "PASS", f"10 USDC -> {sol:.4f} SOL (implied SOL price ${10 / sol:,.2f})"


@check("Databento (futures data)", "check DATABENTO_API_KEY in .env (databento.com > API keys); pip install databento")
def databento():
    if not os.getenv("DATABENTO_API_KEY"):
        return "SKIP", "DATABENTO_API_KEY not set"
    import databento as db
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    cost = client.metadata.get_cost(dataset="GLBX.MDP3", symbols=["NQ.v.0"], stype_in="continuous",
                                    schema="ohlcv-1m", start="2025-01-06", end="2025-01-07")
    return "PASS", f"key works; 1 day of NQ 1-minute bars would cost ${cost:.4f} (nothing bought)"


@check("Synthetic lab run", "run: pytest   and send the output to Claude")
def lab_run():
    import tempfile

    import run_futures_lab
    from scripts.make_synthetic_data import futures
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "noise.parquet")
        futures(years=3, plant_edge=False, seed=11, start="2018-01-02").to_parquet(p)
        devnull = open(os.devnull, "w")
        old, sys.stdout = sys.stdout, devnull
        try:
            v = run_futures_lab.main(["--data", p, "--is-end", "2019-06-30", "--holdout-months", "6",
                                      "--flips", "50", "--prop-random-sets", "3", "--prop-start-step", "5",
                                      "--log", os.path.join(d, "log.jsonl")])
        finally:
            sys.stdout = old
            devnull.close()
    if v != "DOES NOT TRADE":
        return "FAIL", f"pure noise data was rated '{v}' - tell Claude, something is wrong"
    return "PASS", "pure noise correctly rejected"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discord", action="store_true")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    print("prop-lab setup check (no orders, no purchases)\n")
    python_version()
    packages()
    env_file()
    engine()
    lab_run()
    discord(not a.no_discord)
    gecko()
    solana_rpc()
    jup()
    databento()
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} OK" + (" - fix the FAIL lines above" if fails else " - all good"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
