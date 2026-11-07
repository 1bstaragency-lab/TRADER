"""Start a PAPER trader. Nothing here can place a real order.

  # futures - replay a file (test the plumbing on old data)
  python run_paper.py futures --replay data/synthetic_nq.parquet --from 2021-06-01

  # futures - live Databento bars (weekdays; schedule it for 9:00 ET)
  python run_paper.py futures --live

  # solana memecoins - live quotes, paper fills
  python run_paper.py solana --config config/solana_paper.yaml
"""
from __future__ import annotations

import argparse
import json

import yaml
from dotenv import load_dotenv

from proplab.core.instruments import load_instrument
from proplab.data import load_bars
from proplab.strategies import get


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("market", choices=["futures", "solana"])
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--instrument", default="MNQ")
    ap.add_argument("--qty", type=int, default=1, help="micros to simulate (use the size the prop sim picked)")
    ap.add_argument("--replay", help="file to replay instead of live data")
    ap.add_argument("--from", dest="start", help="replay start date")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--symbol", default="NQ.v.0")
    ap.add_argument("--config", default="config/solana_paper.yaml")
    ap.add_argument("--risk", default="config/risk.yaml", help="risk gate config ('none' to disable)")
    ap.add_argument("--size-by-risk", action="store_true", help="size each trade from max_loss_per_trade_usd")
    a = ap.parse_args()

    if a.market == "solana":
        from proplab.solana.paper_trader import SolanaPaperTrader
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        SolanaPaperTrader(cfg).run()
        return

    from proplab.core.risk import RiskLimits
    from proplab.futures.forward_test import DatabentoLiveFeed, ForwardTester, ReplayFeed
    risk = None if a.risk == "none" else RiskLimits.from_yaml(a.risk)
    strat = get(a.strategy or "orb")
    params = {**strat.DEFAULT_PARAMS, **json.loads(a.params)}
    with open("config/instruments.yaml") as f:
        inst = load_instrument(yaml.safe_load(f)[a.instrument])
    if a.replay:
        df = load_bars(a.replay)
        out = "forward_test/futures_replay"
        ft = ForwardTester(strat, params, inst, out_dir=out, qty=a.qty, label="dealer-replay",
                           risk=risk, size_by_risk=a.size_by_risk)
        ft.run(ReplayFeed(df, a.start))
        print(f"\n{len(ft.trades)} paper trades, {len(ft.blocked)} blocked by the risk gate -> {out}/trades.csv")
    elif a.live:
        ForwardTester(strat, params, inst, qty=a.qty, risk=risk, size_by_risk=a.size_by_risk).run(DatabentoLiveFeed(a.symbol))
    else:
        ap.error("futures needs --replay FILE or --live")


if __name__ == "__main__":
    main()
