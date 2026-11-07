"""Run a Solana memecoin strategy through the funnel (no prop sim - prop firms don't do memecoins).

  python scripts/download_solana_history.py --pools 40        # build a dataset first
  python run_solana_lab.py --data data/solana --strategy sol_momentum

Data is split by TIME across all pools: the first --is-frac is the build period,
the next part is unseen, and the last --holdout-frac is locked.

SURVIVORSHIP WARNING: free APIs mostly return pools that still exist and still
trade. The thousands of coins that rugged to zero are under-represented, so a
memecoin backtest looks BETTER than reality. Paper trading is the real test.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from proplab import lab
from proplab.core.engine import check_no_lookahead, to_ns
from proplab.core.instruments import load_instrument
from proplab.core.research_log import ResearchLog
from proplab.data import load_bars
from proplab.strategies import get


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/solana")
    ap.add_argument("--strategy", default="sol_momentum")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--instrument", default="SOL_MEME")
    ap.add_argument("--instruments-file", default="config/instruments.yaml")
    ap.add_argument("--is-frac", type=float, default=0.6)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--flips", type=int, default=500)
    ap.add_argument("--min-beat", type=float, default=95.0)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--log", default="research_log/log.jsonl")
    a = ap.parse_args(argv)

    strat = get(a.strategy)
    params = {**strat.DEFAULT_PARAMS, **json.loads(a.params)}
    with open(a.instruments_file) as f:
        inst = load_instrument(yaml.safe_load(f)[a.instrument])
    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    if not files:
        raise SystemExit(f"no .parquet files in {a.data} - run scripts/download_solana_history.py first")
    pools = [load_bars(p, tz="UTC") for p in files]
    synthetic = any(json.load(open(p[:-8] + ".json")).get("synthetic") for p in files if os.path.exists(p[:-8] + ".json"))

    log = ResearchLog(a.log)
    log.verify()
    idea = log.register(a.strategy, a.strategy, params, a.hypothesis, f"{a.data} ({len(pools)} pools)")
    print(f"Registered test #{idea['seq']} (variation {idea['variation']}). {len(pools)} pools loaded.")
    print("\nSURVIVORSHIP WARNING: dead/rugged coins are under-represented in this data -> results are optimistic.\n")

    print("Checking the strategy cannot see the future...")
    biggest = max(pools, key=len)
    check_no_lookahead(strat.generate_signals, biggest, params)
    print("  ok\n")

    all_t = np.sort(np.concatenate([to_ns(p.index) for p in pools]))
    t_is = pd.Timestamp(all_t[int(len(all_t) * a.is_frac)], tz="UTC")
    t_ho = pd.Timestamp(all_t[int(len(all_t) * (1 - a.holdout_frac))], tz="UTC")
    chunks = {"IN-SAMPLE": [p[p.index < t_is] for p in pools],
              "OUT-OF-SAMPLE": [p[(p.index >= t_is) & (p.index < t_ho)] for p in pools]}
    if a.unlock_holdout:
        log.holdout_unlock(idea)
        chunks["HOLDOUT"] = [p[p.index >= t_ho] for p in pools]
        print("!! HOLDOUT UNLOCKED - logged.\n")

    periods = {}
    for k, ch in chunks.items():
        ch = [c for c in ch if len(c) > 1]
        print(f"Backtesting {k} on {len(ch)} pools + {a.flips} coin flips...")
        periods[k] = lab.run_period(ch, strat, params, inst, a.flips, "anywhere")
    table = lab.side_by_side(periods)
    print("\n" + table.to_string() + "\n")

    checks = lab.checklist(periods, a.min_beat, a.min_trades)
    verdict = "CANDIDATE - next step: paper trade it" if all(ok for _, ok in checks) else "DOES NOT TRADE"
    lines = [("PASS " if ok else "FAIL ") + n for n, ok in checks]
    note = lab.multiple_testing_note(log.stats(a.strategy), a.strategy)
    print("\n".join(lines) + f"\n\nVERDICT: {verdict}\n\n{note}")
    if synthetic:
        print("\n(note: this dataset is SYNTHETIC - the result says nothing about real coins)")
    log.result(idea, lab.metrics_for_log(periods), verdict)
    path = lab.write_report(f"{a.strategy}_{idea['variation']}", [
        ("Test", lab.block(json.dumps({"strategy": a.strategy, "params": params, "hypothesis": a.hypothesis,
                                       "pools": len(pools), "synthetic": synthetic}, indent=2))),
        ("Built-on period vs unseen period", lab.block(table)),
        ("Checklist", lab.block("\n".join(lines))),
        ("Verdict", f"**{verdict}**"),
        ("Warnings", "Survivorship bias: rugged coins are under-represented. Multiple testing: " + note),
    ])
    print(f"\nReport: {path}")
    return verdict


if __name__ == "__main__":
    main()
