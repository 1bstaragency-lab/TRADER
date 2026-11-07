"""Run a futures strategy through the whole funnel.

  python run_futures_lab.py --data data/nq_1m.parquet --strategy orb \
      --params '{"rr": 2.0}' --hypothesis "early breakouts keep going"

The most recent --holdout-months of data are LOCKED: the strategy never
sees them until you pass --unlock-holdout (do that once, at the very end).
Every unlock is written to the research log.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import yaml

from proplab import lab
from proplab.core.engine import NY, check_no_lookahead
from proplab.core.instruments import load_instrument
from proplab.core.propsim import PropRules, compare_sizes
from proplab.core.research_log import ResearchLog
from proplab.core.stats import by_year
from proplab.data import load_bars
from proplab.strategies import get


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--strategy", default="orb")
    ap.add_argument("--params", default="{}", help="JSON of parameters to override")
    ap.add_argument("--hypothesis", default="", help="why should this work, and who loses money to it?")
    ap.add_argument("--instrument", default="MNQ")
    ap.add_argument("--instruments-file", default="config/instruments.yaml")
    ap.add_argument("--prop", default="config/prop_firms/example_50k.yaml")
    ap.add_argument("--is-end", default="2020-12-31", help="last day of the build (in-sample) period")
    ap.add_argument("--holdout-months", type=int, default=12)
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--flips", type=int, default=500)
    ap.add_argument("--sizes", default="1,2,3,5")
    ap.add_argument("--min-beat", type=float, default=95.0)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--prop-random-sets", type=int, default=20)
    ap.add_argument("--prop-start-step", type=int, default=1, help="start an eval every N days (1 = every day)")
    ap.add_argument("--log", default="research_log/log.jsonl")
    a = ap.parse_args(argv)

    strat = get(a.strategy)
    params = {**strat.DEFAULT_PARAMS, **json.loads(a.params)}
    with open(a.instruments_file) as f:
        inst = load_instrument(yaml.safe_load(f)[a.instrument])
    rules = PropRules.from_yaml(a.prop)

    log = ResearchLog(a.log)
    log.verify()
    df = load_bars(a.data)
    idea = log.register(a.strategy, a.strategy, params, a.hypothesis, f"{a.data} {df.index.min()}..{df.index.max()}")
    print(f"Registered test #{idea['seq']} (variation {idea['variation']}) in the research log.\n")

    print("Checking the strategy cannot see the future...")
    check_no_lookahead(strat.generate_signals, df.iloc[: min(len(df), 300_000)], params)
    print("  ok\n")

    holdout_start = df.index.max().normalize() - pd.DateOffset(months=a.holdout_months)
    is_end = pd.Timestamp(a.is_end, tz=NY) + pd.Timedelta(days=1)
    unlocked = df[df.index < holdout_start]
    chunks = {"IN-SAMPLE": unlocked[unlocked.index < is_end], "OUT-OF-SAMPLE": unlocked[unlocked.index >= is_end]}
    if a.unlock_holdout:
        prior = log.stats(a.strategy)["holdout_unlocks_in_family"]
        log.holdout_unlock(idea)
        print(f"!! HOLDOUT UNLOCKED. Previous unlocks for '{a.strategy}': {prior}. "
              "If you tweak and re-run after this, the holdout is no longer unseen.\n")
        chunks["HOLDOUT"] = df[df.index >= holdout_start]
    for k, v in chunks.items():
        if len(v) == 0:
            sys.exit(f"{k} period has no data - check --is-end / --holdout-months against your data range.")

    periods = {}
    for k, chunk in chunks.items():
        print(f"Backtesting {k} + {a.flips} coin flips...")
        periods[k] = lab.run_period([chunk], strat, params, inst, a.flips, "time_of_day",
                                    keep_sets=a.prop_random_sets if k != "HOLDOUT" else 0)
    table = lab.side_by_side(periods)
    print("\n" + table.to_string() + "\n")

    trades = pd.concat([periods[k]["trades"] for k in ("IN-SAMPLE", "OUT-OF-SAMPLE")], ignore_index=True)
    rand = [pd.concat([periods["IN-SAMPLE"]["coinflip"]["random_sets"][i],
                       periods["OUT-OF-SAMPLE"]["coinflip"]["random_sets"][i]], ignore_index=True)
            for i in range(a.prop_random_sets)]
    session_days = sorted(set(unlocked.index.tz_convert(NY).date))
    sizes = [int(s) for s in a.sizes.split(",")]
    print(f"Simulating '{rules.name}' evaluations from every start day at sizes {sizes}...")
    prop = compare_sizes(trades, rand, session_days, rules, sizes, a.prop_start_step)
    print("\n" + prop.to_string() + "\n")

    checks = lab.checklist(periods, a.min_beat, a.min_trades)
    good_sizes = prop[prop["edge_vs_coinflip_pts"] >= 10] if len(prop) else prop
    checks.append(("At some size, passes evals >= 10 pts more often than coin flips", len(good_sizes) > 0))
    verdict = "CANDIDATE - next step: paper trade it" if all(ok for _, ok in checks) else "DOES NOT TRADE"
    best = None
    if len(good_sizes):
        best = int(good_sizes["cost_per_pass_$"].astype(float).idxmin())

    lines = [("PASS " if ok else "FAIL ") + name for name, ok in checks]
    note = lab.multiple_testing_note(log.stats(a.strategy), a.strategy)
    print("\n".join(lines))
    print(f"\nVERDICT: {verdict}" + (f"  (cheapest size that beats coin flips: {best} micros)" if best else ""))
    print("\n" + note)

    log.result(idea, {**lab.metrics_for_log(periods), "prop": prop.reset_index().to_dict("records")}, verdict)
    path = lab.write_report(f"{a.strategy}_{idea['variation']}", [
        ("Test", lab.block(json.dumps({"strategy": a.strategy, "params": params, "hypothesis": a.hypothesis,
                                       "instrument": inst.__dict__, "data": a.data, "log_seq": idea["seq"]},
                                      indent=2, default=str))),
        ("Built-on years vs unseen years", lab.block(table)),
        ("By year (built-on + unseen)", lab.block(by_year(trades))),
        (f"Prop firm simulation: {rules.name}", lab.block(prop)),
        ("Checklist", lab.block("\n".join(lines))),
        ("Verdict", f"**{verdict}**" + (f" - cheapest size beating coin flips: {best} micros" if best else "")),
        ("Multiple testing", note),
    ])
    print(f"\nReport: {path}")
    return verdict


if __name__ == "__main__":
    main()
