import datetime as dt
import json

import pandas as pd
import pytest

from proplab.core.propsim import PropRules, run_eval, simulate
from proplab.core.research_log import ResearchLog

RULES = PropRules(account_size=50_000, profit_target=3_000, max_drawdown=2_000,
                  drawdown_type="eod_trailing", drawdown_lock_at=50_000, daily_loss_limit=1_000,
                  min_trading_days=2, max_days=30)


def days(*trades_per_day):
    return [list(d) for d in trades_per_day]


def test_passes_after_target_and_min_days():
    dt_ = days([(2000, -100, 2000)], [(1500, -100, 1500)])
    assert run_eval(dt_, 0, RULES, 1) == ("pass", 2)


def test_one_big_day_needs_second_day():
    dt_ = days([(3500, -100, 3500)], [], [(10, -5, 10)])
    assert run_eval(dt_, 0, RULES, 1) == ("pass", 3)


def test_open_loss_breaching_floor_fails_even_if_trade_recovers():
    # Daily loss limit is off here, so the drawdown floor is what gets hit.
    rules = PropRules(**{**RULES.__dict__, "daily_loss_limit": None})
    dt_ = days([(500, -2100, 600)])
    assert run_eval(dt_, 0, rules, 1)[0] == "fail"


def test_daily_loss_limit_stops_the_day_not_the_account():
    dt_ = days([(-1500, -1500, 0), (900, -10, 900)], [(3000, -10, 3000)], [(1100, -10, 1100)])
    # day 1 capped at -1000 and the 2nd trade is skipped; then +3000 (=2000 profit), then +1100 -> pass
    assert run_eval(dt_, 0, RULES, 1) == ("pass", 3)


def test_size_scales_risk():
    dt_ = days([(-700, -700, 0)], [(-700, -700, 0)], [(-700, -700, 0)])
    assert run_eval(dt_, 0, RULES, 1)[0] == "fail"   # -2100 total
    assert run_eval(dt_[:2], 0, RULES, 1)[0] == "timeout"
    # 3 micros: -2100 open loss in one trade. The daily loss limit flattens you at -1000,
    # so the account survives the day...
    assert run_eval(dt_[:1], 0, RULES, 3)[0] == "timeout"
    # ...but with no daily loss limit, the same trade blows through the 2000 drawdown.
    no_dll = PropRules(**{**RULES.__dict__, "daily_loss_limit": None})
    assert run_eval(dt_[:1], 0, no_dll, 3)[0] == "fail"
    assert run_eval(dt_[:1], 0, no_dll, 2)[0] == "timeout"


def test_simulate_counts_every_start_day():
    sd = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    t = pd.DataFrame({"entry_time": pd.to_datetime([f"{d} 10:00" for d in sd]).tz_localize("America/New_York"),
                      "pnl": 400.0, "mae_usd": -50.0, "mfe_usd": 400.0})
    r = simulate(t, sd, RULES, 1)
    assert r["attempts"] == 10 and r["pass_%"] == 100.0


def test_research_log_detects_edits(tmp_path):
    p = tmp_path / "log.jsonl"
    log = ResearchLog(str(p))
    idea = log.register("orb", "orb", {"rr": 2}, "breakouts continue", "synthetic")
    log.result(idea, {"net": 1}, "fail")
    log.register("orb", "orb", {"rr": 3}, "bigger target", "synthetic")
    assert log.verify()
    assert log.stats("orb")["distinct_in_family"] == 2
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1]); rec["verdict"] = "pass"; lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError, match="edited"):
        log.verify()
