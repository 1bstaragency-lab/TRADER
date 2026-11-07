import pandas as pd

from proplab.core.instruments import Futures
from proplab.core.risk import DayState, RiskLimits, check_order, contracts_for_risk

L = RiskLimits(max_loss_per_trade_usd=200, daily_loss_limit_usd=500, max_trades_per_day=3, max_contracts=5)
T = pd.Timestamp("2026-11-09 10:05", tz="America/New_York")


def test_sizing_from_max_loss():
    mnq = Futures()
    # 20 pt stop + 3 ticks slippage = 20.75 pts * $2 = $41.50 + $1.24 commission = $42.74/contract
    assert contracts_for_risk(200, 20, mnq, cap=10) == 4
    assert contracts_for_risk(200, 200, mnq, cap=10) == 0     # stop too wide -> no trade
    assert contracts_for_risk(10_000, 5, mnq, cap=5) == 5     # capped


def test_pass_and_blocks():
    assert check_order(L, DayState(), T, 2, 150, 300) == (True, [])
    ok, why = check_order(L, DayState(), T, 2, 250, 500)
    assert not ok and "loss at stop" in why[0]
    ok, why = check_order(L, DayState(), pd.Timestamp("2026-11-09 15:45", tz="America/New_York"), 1, 50, 100)
    assert not ok and "entry window" in why[0]
    ok, why = check_order(L, DayState(), T, 1, 100, 50)
    assert not ok and "reward/risk" in why[0]


def test_daily_limits_and_reset():
    st = DayState()
    st.roll(T, L.timezone)
    st.on_exit(-520, L)
    ok, why = check_order(L, st, T, 1, 50, 100)
    assert not ok and "daily loss" in why[0]
    ok, _ = check_order(L, st, T + pd.Timedelta(days=1), 1, 50, 100)   # new day resets
    assert ok
    st.trades = 3
    assert not check_order(L, st, T + pd.Timedelta(days=1), 1, 50, 100)[0]
