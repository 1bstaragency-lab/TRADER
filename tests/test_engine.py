import numpy as np
import pandas as pd
import pytest

from proplab.core.engine import Bars, PaperPosition, backtest, check_no_lookahead, simulate_signals
from proplab.core.instruments import Futures, Spot
from proplab.strategies import futures_orb, sol_momentum
from scripts.make_synthetic_data import futures as make_futures

MNQ = Futures()


@pytest.fixture(scope="module")
def df():
    return make_futures(years=1, plant_edge=False, seed=3)


def test_vectorized_matches_bar_by_bar(df):
    """The backtest and the live paper tracker must agree on every single trade."""
    bars = Bars(df)
    rng = np.random.default_rng(0)
    k = 400
    idx = rng.integers(0, bars.n - 2, size=k)
    side = rng.choice([-1, 1], size=k)
    sd = rng.uniform(2, 40, size=k)
    td = rng.uniform(2, 80, size=k)
    exit_ns = bars.ns[idx] + rng.integers(1, 300, size=k) * 60_000_000_000
    for inst in (MNQ, Spot(can_short=True)):
        t = simulate_signals(bars, idx, side, sd, td, exit_ns, inst)
        for row in t.itertuples():
            pos = PaperPosition(row.side, bars.o[row.entry_idx], row.stop_dist, row.target_dist,
                                row.planned_exit_ns, inst)
            out, j = None, row.entry_idx
            while out is None:
                out = pos.on_bar(bars.ns[j], bars.o[j], bars.h[j], bars.l[j], bars.c[j])
                j += 1
            assert out["kind"] == row.kind
            assert out["exit"] == pytest.approx(row.exit)
            assert out["pnl"] == pytest.approx(row.pnl)


def test_flat_trade_loses_exactly_the_costs():
    idx = pd.date_range("2024-01-02 09:30", periods=5, freq="min", tz="America/New_York")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    sig = pd.DataFrame({"idx": [0], "side": [1], "stop_dist": [10.0], "target_dist": [10.0],
                        "exit_time": [idx[3]]})
    t = backtest(Bars(df), sig, MNQ)
    # entry slip 1 tick + time-exit slip 1 tick = 0.5 pts * $2 = $1, plus 2 x $0.62 commission
    assert t["pnl"].iloc[0] == pytest.approx(-(0.5 * 2 + 2 * 0.62))
    assert t["kind"].iloc[0] == "time"


def test_stop_wins_when_both_hit_in_same_bar():
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="min", tz="America/New_York")
    df = pd.DataFrame({"open": [100, 100, 100.0], "high": [100, 200, 100.0], "low": [100, 0, 100.0],
                       "close": [100, 100, 100.0], "volume": 1.0}, index=idx)
    sig = pd.DataFrame({"idx": [0], "side": [1], "stop_dist": [5.0], "target_dist": [5.0], "exit_time": [idx[2]]})
    assert backtest(Bars(df), sig, MNQ)["kind"].iloc[0] == "stop"


def test_strategies_do_not_look_ahead(df):
    check_no_lookahead(futures_orb.generate_signals, df.iloc[: 390 * 60])
    sol = df.iloc[:5000].copy()
    check_no_lookahead(sol_momentum.generate_signals, sol)


def test_cheating_strategy_is_caught(df):
    def cheat(d, params=None):
        nxt = d["close"].shift(-1)            # peeks at the NEXT bar
        idx = np.flatnonzero((nxt > d["close"]).to_numpy())
        return pd.DataFrame({"idx": idx, "side": 1, "stop_dist": 5.0, "target_dist": 5.0,
                             "exit_time": d.index[idx] + pd.Timedelta(minutes=5)})
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        check_no_lookahead(cheat, df.iloc[:20000])
