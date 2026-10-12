import pandas as pd
import pytest

from simulator.backtest import (
    BuyAndHold,
    Momentum,
    MovingAverageCrossover,
    benchmark_run,
    compare_strategies,
    run_backtest,
)
from simulator.config import FeeConfig
from simulator.data import PriceStore

NO_FEES = FeeConfig(commission=0, slippage_bps=0)
D1, D3 = "2024-01-02", "2024-01-04"


def panel(**cols):
    idx = pd.bdate_range("2024-01-01", periods=len(next(iter(cols.values()))))
    return pd.DataFrame(cols, index=idx, dtype=float)


# ---------------------------------------------------------------- signals
def test_buy_and_hold_weights():
    w = BuyAndHold().target_weights(panel(A=[1, 2], B=[3, 4]))
    assert (w == 0.5).all().all()


def test_ma_crossover_signal_hand_calc():
    # MA2: -,1.5,2.5,3.5,3.5,2.5,1.5 ; MA3: -,-,2,3,3.33,3,2
    w = MovingAverageCrossover(2, 3).target_weights(panel(A=[1, 2, 3, 4, 3, 2, 1]))
    assert list(w["A"]) == [0, 0, 1, 1, 1, 0, 0]
    two = MovingAverageCrossover(2, 3).target_weights(panel(A=[1, 2, 3], B=[3, 2, 1]))
    assert list(two.iloc[-1]) == [0.5, 0.0]  # 1/N sizing
    assert "2/3" in MovingAverageCrossover(2, 3).name
    with pytest.raises(ValueError):
        MovingAverageCrossover(5, 5)


def test_momentum_picks_top_and_filters_negative():
    closes = panel(A=[10, 11, 12, 13], B=[10, 10.5, 11, 11.5], C=[10, 9, 8, 7])
    w = Momentum(lookback=1, top_n=1, rebalance_every=1).target_weights(closes)
    assert list(w.iloc[0]) == [0, 0, 0]  # no lookback history yet
    assert list(w.iloc[-1]) == [1.0, 0, 0]  # A has the best 1-day return
    w2 = Momentum(lookback=1, top_n=2, rebalance_every=1).target_weights(closes)
    assert list(w2.iloc[-1]) == [0.5, 0.5, 0.0]  # C excluded (negative)
    falling = panel(A=[10, 9, 8], B=[10, 8, 6])
    w3 = Momentum(lookback=1, top_n=1, rebalance_every=1).target_weights(falling)
    assert w3.iloc[-1].sum() == 0  # absolute momentum: all cash
    w4 = Momentum(lookback=1, top_n=1, rebalance_every=1, absolute=False).target_weights(falling)
    assert list(w4.iloc[-1]) == [1.0, 0.0]
    assert "top 1" in Momentum(top_n=1).name
    with pytest.raises(ValueError):
        Momentum(lookback=0)


def test_momentum_holds_weights_between_rebalances():
    closes = panel(A=[10, 11, 12, 10, 9], B=[10, 10, 10, 12, 14])
    w = Momentum(lookback=1, top_n=1, rebalance_every=3).target_weights(closes)
    # rebalance on rows 0 and 3 only
    assert list(w["A"]) == [0, 0, 0, 0, 0]
    assert list(w["B"]) == [0, 0, 0, 1, 1]


def test_signals_have_no_lookahead():
    base = list(range(1, 31))
    changed = base[:20] + [100 - x for x in base[20:]]
    s = MovingAverageCrossover(3, 7)
    a = s.target_weights(panel(A=base))
    b = s.target_weights(panel(A=changed))
    pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])


# ---------------------------------------------------------------- runner
def test_buy_and_hold_backtest_hand_calc(store):
    """Decide at D1 close, execute at D2 open.

    Sizing at D1 close: 10,000 / (102 / 1.25 = £81.60) -> 122 shares desired.
    Affordable at D2 open 102 / 1.20 = £85.00 -> floor(10,000 / 85) = 117.
    D2 value: 117 * 108 / 1.20 + 55 = 10,585 ; D3: 117 * 92 / 1.25 + 55 = 8,666.20
    """
    res = run_backtest(store, BuyAndHold(), ["ABC"], D1, D3, fees=NO_FEES)
    assert list(res.values.round(2)) == [10_000.0, 10_585.0, 8_666.2]
    assert len(res.trades) == 1 and res.trades.iloc[0]["quantity"] == 117
    assert res.metrics["total_return"] == pytest.approx(-0.13338)
    assert "beta" in res.metrics


def test_backtest_applies_same_fees(store):
    fees = FeeConfig(commission=5, slippage_bps=10)
    res = run_backtest(store, BuyAndHold(), ["ABC"], D1, D3, fees=fees)
    trade = res.trades.iloc[0]
    assert trade["commission_gbp"] == 5
    assert trade["price"] == pytest.approx(102 * 1.001)
    # (10,000 - 5) / (102.102 / 1.20 = 85.085) = 117.47 -> 117 shares
    assert trade["quantity"] == 117
    assert res.portfolio.total_commission == 5


def test_benchmark_run_matches_engine(store):
    bench = benchmark_run(store, D1, D3, 10_000, NO_FEES)
    # D1 close 400/1.25=320 -> 31 desired; D2 open 400/1.2=333.33 -> 30 affordable
    # D3: 30 * 398 / 1.25 + (10,000 - 30 * 333.33) = 9,552 + 0 = 9,552
    assert bench.iloc[0] == 10_000
    assert bench.iloc[-1] == pytest.approx(30 * 398 / 1.25 + 10_000 - 30 * 400 / 1.2)
    assert bench.name == "Benchmark (SPY)"


def test_backtest_input_validation(store):
    with pytest.raises(ValueError):
        run_backtest(store, BuyAndHold(), [], D1, D3)
    with pytest.raises(ValueError):
        run_backtest(store, BuyAndHold(), ["ABC"], D1, D1)


def test_compare_strategies_on_sample_data():
    store = PriceStore.default()
    tickers = ["AAPL", "MSFT", "NVDA"]
    values, metrics = compare_strategies(
        store,
        [BuyAndHold(), MovingAverageCrossover(10, 50), Momentum(lookback=60, top_n=1)],
        tickers,
        "2021-01-01",
        "2022-12-31",
    )
    assert list(metrics.index) == [
        "Buy & Hold",
        "MA Crossover (10/50)",
        "Momentum (60d, top 1)",
        "Benchmark (SPY)",
    ]
    assert values.shape[1] == 4
    assert (values.iloc[0] == 10_000).all()
    assert metrics.loc["Benchmark (SPY)", "beta"] == pytest.approx(1.0)
    assert metrics.loc["Buy & Hold", "trades"] == 3
    assert metrics.loc["MA Crossover (10/50)", "trades"] > 3
    # cash never goes negative and no short positions
    res = run_backtest(store, MovingAverageCrossover(10, 50), tickers, "2021-01-01", "2022-12-31")
    assert res.portfolio.cash >= 0
    assert all(p.quantity > 0 for p in res.portfolio.positions.values())


def test_ticker_without_history_is_skipped(store):
    from tests.conftest import make_frame

    late = make_frame([("2024-01-04", 5, 5, 5, 5)])
    frames = {t: store.frame(t) for t in store.tickers}
    s = PriceStore({**frames, "LATE.L": late}, store.fx_series())
    res = run_backtest(s, BuyAndHold(), ["ABC", "LATE.L"], D1, D3, fees=NO_FEES)
    assert set(res.trades["ticker"]) == {"ABC"}
