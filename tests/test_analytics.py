import math

import pandas as pd
import pytest

from simulator import analytics as an
from simulator.config import SAMPLE_DATA_DIR
from simulator.data import PriceStore

IDX = pd.bdate_range("2024-01-01", periods=4)
# returns: +10%, -10%, +10%
VALUES = pd.Series([100.0, 110.0, 99.0, 108.9], index=IDX)


def test_daily_returns():
    assert list(an.daily_returns(VALUES).round(10)) == [0.1, -0.1, 0.1]


def test_total_and_annualised_return():
    assert an.total_return(VALUES) == pytest.approx(0.089)
    # 3 periods: (1.089)^(252/3) - 1
    assert an.annualised_return(VALUES) == pytest.approx(1.089**84 - 1)
    assert math.isnan(an.total_return(VALUES.iloc[:1]))
    assert math.isnan(an.annualised_return(pd.Series([100.0, 0.0])))


def test_annualised_volatility_hand_calc():
    # mean 1/30; deviations 1/15, -2/15, 1/15; var = (6/225)/2 = 1/75
    expected = math.sqrt(1 / 75) * math.sqrt(252)
    assert an.annualised_volatility(an.daily_returns(VALUES)) == pytest.approx(expected)
    assert expected == pytest.approx(1.8330, abs=1e-4)
    assert math.isnan(an.annualised_volatility(pd.Series([0.1])))


def test_sharpe_hand_calc():
    r = an.daily_returns(VALUES)
    expected = (1 / 30) / math.sqrt(1 / 75) * math.sqrt(252)  # ~4.5826
    assert an.sharpe_ratio(r) == pytest.approx(expected)
    assert expected == pytest.approx(4.5826, abs=1e-4)
    # risk-free 2.52% p.a. -> 0.0001 per day subtracted from the mean
    expected_rf = (1 / 30 - 0.0001) / math.sqrt(1 / 75) * math.sqrt(252)
    assert an.sharpe_ratio(r, risk_free=0.0252) == pytest.approx(expected_rf)
    assert math.isnan(an.sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))  # zero vol
    assert math.isnan(an.sharpe_ratio(pd.Series([0.01])))


def test_max_drawdown_hand_calc():
    assert an.max_drawdown(VALUES) == pytest.approx(-0.1)  # 110 -> 99
    v = pd.Series([100.0, 120.0, 90.0, 130.0, 65.0])
    assert an.max_drawdown(v) == pytest.approx(-0.5)  # 130 -> 65
    assert an.max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0
    assert math.isnan(an.max_drawdown(pd.Series([1.0])))
    assert list(an.drawdown_series(v).round(4)) == [0, 0, -0.25, 0, -0.5]


def test_beta_and_alpha_hand_calc():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.Series([0.02, -0.01, 0.03], index=idx)
    b = pd.Series([0.01, -0.01, 0.02], index=idx)
    # deviations (x300): r -> 2,-7,5 ; b -> 1,-5,4 ; beta = 57/42 = 19/14
    assert an.beta(r, b) == pytest.approx(19 / 14)
    # daily alpha = 0.04/3 - 19/14 * 0.02/3 = 3/700 ; annualised x252 = 1.08
    assert an.alpha(r, b) == pytest.approx(1.08)


def test_beta_alpha_of_levered_benchmark():
    idx = pd.bdate_range("2024-01-01", periods=5)
    b = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005], index=idx)
    assert an.beta(2 * b, b) == pytest.approx(2.0)
    assert an.alpha(2 * b, b) == pytest.approx(0.0, abs=1e-12)
    # With rf > 0: daily alpha = mean(2b - rf) - 2*mean(b - rf) = rf_daily,
    # so the annualised alpha equals the annual risk-free rate.
    rf = 0.0252
    assert an.alpha(2 * b, b, risk_free=rf) == pytest.approx(rf)


def test_beta_alignment_and_degenerate_inputs():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.Series([0.01, 0.02, 0.03], index=idx)
    flat = pd.Series([0.01, 0.01, 0.01], index=idx)
    assert math.isnan(an.beta(r, flat))
    assert math.isnan(an.alpha(r, flat))
    other = pd.Series([0.01], index=[pd.Timestamp("2030-01-01")])
    assert math.isnan(an.beta(r, other))


def test_normalise():
    assert list(an.normalise(VALUES).round(2)) == [100.0, 110.0, 99.0, 108.9]
    assert list(an.normalise(pd.Series([2.0, 3.0]), base=1)) == [1.0, 1.5]
    assert an.normalise(pd.Series([], dtype=float)).empty


def test_benchmark_values_in_gbp(store):
    bench = an.benchmark_values(store, "2024-01-02", "2024-01-04", initial=1000)
    # SPY GBP prices: 400/1.25=320, 404/1.20=336.67, 398/1.25=318.4
    expected = [1000.0, (404 / 1.20) / 320 * 1000, (398 / 1.25) / 320 * 1000]
    assert list(bench) == pytest.approx(expected)
    assert expected[1] == pytest.approx(1052.0833, abs=1e-4)
    assert an.benchmark_values(store, "2030-01-01", "2030-02-01", 1000).empty


def test_performance_summary(store):
    pf_values = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-02", periods=3))
    bench = an.benchmark_values(store, "2024-01-02", "2024-01-04", initial=100)
    s = an.performance_summary(pf_values, bench)
    assert set(s) >= {"total_return", "sharpe_ratio", "max_drawdown", "beta", "alpha"}
    assert s["total_return"] == pytest.approx(-0.01)
    assert "beta" not in an.performance_summary(pf_values)


def test_benchmark_beta_vs_itself_on_sample_data():
    store = PriceStore.from_directories(SAMPLE_DATA_DIR)
    spy = an.daily_returns(store.closes("SPY"))
    assert an.beta(spy, spy) == pytest.approx(1.0)
    assert an.alpha(spy, spy) == pytest.approx(0.0, abs=1e-12)
    nvda = an.daily_returns(store.closes("NVDA"))
    assert an.beta(nvda, spy) > 1.3  # generated with beta 1.7
