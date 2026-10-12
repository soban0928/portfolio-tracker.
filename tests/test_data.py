import pandas as pd
import pytest

from simulator.config import SAMPLE_DATA_DIR
from simulator.data import PriceStore, currency_for, load_price_csv, to_timestamp
from simulator.errors import NoPriceDataError, UnknownTickerError


def test_currency_inference():
    assert currency_for("AAPL") == "USD"
    assert currency_for("vod.l") == "GBP"


def test_get_bar_exact_date(store):
    bar = store.get_bar("abc", "2024-01-03")
    assert (bar.open, bar.high, bar.low, bar.close) == (102.0, 110.0, 101.0, 108.0)


def test_get_bar_missing_date_raises(store):
    with pytest.raises(NoPriceDataError):
        store.get_bar("ABC", "2024-01-06")  # Saturday
    assert not store.has_bar("ABC", "2024-01-06")
    assert store.has_bar("ABC", "2024-01-02")


def test_unknown_ticker(store):
    with pytest.raises(UnknownTickerError):
        store.get_bar("NOPE", "2024-01-02")


def test_close_on_or_before_uses_last_available(store):
    assert store.close_on_or_before("ABC", "2024-01-07") == 92.0
    with pytest.raises(NoPriceDataError):
        store.close_on_or_before("ABC", "2023-12-31")


def test_fx_conversion(store):
    # $125 at 1.25 USD/GBP = £100
    assert store.to_base(125.0, "USD", "2024-01-02") == pytest.approx(100.0)
    assert store.from_base(100.0, "USD", "2024-01-03") == pytest.approx(120.0)
    assert store.to_base(50.0, "GBP", "2024-01-02") == 50.0
    # weekend uses Friday's... here last available (Jan 4) rate
    assert store.fx_rate("2024-01-06") == 1.25
    with pytest.raises(NoPriceDataError):
        store.fx_rate("2023-01-01")
    with pytest.raises(ValueError):
        store.to_base(1.0, "EUR", "2024-01-02")
    with pytest.raises(ValueError):
        store.from_base(1.0, "EUR", "2024-01-02")


def test_closes_in_base(store):
    s = store.closes_in_base("ABC")
    assert list(s.round(4)) == [round(102 / 1.25, 4), 90.0, round(92 / 1.25, 4)]
    assert list(store.closes_in_base("XYZ.L")) == [10.0, 11.0, 12.0]


def test_closes_slicing(store):
    s = store.closes("ABC", "2024-01-03", "2024-01-03")
    assert list(s) == [108.0]


def test_trading_days_and_ranges(store):
    days = store.trading_days("2024-01-01", "2024-01-03")
    assert list(days) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert store.date_range("ABC") == (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"))
    assert store.date_range()[1] == pd.Timestamp("2024-01-04")
    assert len(store.trading_days("2024-01-01", "2024-12-31", ticker="XYZ.L")) == 3


def test_trading_days_without_benchmark():
    from tests.conftest import make_frame

    a = make_frame([("2024-01-02", 1, 1, 1, 1)])
    b = make_frame([("2024-01-03", 1, 1, 1, 1)])
    s = PriceStore({"A.L": a, "B.L": b})
    assert len(s.trading_days("2024-01-01", "2024-01-31")) == 2


def test_store_requires_fx_for_usd_tickers():
    from tests.conftest import make_frame

    with pytest.raises(ValueError):
        PriceStore({"ABC": make_frame([("2024-01-02", 1, 1, 1, 1)])})
    s = PriceStore({"ABC.L": make_frame([("2024-01-02", 1, 1, 1, 1)])})
    with pytest.raises(NoPriceDataError):
        s.fx_series()
    assert s.currency("ABC.L") == "GBP"


def test_to_timestamp_strips_timezone():
    ts = to_timestamp(pd.Timestamp("2024-01-02 15:30", tz="UTC"))
    assert ts == pd.Timestamp("2024-01-02") and ts.tzinfo is None


def test_load_price_csv(tmp_path):
    p = tmp_path / "T.csv"
    p.write_text("date,open,high,low,close\n2024-01-03,2,3,1,2\n2024-01-02,1,2,1,1.5\n")
    df = load_price_csv(p)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert (df["Volume"] == 0).all()

    bad = tmp_path / "B.csv"
    bad.write_text("Date,Open\n2024-01-02,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_price_csv(bad)


def test_from_directories_override_and_errors(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "T.L.csv").write_text("Date,Open,High,Low,Close\n2024-01-02,1,1,1,1\n")
    (b / "T.L.csv").write_text("Date,Open,High,Low,Close\n2024-01-02,2,2,2,2\n")
    s = PriceStore.from_directories(a, b, tmp_path / "missing")
    assert s.get_bar("T.L", "2024-01-02").close == 2.0
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        PriceStore.from_directories(empty)


def test_bundled_sample_data_loads():
    s = PriceStore.from_directories(SAMPLE_DATA_DIR)
    assert {"AAPL", "MSFT", "SPY", "VOD.L"} <= set(s.tickers)
    assert "GBPUSD" not in s.tickers
    assert s.benchmark == "SPY"
    first, last = s.date_range("SPY")
    assert first.year == 2019 and last.year == 2024
    df = s.frame("AAPL")
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert s.closes_in_base("AAPL").notna().all()


def test_default_store_loads():
    assert "SPY" in PriceStore.default().tickers
