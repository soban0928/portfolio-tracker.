import pandas as pd
import pytest

from simulator.config import FeeConfig
from simulator.errors import (
    InsufficientCashError,
    InsufficientSharesError,
    InvalidOrderError,
    NoPriceDataError,
    UnknownTickerError,
)
from simulator.orders import OrderStatus, make_order
from simulator.portfolio import Portfolio

D1, D2, D3 = "2024-01-02", "2024-01-03", "2024-01-04"
COMMISSION_ONLY = FeeConfig(commission=5, slippage_bps=0)
NO_FEES = FeeConfig(commission=0, slippage_bps=0)


def buy(pf, store, ticker, qty, when, **kw):
    return pf.place_order(make_order(ticker, "buy", qty, when, **kw), store)


def sell(pf, store, ticker, qty, when, **kw):
    return pf.place_order(make_order(ticker, "sell", qty, when, **kw), store)


@pytest.fixture
def traded(store):
    """GBP stock, £5 commission, no slippage.

    D1 buy 100 @ 10 -> cost 1,005       (avg 10.05)
    D2 buy 100 @ 10 -> cost 1,005       (avg 10.05, 200 sh, cost 2,010)
    D3 sell 50 @ 11 -> proceeds 545     realised = 545 - 50*10.05 = 42.50
    """
    pf = Portfolio(fees=COMMISSION_ONLY)
    buy(pf, store, "XYZ.L", 100, D1)
    buy(pf, store, "XYZ.L", 100, D2)
    sell(pf, store, "XYZ.L", 50, D3)
    return pf


# ------------------------------------------------------------- cost basis
def test_weighted_average_cost_and_realised_pnl(traded, store):
    pos = traded.positions["XYZ.L"]
    assert pos.quantity == 150
    assert pos.avg_cost == pytest.approx(10.05)
    assert pos.cost_basis == pytest.approx(1507.5)
    assert traded.realised_pnl == pytest.approx(42.5)
    assert traded.trades[-1].realised_pnl == pytest.approx(42.5)
    assert traded.cash == pytest.approx(10_000 - 1005 - 1005 + 545)


def test_average_cost_blends_different_prices(store):
    pf = Portfolio(fees=COMMISSION_ONLY)
    buy(pf, store, "XYZ.L", 10, D1)  # 10 @ 10 + 5 = 105
    buy(pf, store, "XYZ.L", 10, D3)  # 10 @ 11 + 5 = 115
    pos = pf.positions["XYZ.L"]
    assert pos.avg_cost == pytest.approx(11.0)  # 220 / 20
    assert pos.avg_price_native == pytest.approx(10.5)


def test_unrealised_and_total_return(traded, store):
    # 150 sh @ close 12 = 1,800 ; cost 1,507.50 -> unrealised 292.50
    assert traded.market_value(store, D3) == pytest.approx(1800)
    assert traded.unrealised_pnl(store, D3) == pytest.approx(292.5)
    assert traded.total_value(store, D3) == pytest.approx(10_335)
    assert traded.total_return(store, D3) == pytest.approx(0.0335)
    s = traded.summary(store)
    assert s["total_pnl"] == pytest.approx(s["realised_pnl"] + s["unrealised_pnl"])
    assert s["fees_paid"] == pytest.approx(15)


def test_selling_entire_position_removes_it(store):
    pf = Portfolio(fees=NO_FEES)
    buy(pf, store, "XYZ.L", 10, D1)
    sell(pf, store, "XYZ.L", 10, D2)
    assert "XYZ.L" not in pf.positions
    assert pf.realised_pnl == pytest.approx(0)  # bought and sold at open 10
    assert pf.holdings(store).empty
    assert pf.market_value(store) == 0.0 and pf.unrealised_pnl(store) == 0.0


# ------------------------------------------------------------ fees
def test_commission_and_slippage_applied_to_every_trade(store):
    pf = Portfolio(fees=FeeConfig(commission=5, slippage_bps=10))
    t1 = buy(pf, store, "XYZ.L", 100, D1)  # 10 * 1.001 = 10.01
    assert t1.price == pytest.approx(10.01)
    assert t1.cash_flow == pytest.approx(-1006.0)
    assert t1.slippage_cost == pytest.approx(1.0)
    t2 = sell(pf, store, "XYZ.L", 100, D2)  # 10 * 0.999 = 9.99
    assert t2.price == pytest.approx(9.99)
    assert t2.cash_flow == pytest.approx(994.0)
    assert t2.realised_pnl == pytest.approx(-12.0)  # all of it is fees
    assert pf.total_commission == 10
    assert pf.total_slippage == pytest.approx(2.0)
    assert pf.cash == pytest.approx(10_000 - 12)


# ------------------------------------------------------------ FX
def test_usd_trades_convert_at_trade_date_fx(store):
    pf = Portfolio(fees=NO_FEES)
    t = buy(pf, store, "ABC", 10, D1)  # $1,000 at 1.25 = £800
    assert t.fx_rate == 1.25 and t.currency == "USD"
    assert pf.cash == pytest.approx(9200)
    assert pf.positions["ABC"].avg_cost == pytest.approx(80)
    # D2: close 108, FX 1.20 -> $1,080 = £900 ; unrealised +£100
    assert pf.unrealised_pnl(store, D2) == pytest.approx(100)
    # D3: close 92, FX 1.25 -> £736 ; unrealised -£64
    assert pf.unrealised_pnl(store, D3) == pytest.approx(-64)
    t2 = sell(pf, store, "ABC", 10, D2)  # $1,020 at 1.20 = £850
    assert t2.realised_pnl == pytest.approx(50)


# ------------------------------------------------------------ validation
def test_insufficient_cash_leaves_state_unchanged(store):
    pf = Portfolio(fees=COMMISSION_ONLY)
    with pytest.raises(InsufficientCashError):
        buy(pf, store, "XYZ.L", 1000, D1)  # 10,000 + 5 > 10,000
    assert pf.cash == 10_000 and not pf.positions and not pf.trades and not pf.orders
    buy(pf, store, "XYZ.L", 999, D1)  # 9,990 + 5 fits


def test_cannot_short_or_oversell(store):
    pf = Portfolio(fees=NO_FEES)
    with pytest.raises(InsufficientSharesError):
        sell(pf, store, "XYZ.L", 1, D1)
    buy(pf, store, "XYZ.L", 5, D1)
    with pytest.raises(InsufficientSharesError):
        sell(pf, store, "XYZ.L", 6, D2)
    with pytest.raises(InsufficientSharesError):
        sell(pf, store, "XYZ.L", 6, D2, order_type="limit", limit_price=50)


def test_no_price_data_for_date(store):
    pf = Portfolio()
    with pytest.raises(NoPriceDataError):
        buy(pf, store, "XYZ.L", 1, "2024-01-06")
    with pytest.raises(UnknownTickerError):
        buy(pf, store, "NOPE", 1, D1)


def test_orders_must_be_chronological(store):
    pf = Portfolio()
    buy(pf, store, "XYZ.L", 1, D2)
    with pytest.raises(InvalidOrderError, match="chronological"):
        buy(pf, store, "XYZ.L", 1, D1)


def test_invalid_starting_cash():
    with pytest.raises(ValueError):
        Portfolio(starting_cash=0)


# ------------------------------------------------------------ pending orders
def test_limit_buy_fills_same_day_when_triggered(store):
    pf = Portfolio(fees=NO_FEES)
    t = buy(pf, store, "XYZ.L", 10, D1, order_type="limit", limit_price=9.6)
    assert t is not None and t.price == 9.6


def test_stop_loss_fills_on_later_day_at_stop(store):
    pf = Portfolio(fees=NO_FEES)
    buy(pf, store, "ABC", 10, D1)  # £800
    assert sell(pf, store, "ABC", 10, D2, order_type="stop", stop_price=95) is None
    assert len(pf.pending_orders) == 1
    trades = pf.process_pending(store, D3)  # D3 low 90 <= 95 -> fill at min(108, 95)
    assert len(trades) == 1 and trades[0].price == 95.0
    assert trades[0].realised_pnl == pytest.approx(950 / 1.25 - 800)  # -40
    assert pf.orders[-1].status is OrderStatus.FILLED
    assert pf.clock == pd.Timestamp(D3)


def test_limit_sell_pending_then_filled_when_new_order_placed(store):
    pf = Portfolio(fees=NO_FEES)
    buy(pf, store, "XYZ.L", 100, D1)
    assert sell(pf, store, "XYZ.L", 50, D2, order_type="limit", limit_price=11.5) is None
    # placing another order on D3 processes pending orders through D3 first
    buy(pf, store, "XYZ.L", 1, D3)
    filled = [t for t in pf.trades if t.order_type == "LIMIT"]
    assert len(filled) == 1 and filled[0].price == 11.5
    assert pf.shares_held("XYZ.L") == 51


def test_pending_order_rejected_if_unaffordable_at_fill(store):
    pf = Portfolio(fees=NO_FEES)
    buy(pf, store, "XYZ.L", 100, D1)
    sell(pf, store, "XYZ.L", 100, D2, order_type="limit", limit_price=11.5)
    sell(pf, store, "XYZ.L", 100, D2)  # sells the shares the limit order needed
    pf.process_pending(store, D3)
    rejected = pf.orders[1]
    assert rejected.status is OrderStatus.REJECTED and "only 0 held" in rejected.reason


def test_pending_buy_precheck_and_untriggered(store):
    pf = Portfolio(fees=NO_FEES)
    with pytest.raises(InsufficientCashError):
        buy(pf, store, "XYZ.L", 2000, D2, order_type="limit", limit_price=9.0)
    assert buy(pf, store, "XYZ.L", 10, D2, order_type="limit", limit_price=9.7) is None
    assert pf.process_pending(store, D3) == []  # lows 10.5 never reach 9.7
    assert pf.pending_orders[0].status is OrderStatus.PENDING


def test_cancel_order(store):
    pf = Portfolio(fees=NO_FEES)
    buy(pf, store, "XYZ.L", 10, D2, order_type="limit", limit_price=9.0)
    oid = pf.pending_orders[0].id
    assert pf.cancel_order(oid).status is OrderStatus.CANCELLED
    assert pf.pending_orders == []
    with pytest.raises(InvalidOrderError):
        pf.cancel_order(oid)


# ------------------------------------------------------------ history & replay
def test_value_history(traded, store):
    hist = traded.value_history(store)
    assert list(hist["total"].round(2)) == [9995.0, 10190.0, 10335.0]
    assert list(hist["cash"].round(2)) == [8995.0, 7990.0, 8535.0]


def test_value_history_empty_and_no_trades(store):
    pf = Portfolio()
    assert pf.value_history(store).empty
    flat = pf.value_history(store, D1, D3)
    assert list(flat["total"]) == [10_000.0] * 3
    assert pf.value_history(store, "2025-01-01", "2025-01-02").empty
    assert pf.trades_frame().empty
    with pytest.raises(ValueError, match="No valuation date"):
        pf.value_history(store, start=D1)  # no end date and no clock
    assert pf.summary(store)["total_value"] == 10_000


def test_replay_reproduces_state(traded, store):
    clone = Portfolio.from_records(traded.trades, traded.orders, fees=COMMISSION_ONLY, name="copy")
    assert clone.cash == pytest.approx(traded.cash)
    assert clone.realised_pnl == pytest.approx(traded.realised_pnl)
    assert clone.positions["XYZ.L"].cost_basis == pytest.approx(1507.5)
    assert clone.clock == pd.Timestamp(D3)
    explicit = Portfolio.from_records([], clock=D2)
    assert explicit.clock == pd.Timestamp(D2)


def test_holdings_table(traded, store):
    h = traded.holdings(store, D3)
    row = h.iloc[0]
    assert row["ticker"] == "XYZ.L" and row["quantity"] == 150
    assert row["unrealised_pct"] == pytest.approx(292.5 / 1507.5)
    assert row["weight"] == pytest.approx(1800 / 10_335)
    df = traded.trades_frame()
    assert list(df["side"]) == ["BUY", "BUY", "SELL"]
    assert df["cash_flow_gbp"].sum() == pytest.approx(-1465)
