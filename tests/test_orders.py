import math

import pandas as pd
import pytest

from simulator.config import FeeConfig
from simulator.data import Bar
from simulator.errors import InvalidOrderError, InvalidQuantityError
from simulator.orders import (
    Order,
    OrderType,
    Side,
    apply_slippage,
    execution_price,
    make_order,
    max_affordable_quantity,
    reference_fill_price,
    validate_quantity,
)

BAR = Bar(pd.Timestamp("2024-01-02"), open=100.0, high=105.0, low=95.0, close=102.0)
NO_FEES = FeeConfig(commission=0, slippage_bps=0)


def order(side, order_type=OrderType.MARKET, limit=None, stop=None):
    return Order("ABC", side, 10, pd.Timestamp("2024-01-02"), order_type, limit, stop)


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True, None, math.nan, math.inf])
def test_invalid_quantities(bad):
    with pytest.raises(InvalidQuantityError):
        validate_quantity(bad)


def test_valid_quantities():
    assert validate_quantity(10) == 10
    assert validate_quantity(10.0) == 10


def test_make_order_normalises_and_validates():
    o = make_order("abc ", "buy", 5, "2024-01-02", "limit", limit_price=99)
    assert (o.ticker, o.side, o.order_type, o.quantity) == ("ABC", Side.BUY, OrderType.LIMIT, 5)
    assert o.trigger_price == 99
    assert o.describe() == "BUY 5 ABC LIMIT @ 99"
    with pytest.raises(InvalidOrderError):
        make_order("ABC", "hold", 5, "2024-01-02")
    with pytest.raises(InvalidOrderError):
        make_order("ABC", "buy", 5, "2024-01-02", "limit")  # no limit price
    with pytest.raises(InvalidOrderError):
        make_order("ABC", "sell", 5, "2024-01-02", "stop", stop_price=-1)
    with pytest.raises(InvalidOrderError):
        make_order("  ", "buy", 5, "2024-01-02")
    with pytest.raises(InvalidQuantityError):
        make_order("ABC", "buy", 0, "2024-01-02")


def test_market_order_has_no_trigger():
    o = make_order("ABC", "sell", 1, "2024-01-02")
    assert o.trigger_price is None
    assert o.describe() == "SELL 1 ABC MARKET"
    assert make_order("ABC", "sell", 1, "2024-01-02", "stop", stop_price=9).trigger_price == 9


# ------------------------------------------------------------ fill rules
@pytest.mark.parametrize(
    ("side", "otype", "limit", "stop", "expected"),
    [
        (Side.BUY, OrderType.MARKET, None, None, 100.0),
        (Side.SELL, OrderType.MARKET, None, None, 100.0),
        # limit buy: fills if low <= limit, at min(open, limit)
        (Side.BUY, OrderType.LIMIT, 97.0, None, 97.0),
        (Side.BUY, OrderType.LIMIT, 101.0, None, 100.0),
        (Side.BUY, OrderType.LIMIT, 94.0, None, None),
        # limit sell: fills if high >= limit, at max(open, limit)
        (Side.SELL, OrderType.LIMIT, 104.0, None, 104.0),
        (Side.SELL, OrderType.LIMIT, 99.0, None, 100.0),
        (Side.SELL, OrderType.LIMIT, 106.0, None, None),
        # stop-loss sell: triggers if low <= stop, at min(open, stop) (gap-aware)
        (Side.SELL, OrderType.STOP, None, 96.0, 96.0),
        (Side.SELL, OrderType.STOP, None, 101.0, 100.0),
        (Side.SELL, OrderType.STOP, None, 94.0, None),
        # stop buy: triggers if high >= stop, at max(open, stop)
        (Side.BUY, OrderType.STOP, None, 103.0, 103.0),
        (Side.BUY, OrderType.STOP, None, 99.0, 100.0),
        (Side.BUY, OrderType.STOP, None, 106.0, None),
    ],
)
def test_reference_fill_price(side, otype, limit, stop, expected):
    assert reference_fill_price(order(side, otype, limit, stop), BAR) == expected


def test_slippage_moves_price_against_trader():
    fees = FeeConfig(commission=0, slippage_bps=10)
    assert apply_slippage(100.0, Side.BUY, fees) == pytest.approx(100.1)
    assert apply_slippage(100.0, Side.SELL, fees) == pytest.approx(99.9)


def test_limit_orders_never_fill_beyond_limit_after_slippage():
    fees = FeeConfig(commission=0, slippage_bps=10)
    price, ref = execution_price(order(Side.BUY, OrderType.LIMIT, 97.0), BAR, fees)
    assert (price, ref) == (97.0, 97.0)  # 97.097 capped at the limit
    price, _ = execution_price(order(Side.BUY, OrderType.LIMIT, 101.0), BAR, fees)
    assert price == pytest.approx(100.1)  # price improvement absorbs slippage
    price, _ = execution_price(order(Side.SELL, OrderType.LIMIT, 104.0), BAR, fees)
    assert price == 104.0
    assert execution_price(order(Side.BUY, OrderType.LIMIT, 90.0), BAR, fees) is None


def test_market_and_stop_orders_take_full_slippage():
    fees = FeeConfig(commission=0, slippage_bps=50)
    price, ref = execution_price(order(Side.SELL, OrderType.STOP, stop=96.0), BAR, fees)
    assert ref == 96.0 and price == pytest.approx(95.52)


def test_max_affordable_quantity():
    fees = FeeConfig(commission=5, slippage_bps=0)
    assert max_affordable_quantity(1000, 10.0, 1.0, fees) == 99  # (1000-5)/10
    assert max_affordable_quantity(1000, 100.0, 1.25, fees) == 12  # 995 / £80
    assert max_affordable_quantity(4, 10.0, 1.0, fees) == 0
    assert max_affordable_quantity(1000, 0.0, 1.0, fees) == 0
    slip = FeeConfig(commission=0, slippage_bps=100)
    assert max_affordable_quantity(1010, 10.0, 1.0, slip) == 100  # 10.10 per share
