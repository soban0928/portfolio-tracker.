"""Order model, parameter validation and fill rules against daily bars.

Fill rules (all prices in the instrument's own currency):

=========  ======  ==========================================  ======================
Type       Side    Triggers when                               Reference fill price
=========  ======  ==========================================  ======================
Market     any     always (bar exists)                         ``open``
Limit      buy     ``low <= limit``                            ``min(open, limit)``
Limit      sell    ``high >= limit``                           ``max(open, limit)``
Stop       sell    ``low <= stop`` (stop-loss)                 ``min(open, stop)``
Stop       buy     ``high >= stop``                            ``max(open, stop)``
=========  ======  ==========================================  ======================

Using ``min/max(open, level)`` models gaps: if a stock gaps down through a
stop-loss, the order fills at the (worse) open, not at the stop level.

Slippage is then applied against the trader: buys fill ``slippage_bps`` above
the reference price and sells below it. Limit orders are capped at their limit
price, so slippage can erode price improvement but never breach the limit.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from simulator.config import FeeConfig
from simulator.data import Bar, DateLike, to_timestamp
from simulator.errors import InvalidOrderError, InvalidQuantityError


class Side(StrEnum):
    """Direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(StrEnum):
    """Lifecycle state of an order."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


def new_id() -> str:
    """Short random identifier for orders and trades."""
    return uuid.uuid4().hex[:12]


@dataclass
class Order:
    """An instruction to buy or sell a whole number of shares.

    Attributes:
        ticker: Instrument ticker (upper-cased on creation).
        side: BUY or SELL.
        quantity: Whole number of shares (> 0).
        created: Date the order was placed.
        order_type: MARKET, LIMIT or STOP.
        limit_price: Limit price (native currency), required for LIMIT orders.
        stop_price: Stop trigger price (native currency), required for STOP orders.
        id: Unique identifier.
        status: Current lifecycle state.
        reason: Why the order was rejected or cancelled, if applicable.
    """

    ticker: str
    side: Side
    quantity: int
    created: pd.Timestamp
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    id: str = field(default_factory=new_id)
    status: OrderStatus = OrderStatus.PENDING
    reason: str = ""

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()
        self.side = Side(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        self.created = to_timestamp(self.created)

    @property
    def trigger_price(self) -> float | None:
        """Limit or stop level, whichever applies (``None`` for market orders)."""
        if self.order_type is OrderType.LIMIT:
            return self.limit_price
        if self.order_type is OrderType.STOP:
            return self.stop_price
        return None

    def describe(self) -> str:
        """Human-readable one-line summary."""
        level = f" @ {self.trigger_price:g}" if self.trigger_price is not None else ""
        return f"{self.side} {self.quantity} {self.ticker} {self.order_type}{level}"


def make_order(
    ticker: str,
    side: Side | str,
    quantity: int | float,
    when: DateLike,
    order_type: OrderType | str = OrderType.MARKET,
    limit_price: float | None = None,
    stop_price: float | None = None,
) -> Order:
    """Create and validate an :class:`Order`.

    Raises:
        InvalidQuantityError: If quantity is not a positive whole number.
        InvalidOrderError: If side/type are unknown or a required price is missing.
    """
    try:
        side = Side(str(side).upper())
        order_type = OrderType(str(order_type).upper())
    except ValueError as exc:
        raise InvalidOrderError(str(exc)) from None
    order = Order(
        ticker=ticker,
        side=side,
        quantity=validate_quantity(quantity),
        created=to_timestamp(when),
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
    )
    validate_order_params(order)
    return order


def validate_quantity(quantity: object) -> int:
    """Return ``quantity`` as an int if it is a positive whole number.

    Accepts ints and integral floats (e.g. ``10.0``); rejects booleans,
    fractions, zero, negatives, NaN and non-numbers.

    Raises:
        InvalidQuantityError: If the quantity is not valid.
    """
    if isinstance(quantity, bool) or not isinstance(quantity, int | float):
        raise InvalidQuantityError(f"Quantity must be a number, got {quantity!r}")
    if isinstance(quantity, float) and (not math.isfinite(quantity) or not quantity.is_integer()):
        raise InvalidQuantityError(f"Quantity must be a whole number of shares, got {quantity}")
    if quantity <= 0:
        raise InvalidQuantityError(f"Quantity must be positive, got {quantity}")
    return int(quantity)


def validate_order_params(order: Order) -> None:
    """Check an order's internal consistency (not affordability).

    Raises:
        InvalidQuantityError: If quantity is invalid.
        InvalidOrderError: If a limit/stop price is missing or non-positive,
            or the ticker is blank.
    """
    validate_quantity(order.quantity)
    if not order.ticker:
        raise InvalidOrderError("Ticker is required")
    if order.order_type is OrderType.LIMIT:
        _require_positive(order.limit_price, "limit_price")
    elif order.order_type is OrderType.STOP:
        _require_positive(order.stop_price, "stop_price")


def _require_positive(value: float | None, name: str) -> None:
    if value is None or not math.isfinite(value) or value <= 0:
        raise InvalidOrderError(f"{name} must be a positive number")


def reference_fill_price(order: Order, bar: Bar) -> float | None:
    """Price at which ``order`` would fill on ``bar`` before slippage.

    Returns:
        The reference price, or ``None`` if the order does not trigger on this bar.
    """
    is_buy = order.side is Side.BUY
    if order.order_type is OrderType.MARKET:
        return bar.open
    if order.order_type is OrderType.LIMIT:
        limit = float(order.limit_price)  # type: ignore[arg-type]
        if is_buy:
            return min(bar.open, limit) if bar.low <= limit else None
        return max(bar.open, limit) if bar.high >= limit else None
    stop = float(order.stop_price)  # type: ignore[arg-type]
    if is_buy:
        return max(bar.open, stop) if bar.high >= stop else None
    return min(bar.open, stop) if bar.low <= stop else None


def apply_slippage(price: float, side: Side, fees: FeeConfig) -> float:
    """Move ``price`` against the trader by the configured slippage."""
    if side is Side.BUY:
        return price * (1 + fees.slippage_rate)
    return price * (1 - fees.slippage_rate)


def execution_price(order: Order, bar: Bar, fees: FeeConfig) -> tuple[float, float] | None:
    """Final fill price including slippage, plus the pre-slippage reference.

    Returns:
        ``(fill_price, reference_price)`` or ``None`` if the order doesn't trigger.
    """
    ref = reference_fill_price(order, bar)
    if ref is None:
        return None
    price = apply_slippage(ref, order.side, fees)
    if order.order_type is OrderType.LIMIT:
        limit = float(order.limit_price)  # type: ignore[arg-type]
        price = min(price, limit) if order.side is Side.BUY else max(price, limit)
    return price, ref


def max_affordable_quantity(
    cash_gbp: float, price_native: float, fx_rate: float, fees: FeeConfig
) -> int:
    """Largest whole number of shares a buy can afford, including fees.

    Args:
        cash_gbp: Available cash in GBP.
        price_native: Reference price per share in the instrument currency.
        fx_rate: Units of instrument currency per 1 GBP (1.0 for GBP stocks).
        fees: Fee configuration.
    """
    unit_cost = apply_slippage(price_native, Side.BUY, fees) / fx_rate
    budget = cash_gbp - fees.commission
    if budget <= 0 or unit_cost <= 0:
        return 0
    qty = math.floor(budget / unit_cost)
    # Guard against floating-point edge cases at the boundary.
    while qty > 0 and qty * unit_cost + fees.commission > cash_gbp + 1e-9:
        qty -= 1
    return qty
