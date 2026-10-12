"""Portfolio engine: cash, positions, trade execution, P&L and valuation.

Accounting conventions
----------------------
* The portfolio's base currency is GBP. Every trade is converted to GBP at the
  GBP/USD rate of the trade date, and that GBP amount is what's booked.
* **Cost basis is weighted-average cost (WAC) in GBP, including buy fees.**
  Each buy adds ``quantity * price / fx + commission`` to the position's total
  cost. Average cost = total cost / shares held.
* A sell removes ``quantity * average cost`` from the total cost (so the average
  cost of the remaining shares is unchanged) and books
  ``realised P&L = net proceeds (after commission) - quantity * average cost``.
* Unrealised P&L = current GBP market value - remaining total cost. Because cost
  is locked at historical FX rates, unrealised P&L on US stocks includes
  currency gains/losses.
* Slippage is embedded in the fill price, so it's automatically reflected in
  cost basis and proceeds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import pandas as pd

from simulator.config import BASE_CURRENCY, STARTING_CASH, FeeConfig
from simulator.data import Bar, DateLike, PriceStore, to_timestamp
from simulator.errors import (
    InsufficientCashError,
    InsufficientSharesError,
    InvalidOrderError,
    SimulatorError,
)
from simulator.orders import (
    Order,
    OrderStatus,
    OrderType,
    Side,
    execution_price,
    new_id,
    validate_order_params,
)

EPS = 1e-9


@dataclass(frozen=True)
class Trade:
    """An executed fill. This is the record that gets persisted.

    Attributes:
        date: Trade date.
        ticker: Instrument ticker.
        side: BUY or SELL.
        quantity: Shares traded.
        price: Executed price per share in ``currency`` (including slippage).
        currency: Instrument currency.
        fx_rate: Units of ``currency`` per 1 GBP on the trade date (1.0 for GBP).
        commission: Commission charged, in GBP.
        reference_price: Pre-slippage price (native), for fee reporting.
        order_type: Type of the order that produced this fill.
        realised_pnl: Realised P&L in GBP (sells only; 0 for buys). Filled in
            when the trade is applied to a portfolio.
        order_id: Identifier of the originating order.
        id: Unique trade identifier.
    """

    date: pd.Timestamp
    ticker: str
    side: Side
    quantity: int
    price: float
    currency: str = "USD"
    fx_rate: float = 1.0
    commission: float = 0.0
    reference_price: float | None = None
    order_type: OrderType = OrderType.MARKET
    realised_pnl: float = 0.0
    order_id: str = ""
    id: str = field(default_factory=new_id)

    @property
    def gross_base(self) -> float:
        """Trade notional in GBP, excluding commission."""
        return self.quantity * self.price / self.fx_rate

    @property
    def cash_flow(self) -> float:
        """Signed change to GBP cash: negative for buys, positive for sells."""
        if self.side is Side.BUY:
            return -(self.gross_base + self.commission)
        return self.gross_base - self.commission

    @property
    def slippage_cost(self) -> float:
        """Cost of slippage in GBP (0 if no reference price recorded)."""
        if self.reference_price is None:
            return 0.0
        return self.quantity * abs(self.price - self.reference_price) / self.fx_rate

    def to_dict(self) -> dict[str, object]:
        """Flat dict suitable for DataFrames and CSV export."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "date": self.date.date().isoformat(),
            "ticker": self.ticker,
            "side": str(self.side),
            "order_type": str(self.order_type),
            "quantity": self.quantity,
            "price": self.price,
            "currency": self.currency,
            "fx_rate": self.fx_rate,
            "gross_gbp": self.gross_base,
            "commission_gbp": self.commission,
            "slippage_gbp": self.slippage_cost,
            "cash_flow_gbp": self.cash_flow,
            "realised_pnl_gbp": self.realised_pnl,
        }


@dataclass
class Position:
    """Shares held in one instrument, with weighted-average cost in GBP."""

    ticker: str
    currency: str
    quantity: int = 0
    cost_basis: float = 0.0  # total GBP cost of shares held, incl. buy fees
    cost_native: float = 0.0  # total native-currency cost (no fees), for display

    @property
    def avg_cost(self) -> float:
        """Average GBP cost per share (including buy commission)."""
        return self.cost_basis / self.quantity if self.quantity else 0.0

    @property
    def avg_price_native(self) -> float:
        """Average purchase price per share in the instrument's currency."""
        return self.cost_native / self.quantity if self.quantity else 0.0


class Portfolio:
    """A virtual trading account denominated in GBP.

    Orders must be placed in chronological order: the portfolio keeps a
    ``clock`` (latest simulated date) and rejects orders dated before it.
    """

    def __init__(
        self,
        name: str = "My Portfolio",
        starting_cash: float = STARTING_CASH,
        fees: FeeConfig | None = None,
        id: int | None = None,
    ) -> None:
        """Create an empty portfolio holding only cash."""
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.id = id
        self.name = name
        self.starting_cash = float(starting_cash)
        self.fees = fees or FeeConfig()
        self.cash = float(starting_cash)
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.orders: list[Order] = []
        self.realised_pnl = 0.0
        self.clock: pd.Timestamp | None = None

    # ---------------------------------------------------------------- replay
    @classmethod
    def from_records(
        cls,
        trades: Iterable[Trade],
        orders: Iterable[Order] = (),
        clock: DateLike | None = None,
        **kwargs: object,
    ) -> Portfolio:
        """Rebuild a portfolio by replaying stored trades (used by persistence).

        Realised P&L is recomputed during replay, so state is always consistent
        with the transaction log.
        """
        pf = cls(**kwargs)  # type: ignore[arg-type]
        for trade in sorted(trades, key=lambda t: t.date):
            pf.apply_trade(trade)
        pf.orders = list(orders)
        if clock is not None:
            pf.clock = to_timestamp(clock)
        elif pf.trades:
            pf.clock = pf.trades[-1].date
        return pf

    # ---------------------------------------------------------------- orders
    @property
    def pending_orders(self) -> list[Order]:
        """Open limit/stop orders awaiting a fill, oldest first."""
        return [o for o in self.orders if o.status is OrderStatus.PENDING]

    def shares_held(self, ticker: str) -> int:
        """Number of shares currently held in ``ticker``."""
        pos = self.positions.get(ticker.upper())
        return pos.quantity if pos else 0

    def place_order(self, order: Order, store: PriceStore) -> Trade | None:
        """Submit an order dated ``order.created``.

        Pending orders are first processed up to and including that date. A
        market order then fills at that day's open; a limit/stop order fills
        immediately if that day's bar triggers it, otherwise it stays pending.

        Returns:
            The resulting :class:`Trade`, or ``None`` if the order is pending.

        Raises:
            OrderValidationError: For invalid quantity, insufficient cash or
                shares, or orders dated before the portfolio clock.
            NoPriceDataError: If the ticker has no bar on the order date.
        """
        validate_order_params(order)
        when = order.created
        if self.clock is not None and when < self.clock:
            raise InvalidOrderError(
                f"Order date {when.date()} is before the portfolio's current date "
                f"{self.clock.date()}; orders must be placed in chronological order"
            )
        bar = store.get_bar(order.ticker, when)  # raises NoPriceDataError
        self.process_pending(store, when)

        if order.side is Side.SELL and order.quantity > self.shares_held(order.ticker):
            raise InsufficientSharesError(
                f"Cannot sell {order.quantity} {order.ticker}: only "
                f"{self.shares_held(order.ticker)} held (short selling is not allowed)"
            )

        trade = self._try_fill(order, bar, store)
        if trade is None:
            if order.side is Side.BUY:
                self._check_cash(order, float(order.trigger_price or bar.open), store, when)
            order.status = OrderStatus.PENDING
        self.orders.append(order)
        return trade

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order by id.

        Raises:
            InvalidOrderError: If no pending order has that id.
        """
        for order in self.pending_orders:
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                order.reason = "Cancelled by user"
                return order
        raise InvalidOrderError(f"No pending order with id {order_id}")

    def process_pending(self, store: PriceStore, through: DateLike) -> list[Trade]:
        """Evaluate pending orders day by day up to and including ``through``.

        Each order is checked on trading days strictly after the day it was
        placed. Orders that trigger but can't be executed (e.g. not enough cash
        at that point) are marked REJECTED with a reason.

        Returns:
            Trades filled during processing, in chronological order.
        """
        end = to_timestamp(through)
        filled: list[Trade] = []
        pending = self.pending_orders
        if pending:
            start = min(o.created for o in pending)
            days = sorted(
                set().union(
                    *(
                        store.frame(t).index[
                            (store.frame(t).index > start) & (store.frame(t).index <= end)
                        ]
                        for t in {o.ticker for o in pending}
                    )
                )
            )
            for day in days:
                for order in self.pending_orders:
                    if order.created >= day or not store.has_bar(order.ticker, day):
                        continue
                    try:
                        trade = self._try_fill(order, store.get_bar(order.ticker, day), store)
                    except SimulatorError as exc:
                        order.status = OrderStatus.REJECTED
                        order.reason = str(exc)
                        continue
                    if trade is not None:
                        filled.append(trade)
        if self.clock is None or end > self.clock:
            self.clock = end
        return filled

    def _try_fill(self, order: Order, bar: Bar, store: PriceStore) -> Trade | None:
        """Execute ``order`` on ``bar`` if it triggers; validate and book it."""
        result = execution_price(order, bar, self.fees)
        if result is None:
            return None
        price, ref = result
        currency = store.currency(order.ticker)
        fx = store.fx_rate(bar.date) if currency != BASE_CURRENCY else 1.0
        trade = Trade(
            date=bar.date,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=price,
            currency=currency,
            fx_rate=fx,
            commission=self.fees.commission,
            reference_price=ref,
            order_type=order.order_type,
            order_id=order.id,
        )
        trade = self.apply_trade(trade)
        order.status = OrderStatus.FILLED
        return trade

    def _check_cash(
        self, order: Order, price: float, store: PriceStore, when: pd.Timestamp
    ) -> None:
        """Pre-check that a buy at ``price`` (native) would be affordable."""
        fx = store.fx_rate(when) if store.currency(order.ticker) != BASE_CURRENCY else 1.0
        est = order.quantity * price * (1 + self.fees.slippage_rate) / fx + self.fees.commission
        if est > self.cash + EPS:
            raise InsufficientCashError(
                f"Estimated cost £{est:,.2f} exceeds available cash £{self.cash:,.2f}"
            )

    # ------------------------------------------------------------ accounting
    def apply_trade(self, trade: Trade) -> Trade:
        """Validate a fully-priced trade and update cash, positions and P&L.

        Returns:
            The trade as booked (sells get their ``realised_pnl`` filled in).

        Raises:
            InsufficientCashError: If a buy costs more than available cash.
            InsufficientSharesError: If a sell exceeds shares held.
        """
        ticker = trade.ticker.upper()
        pos = self.positions.get(ticker)
        if trade.side is Side.BUY:
            cost = -trade.cash_flow
            if cost > self.cash + EPS:
                raise InsufficientCashError(
                    f"Buying {trade.quantity} {ticker} costs £{cost:,.2f} "
                    f"but only £{self.cash:,.2f} cash is available"
                )
            if pos is None:
                pos = self.positions[ticker] = Position(ticker, trade.currency)
            pos.quantity += trade.quantity
            pos.cost_basis += cost
            pos.cost_native += trade.quantity * trade.price
            self.cash -= cost
            booked = trade
        else:
            held = pos.quantity if pos else 0
            if pos is None or trade.quantity > held:
                raise InsufficientSharesError(
                    f"Cannot sell {trade.quantity} {ticker}: only {held} held "
                    "(short selling is not allowed)"
                )
            avg_cost, avg_native = pos.avg_cost, pos.avg_price_native
            proceeds = trade.cash_flow
            realised = proceeds - trade.quantity * avg_cost
            pos.quantity -= trade.quantity
            pos.cost_basis -= trade.quantity * avg_cost
            pos.cost_native -= trade.quantity * avg_native
            if pos.quantity == 0:
                del self.positions[ticker]
            self.cash += proceeds
            self.realised_pnl += realised
            booked = replace(trade, realised_pnl=realised)
        self.trades.append(booked)
        if self.clock is None or booked.date > self.clock:
            self.clock = booked.date
        return booked

    # -------------------------------------------------------------- valuation
    def _as_of(self, when: DateLike | None) -> pd.Timestamp:
        if when is not None:
            return to_timestamp(when)
        if self.clock is None:
            raise ValueError("No valuation date: pass `when` or place a trade first")
        return self.clock

    def holdings(self, store: PriceStore, when: DateLike | None = None) -> pd.DataFrame:
        """Current holdings marked to market on ``when`` (default: portfolio clock).

        Columns: ticker, currency, quantity, avg_price, price, avg_cost_gbp,
        cost_basis_gbp, market_value_gbp, unrealised_pnl_gbp, unrealised_pct, weight.
        """
        cols = [
            "ticker", "currency", "quantity", "avg_price", "price", "avg_cost_gbp",
            "cost_basis_gbp", "market_value_gbp", "unrealised_pnl_gbp", "unrealised_pct",
            "weight",
        ]  # fmt: skip
        if not self.positions:
            return pd.DataFrame(columns=cols)
        ts = self._as_of(when)
        rows = []
        for ticker, pos in sorted(self.positions.items()):
            price = store.close_on_or_before(ticker, ts)
            value = store.to_base(pos.quantity * price, pos.currency, ts)
            pnl = value - pos.cost_basis
            rows.append(
                {
                    "ticker": ticker,
                    "currency": pos.currency,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price_native,
                    "price": price,
                    "avg_cost_gbp": pos.avg_cost,
                    "cost_basis_gbp": pos.cost_basis,
                    "market_value_gbp": value,
                    "unrealised_pnl_gbp": pnl,
                    "unrealised_pct": pnl / pos.cost_basis if pos.cost_basis else 0.0,
                }
            )
        df = pd.DataFrame(rows)
        total = df["market_value_gbp"].sum() + self.cash
        df["weight"] = df["market_value_gbp"] / total if total else 0.0
        return df[cols]

    def market_value(self, store: PriceStore, when: DateLike | None = None) -> float:
        """GBP market value of all positions (excluding cash)."""
        h = self.holdings(store, when)
        return float(h["market_value_gbp"].sum()) if not h.empty else 0.0

    def total_value(self, store: PriceStore, when: DateLike | None = None) -> float:
        """Cash plus market value of positions, in GBP."""
        return self.cash + self.market_value(store, when)

    def unrealised_pnl(self, store: PriceStore, when: DateLike | None = None) -> float:
        """Unrealised P&L in GBP across open positions."""
        h = self.holdings(store, when)
        return float(h["unrealised_pnl_gbp"].sum()) if not h.empty else 0.0

    def total_return(self, store: PriceStore, when: DateLike | None = None) -> float:
        """Total return since inception as a fraction (0.05 = +5%)."""
        return self.total_value(store, when) / self.starting_cash - 1

    @property
    def total_commission(self) -> float:
        """Total commission paid, in GBP."""
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        """Total slippage cost, in GBP."""
        return sum(t.slippage_cost for t in self.trades)

    def summary(self, store: PriceStore, when: DateLike | None = None) -> dict[str, float]:
        """Headline numbers for dashboards."""
        if self.clock is None and when is None:
            mv = unreal = 0.0
        else:
            mv = self.market_value(store, when)
            unreal = self.unrealised_pnl(store, when)
        total = self.cash + mv
        return {
            "cash": self.cash,
            "market_value": mv,
            "total_value": total,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": unreal,
            "total_pnl": total - self.starting_cash,
            "total_return": total / self.starting_cash - 1,
            "fees_paid": self.total_commission + self.total_slippage,
        }

    def trades_frame(self) -> pd.DataFrame:
        """All trades as a DataFrame (see :meth:`Trade.to_dict` for columns)."""
        if not self.trades:
            return pd.DataFrame(columns=list(Trade(pd.Timestamp(0), "X", Side.BUY, 1, 1).to_dict()))
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def value_history(
        self, store: PriceStore, start: DateLike | None = None, end: DateLike | None = None
    ) -> pd.DataFrame:
        """Daily end-of-day portfolio value, reconstructed from the trade log.

        Args:
            store: Price source.
            start: First date (default: first trade date).
            end: Last date (default: portfolio clock).

        Returns:
            DataFrame indexed by date with columns ``cash``, ``holdings``, ``total``.
        """
        if start is None:
            if not self.trades:
                return pd.DataFrame(columns=["cash", "holdings", "total"])
            start = self.trades[0].date
        end_ts = self._as_of(end)
        dates = store.trading_days(start, end_ts)
        if len(dates) == 0:
            return pd.DataFrame(columns=["cash", "holdings", "total"])

        def as_of(series: pd.Series) -> pd.Series:
            """Align an event series to ``dates`` with carry-forward semantics."""
            s = series.groupby(level=0).sum().cumsum()
            return s.reindex(s.index.union(dates)).ffill().fillna(0.0).reindex(dates)

        trades = self.trades_frame()
        if trades.empty:
            cash = pd.Series(self.starting_cash, index=dates)
            holdings = pd.Series(0.0, index=dates)
        else:
            trades["date"] = pd.to_datetime(trades["date"])
            cash = self.starting_cash + as_of(trades.set_index("date")["cash_flow_gbp"])
            holdings = pd.Series(0.0, index=dates)
            signed = trades["quantity"].where(trades["side"] == "BUY", -trades["quantity"])
            trades["signed_qty"] = signed
            for ticker, grp in trades.groupby("ticker"):
                qty = as_of(grp.set_index("date")["signed_qty"].astype(float))
                px = store.closes_in_base(str(ticker))
                px = px.reindex(px.index.union(dates)).ffill().reindex(dates).fillna(0.0)
                holdings = holdings + qty * px
        return pd.DataFrame({"cash": cash, "holdings": holdings, "total": cash + holdings})
