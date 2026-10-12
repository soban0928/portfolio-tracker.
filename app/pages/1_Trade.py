"""Trade ticket: place market / limit / stop orders at historical prices."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts import candlestick_chart
from ui_helpers import gbp, get_db, get_store, page_setup, sidebar_portfolio

from simulator.errors import SimulatorError
from simulator.orders import OrderType, make_order, max_affordable_quantity

page_setup("Trade", "🛒")
pf = sidebar_portfolio()
store = get_store()
db = get_db()

first, last = store.date_range(store.benchmark)
min_date = (pf.clock or first).date()
flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

left, right = st.columns([2, 3], gap="large")

with left:
    trade_date = st.date_input(
        "Trade date",
        value=min_date,
        min_value=min_date,
        max_value=last.date(),
        help="Orders must be placed in chronological order, so dates before the "
        "portfolio's simulation date are disabled.",
    )
    ticker = st.selectbox("Ticker", store.tickers, key="ticker")
    currency = store.currency(ticker)
    bar = store.get_bar(ticker, trade_date) if store.has_bar(ticker, trade_date) else None

    side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
    order_type = st.radio(
        "Order type",
        [t.value for t in OrderType],
        horizontal=True,
        format_func=lambda t: {"MARKET": "Market", "LIMIT": "Limit", "STOP": "Stop"}[t],
    )
    default_px = float(round(bar.open, 2)) if bar else 1.0
    level = None
    if order_type != "MARKET":
        label = "Limit price" if order_type == "LIMIT" else "Stop price"
        level = st.number_input(f"{label} ({currency})", min_value=0.01, value=default_px)
    quantity = st.number_input("Quantity (shares)", min_value=1, value=10, step=1)

    if bar is None:
        st.warning(f"No price data for {ticker} on {trade_date} (weekend, holiday or no data).")
    else:
        fx = store.fx_rate(trade_date) if currency != "GBP" else 1.0
        ref = level if (level and order_type != "MARKET") else bar.open
        est_native = quantity * ref
        slip = pf.fees.slippage_rate * (1 if side == "BUY" else -1)
        est_gbp = est_native * (1 + slip) / fx
        est_total = est_gbp + pf.fees.commission if side == "BUY" else est_gbp - pf.fees.commission
        st.markdown(
            f"**Estimate:** {quantity} × {ref:,.2f} {currency}"
            + (f" ÷ {fx:.4f} GBP/USD" if currency != "GBP" else "")
            + f" → **{gbp(est_total)}** {'cost' if side == 'BUY' else 'proceeds'} incl. fees"
        )
        if side == "BUY":
            st.caption(
                f"Cash {gbp(pf.cash)} · max affordable at the open: "
                f"{max_affordable_quantity(pf.cash, bar.open, fx, pf.fees)} shares"
            )
        else:
            st.caption(f"Shares held: {pf.shares_held(ticker)}")

    if st.button("Place order", type="primary", disabled=bar is None):
        try:
            order = make_order(
                ticker,
                side,
                int(quantity),
                trade_date,
                order_type,
                limit_price=level if order_type == "LIMIT" else None,
                stop_price=level if order_type == "STOP" else None,
            )
            trade = pf.place_order(order, store)
        except SimulatorError as exc:
            st.error(f"Order rejected: {exc}")
        else:
            db.save_portfolio(pf)
            if trade is None:
                msg = ("info", f"{order.describe()} is pending; it will be checked on later days.")
            else:
                msg = (
                    "success",
                    f"Filled: {trade.side} {trade.quantity} {trade.ticker} @ "
                    f"{trade.price:,.4f} {trade.currency} on {trade.date.date()} "
                    f"(cash flow {gbp(trade.cash_flow)}).",
                )
            st.session_state["flash"] = msg
            st.rerun()

with right:
    end = pd.Timestamp(trade_date)
    hist = store.frame(ticker)
    hist = hist[hist.index <= end].tail(90)
    if not hist.empty:
        st.plotly_chart(candlestick_chart(hist, ticker, currency))
    if bar is not None:
        cols = st.columns(4)
        for col, (name, val) in zip(
            cols,
            [("Open", bar.open), ("High", bar.high), ("Low", bar.low), ("Close", bar.close)],
            strict=True,
        ):
            col.metric(name, f"{val:,.2f}")
        st.caption(
            "Market orders fill at the open. Limit buys fill if the low reaches the limit; "
            "stop-loss sells trigger if the low reaches the stop and fill at the lower of "
            "the stop and the open (gaps fill at the open)."
        )

st.divider()
st.subheader("Pending orders")
pending = pf.pending_orders
if not pending:
    st.caption("No pending limit or stop orders.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "id": o.id,
                    "placed": o.created.date(),
                    "order": o.describe(),
                    "currency": store.currency(o.ticker),
                }
                for o in pending
            ]
        ),
        hide_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        to_cancel = st.selectbox(
            "Cancel order", [o.id for o in pending],
            format_func=lambda i: next(o.describe() for o in pending if o.id == i),
        )  # fmt: skip
        if st.button("Cancel selected order"):
            pf.cancel_order(to_cancel)
            db.save_portfolio(pf)
            st.session_state["flash"] = ("info", "Order cancelled.")
            st.rerun()

st.subheader("Advance simulation")
st.caption(
    "Move the portfolio's date forward to the trade date above, checking pending orders "
    "on each trading day. Analytics and charts run up to the simulation date."
)
if st.button(f"Advance to {trade_date}"):
    try:
        filled = pf.process_pending(store, trade_date)
    except SimulatorError as exc:
        st.error(str(exc))
    else:
        db.save_portfolio(pf)
        st.session_state["flash"] = (
            "success",
            f"Advanced to {trade_date}. {len(filled)} pending order(s) filled.",
        )
        st.rerun()
