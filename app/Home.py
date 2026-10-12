"""Streamlit entry point: overview of the active portfolio."""

from __future__ import annotations

import streamlit as st
from ui_helpers import gbp, get_store, page_setup, pct, sidebar_portfolio

page_setup("Stock Trading Simulator")
pf = sidebar_portfolio()
store = get_store()

st.markdown(
    f"Trade US and UK stocks with **£{pf.starting_cash:,.0f}** of virtual cash at "
    "historical prices. Every fill pays commission and slippage, US trades convert at "
    "that day's GBP/USD rate, and your portfolio is saved to SQLite."
)

summary = pf.summary(store)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total value", gbp(summary["total_value"]), pct(summary["total_return"]))
c2.metric("Cash", gbp(summary["cash"]))
c3.metric("Realised P&L", gbp(summary["realised_pnl"]))
c4.metric("Unrealised P&L", gbp(summary["unrealised_pnl"]))

first, last = store.date_range(store.benchmark)
st.subheader("Getting started")
st.markdown(
    f"""
1. **Trade**: pick a date between {first.date()} and {last.date()}, a ticker and an order
   type (market, limit or stop-loss).
2. **Portfolio**: holdings, allocation and performance against the S&P 500 (SPY).
3. **Transactions**: full trade history with CSV export.
4. **Analytics**: volatility, Sharpe ratio, max drawdown, beta and alpha.
5. **Backtest**: run buy-and-hold, moving-average crossover and momentum strategies.

Available tickers: {", ".join(f"`{t}`" for t in store.tickers)}
"""
)
