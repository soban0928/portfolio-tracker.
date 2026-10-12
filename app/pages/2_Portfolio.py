"""Portfolio overview: KPIs, performance vs benchmark, allocation and holdings."""

from __future__ import annotations

import streamlit as st
from charts import allocation_chart, performance_chart, pnl_bar_chart
from ui_helpers import gbp, get_store, page_setup, pct, sidebar_portfolio

from simulator.analytics import benchmark_values

page_setup("Portfolio", "💼")
pf = sidebar_portfolio()
store = get_store()

s = pf.summary(store)
top, bottom = st.columns(3), st.columns(3)
top[0].metric("Total value", gbp(s["total_value"]), pct(s["total_return"]), border=True)
top[1].metric("Cash", gbp(s["cash"]), border=True)
top[2].metric("Invested", gbp(s["market_value"]), border=True)
bottom[0].metric("Realised P&L", gbp(s["realised_pnl"]), border=True)
bottom[1].metric("Unrealised P&L", gbp(s["unrealised_pnl"]), border=True)
bottom[2].metric(
    "Fees paid", gbp(s["fees_paid"]), help="Commission plus slippage cost", border=True
)

if not pf.trades:
    st.info("No trades yet. Head to the **Trade** page to place your first order.")
    st.stop()

history = pf.value_history(store)
bench = benchmark_values(store, history.index[0], history.index[-1], pf.starting_cash)
bench = bench.rename(f"Benchmark ({store.benchmark})")
st.plotly_chart(performance_chart(history["total"], bench))
st.caption(
    f"Benchmark: £{pf.starting_cash:,.0f} invested in {store.benchmark} (in GBP) on "
    f"{history.index[0].date()}, no fees. Valued through the simulation date "
    f"{pf.clock.date()}."
)

holdings = pf.holdings(store)
left, right = st.columns(2)
with left:
    st.plotly_chart(allocation_chart(holdings, pf.cash))
with right:
    if holdings.empty:
        st.caption("No open positions.")
    else:
        st.plotly_chart(pnl_bar_chart(holdings))

st.subheader("Holdings")
if holdings.empty:
    st.caption("No open positions; everything is in cash.")
else:
    st.dataframe(
        holdings,
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "currency": "Ccy",
            "quantity": st.column_config.NumberColumn("Shares", format="%d"),
            "avg_price": st.column_config.NumberColumn("Avg price (native)", format="%.2f"),
            "price": st.column_config.NumberColumn("Last price (native)", format="%.2f"),
            "avg_cost_gbp": st.column_config.NumberColumn("Avg cost £", format="£%.4f"),
            "cost_basis_gbp": st.column_config.NumberColumn("Cost basis £", format="£%.2f"),
            "market_value_gbp": st.column_config.NumberColumn("Value £", format="£%.2f"),
            "unrealised_pnl_gbp": st.column_config.NumberColumn("Unrealised £", format="£%.2f"),
            "unrealised_pct": st.column_config.NumberColumn("Unrealised %", format="percent"),
            "weight": st.column_config.NumberColumn("Weight", format="percent"),
        },
    )
    st.caption(
        "Cost basis is weighted-average cost in GBP, including buy commission, at the "
        "FX rate on each purchase date. Unrealised P&L therefore includes currency moves."
    )
