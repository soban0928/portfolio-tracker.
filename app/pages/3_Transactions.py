"""Trade history, order log and CSV export."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui_helpers import gbp, get_db, page_setup, sidebar_portfolio

from simulator.db import export_trades_csv

page_setup("Transactions", "🧾")
pf = sidebar_portfolio()
db = get_db()

trades = pf.trades_frame()
if trades.empty:
    st.info("No trades yet.")
else:
    f1, f2 = st.columns(2)
    tickers = f1.multiselect("Ticker", sorted(trades["ticker"].unique()))
    sides = f2.multiselect("Side", ["BUY", "SELL"])
    view = trades
    if tickers:
        view = view[view["ticker"].isin(tickers)]
    if sides:
        view = view[view["side"].isin(sides)]

    c = st.columns(4)
    c[0].metric("Trades", len(view))
    c[1].metric("Commission", gbp(view["commission_gbp"].sum()))
    c[2].metric("Slippage", gbp(view["slippage_gbp"].sum()))
    c[3].metric("Realised P&L", gbp(view["realised_pnl_gbp"].sum()))

    st.dataframe(
        view.drop(columns=["id", "order_id"]),
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("Price (native)", format="%.4f"),
            "fx_rate": st.column_config.NumberColumn("GBP/USD", format="%.4f"),
            "gross_gbp": st.column_config.NumberColumn("Gross £", format="£%.2f"),
            "commission_gbp": st.column_config.NumberColumn("Commission £", format="£%.2f"),
            "slippage_gbp": st.column_config.NumberColumn("Slippage £", format="£%.2f"),
            "cash_flow_gbp": st.column_config.NumberColumn("Cash flow £", format="£%.2f"),
            "realised_pnl_gbp": st.column_config.NumberColumn("Realised £", format="£%.2f"),
        },
    )
    st.download_button(
        "⬇️ Export trade history (CSV)",
        export_trades_csv(pf),
        file_name=f"{pf.name.replace(' ', '_').lower()}_trades.csv",
        mime="text/csv",
    )

st.subheader("Order log")
if not pf.orders:
    st.caption("No orders yet.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "placed": o.created.date(),
                    "order": o.describe(),
                    "status": str(o.status),
                    "note": o.reason,
                }
                for o in reversed(pf.orders)
            ]
        ),
        hide_index=True,
    )

with st.expander("Danger zone"):
    st.caption("Reset clears all trades and orders and restores the starting cash.")
    confirm = st.checkbox(f"Yes, reset “{pf.name}”")
    if st.button("Reset portfolio", disabled=not confirm):
        db.reset_portfolio(pf.id)
        st.rerun()
