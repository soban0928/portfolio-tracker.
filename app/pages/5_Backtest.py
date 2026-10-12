"""Run and compare strategies against the benchmark."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from charts import drawdown_chart, line_chart
from ui_helpers import get_store, page_setup, sidebar_portfolio

from simulator import analytics as an
from simulator.backtest import BuyAndHold, Momentum, MovingAverageCrossover, compare_strategies
from simulator.config import FeeConfig

page_setup("Backtest", "🧪")
pf = sidebar_portfolio()
store = get_store()

first, last = store.date_range(store.benchmark)
stocks = [t for t in store.tickers if t != store.benchmark]

with st.form("backtest"):
    c1, c2 = st.columns([3, 2])
    tickers = c1.multiselect("Universe", stocks, default=stocks[: min(5, len(stocks))])
    dates = c2.date_input(
        "Date range",
        value=(max(first.date(), dt.date(2020, 1, 1)), last.date()),
        min_value=first.date(),
        max_value=last.date(),
    )
    c1, c2, c3, c4 = st.columns(4)
    cash = c1.number_input("Starting cash (£)", 100.0, 10_000_000.0, pf.starting_cash, 500.0)
    commission = c2.number_input("Commission (£)", 0.0, 100.0, pf.fees.commission, 0.5)
    slippage = c3.number_input("Slippage (bps)", 0.0, 500.0, pf.fees.slippage_bps, 1.0)
    rf = c4.number_input("Risk-free rate (%)", 0.0, 20.0, 0.0, 0.25) / 100

    st.markdown("**Strategies**")
    s1, s2, s3 = st.columns(3)
    use_bh = s1.checkbox("Buy & Hold (equal weight)", value=True)
    use_ma = s2.checkbox("Moving-average crossover", value=True)
    ma_short = s2.number_input("Short MA (days)", 2, 250, 50)
    ma_long = s2.number_input("Long MA (days)", 3, 400, 200)
    use_mom = s3.checkbox("Momentum (top-N)", value=True)
    lookback = s3.number_input("Lookback (days)", 5, 400, 126)
    top_n = s3.number_input("Hold top N", 1, 10, 2)
    rebalance = s3.number_input("Rebalance every (days)", 1, 126, 21)
    submitted = st.form_submit_button("Run backtest", type="primary")

if submitted:
    strategies = []
    errors = []
    if use_bh:
        strategies.append(BuyAndHold())
    if use_ma:
        try:
            strategies.append(MovingAverageCrossover(int(ma_short), int(ma_long)))
        except ValueError as exc:
            errors.append(f"Moving average: {exc}")
    if use_mom:
        strategies.append(Momentum(int(lookback), int(top_n), int(rebalance)))
    if not isinstance(dates, tuple) or len(dates) != 2:
        errors.append("Pick a start and end date.")
    if not tickers:
        errors.append("Pick at least one ticker.")
    if not strategies:
        errors.append("Pick at least one strategy.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Running backtests…"):
            try:
                values, metrics = compare_strategies(
                    store, strategies, tickers, dates[0], dates[1], cash,
                    FeeConfig(commission, slippage), rf,
                )  # fmt: skip
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["backtest_result"] = (values, metrics)

if "backtest_result" in st.session_state:
    values, metrics = st.session_state["backtest_result"]
    st.plotly_chart(line_chart(values, "Strategy value (GBP, after fees)", "Value", prefix="£"))
    table = metrics.copy()
    for col in ["total_return", "annualised_return", "annualised_volatility",
                "max_drawdown", "alpha", "benchmark_return"]:  # fmt: skip
        table[col] = table[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.2%}")
    for col in ["sharpe_ratio", "beta"]:
        table[col] = table[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    table["trades"] = table["trades"].astype(int)
    table = table.drop(columns=["benchmark_return"]).rename(
        columns={
            "total_return": "Total return",
            "annualised_return": "Annualised",
            "annualised_volatility": "Volatility",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "Max DD",
            "beta": "Beta",
            "alpha": "Alpha",
            "trades": "Trades",
        }
    )
    st.dataframe(table)
    st.plotly_chart(drawdown_chart(values.apply(an.drawdown_series), "Drawdown from peak"))
    st.caption(
        "Signals use each day's close and trade at the next day's open through the same "
        "engine as manual trading, so commission, slippage and FX apply identically. "
        "The benchmark (buy-and-hold SPY) pays the same fees."
    )
else:
    st.info("Choose a universe, date range and strategies, then press **Run backtest**.")
