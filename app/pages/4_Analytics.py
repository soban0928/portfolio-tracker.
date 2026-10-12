"""Risk and performance analytics vs the benchmark."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts import drawdown_chart, line_chart, returns_scatter
from ui_helpers import get_store, num, page_setup, pct, sidebar_portfolio

from simulator import analytics as an
from simulator.config import TRADING_DAYS

page_setup("Analytics", "📊")
pf = sidebar_portfolio()
store = get_store()

rf = (
    st.number_input(
        "Risk-free rate (annual, %)", min_value=0.0, max_value=20.0, value=0.0, step=0.25
    )
    / 100
)

history = pf.value_history(store) if pf.trades else None
if history is None or len(history) < 3:
    st.info(
        "Analytics need at least three days of history. Place a trade, then use "
        "**Advance simulation** on the Trade page to move the date forward."
    )
    st.stop()

values = history["total"]
bench_name = f"Benchmark ({store.benchmark})"
bench = an.benchmark_values(store, values.index[0], values.index[-1], pf.starting_cash)
p_sum = an.performance_summary(values, bench, rf)
b_sum = an.performance_summary(bench, bench, rf)

st.caption(
    f"{values.index[0].date()} → {values.index[-1].date()} · {len(values)} trading days · "
    f"values in GBP · annualised with {TRADING_DAYS} days"
)
cols = st.columns(6)
cols[0].metric("Total return", pct(p_sum["total_return"]),
               f"{(p_sum['total_return'] - b_sum['total_return']):+.2%} vs SPY")  # fmt: skip
cols[1].metric("Volatility (ann.)", pct(p_sum["annualised_volatility"], signed=False))
cols[2].metric("Sharpe ratio", num(p_sum["sharpe_ratio"]))
cols[3].metric("Max drawdown", pct(p_sum["max_drawdown"]))
cols[4].metric("Beta", num(p_sum["beta"]))
cols[5].metric("Alpha (ann.)", pct(p_sum["alpha"]))

labels = {
    "total_return": "Total return",
    "annualised_return": "Annualised return",
    "annualised_volatility": "Annualised volatility",
    "sharpe_ratio": "Sharpe ratio",
    "max_drawdown": "Max drawdown",
    "beta": "Beta",
    "alpha": "Alpha (annualised)",
}
ratio_keys = {"sharpe_ratio", "beta"}
table = {
    labels[k]: {
        "Portfolio": num(p_sum[k]) if k in ratio_keys else pct(p_sum[k]),
        bench_name: num(b_sum[k]) if k in ratio_keys else pct(b_sum[k]),
    }
    for k in labels
}

left, right = st.columns([3, 2])
with left:
    rebased = an.normalise(values).rename("Portfolio").to_frame()
    rebased[bench_name] = an.normalise(bench)
    st.plotly_chart(line_chart(rebased, "Growth of 100 (rebased)", "Index", y_format=",.1f"))
with right:
    st.dataframe(pd.DataFrame(table).T)

dd = an.drawdown_series(values).rename("Portfolio").to_frame()
dd[bench_name] = an.drawdown_series(bench)
r_p, r_b = an.daily_returns(values), an.daily_returns(bench)
alpha_daily = (r_p.mean() - p_sum["beta"] * r_b.mean()) if p_sum["beta"] == p_sum["beta"] else 0
left, right = st.columns(2)
with left:
    st.plotly_chart(drawdown_chart(dd, "Drawdown from peak"))
with right:
    st.plotly_chart(returns_scatter(r_p, r_b, p_sum["beta"], alpha_daily))

with st.expander("How these are calculated"):
    st.markdown(
        """
- **Total return**: final value ÷ starting value − 1 over the period shown.
- **Volatility**: sample standard deviation of daily returns × √252.
- **Sharpe ratio**: mean daily excess return ÷ its standard deviation × √252, using a
  daily risk-free rate of (annual rate ÷ 252).
- **Max drawdown**: the largest fall from a running peak.
- **Beta**: covariance of daily returns with SPY ÷ variance of SPY returns.
- **Alpha**: Jensen's alpha, mean(rₚ − r_f) − β·mean(r_b − r_f), annualised × 252.

The benchmark is SPY converted to GBP (no fees), starting with the same cash on the
first trade date.
"""
    )
