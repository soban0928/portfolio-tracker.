"""Plotly figure builders. Pure functions: DataFrames in, figures out.

Colours come from a CVD-validated categorical palette (fixed slot order) so
a series keeps its colour across every chart. Text stays in neutral ink; only
marks carry series colour.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Categorical slots, in fixed order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GAIN = "#2a78d6"  # diverging pair: blue (gain) <-> red (loss)
LOSS = "#e34948"
NEUTRAL = "#898781"  # cash / muted
BENCHMARK = "#52514e"  # benchmark reference line: neutral ink, dotted, same in every chart
UP, DOWN = "#0ca30c", "#d03b3b"  # status colours for candles

_LAYOUT = dict(
    margin=dict(l=8, r=8, t=72, b=8),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
)


def _base(title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, y=0.98, yanchor="top", yref="container", font=dict(size=15)),
        height=height,
        **_LAYOUT,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def line_chart(
    series: pd.DataFrame | pd.Series,
    title: str,
    y_title: str = "",
    y_format: str = ",.0f",
    prefix: str = "",
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """Multi-series line chart on a single shared axis.

    Args:
        series: One column per line, indexed by date.
        title: Chart title.
        y_title: Y-axis label.
        y_format: d3 format for ticks/hover values.
        prefix: Value prefix such as "£".
        colors: Optional explicit colour per column name; others take slots in order.
    """
    df = series.to_frame() if isinstance(series, pd.Series) else series
    fig = _base(title)
    colors = colors or {}
    slot = 0
    for col in df.columns:
        is_bench = str(col).startswith("Benchmark")
        color = colors.get(str(col), BENCHMARK if is_bench else None)
        if color is None:
            color = SERIES[slot % len(SERIES)]
            slot += 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=str(col),
                mode="lines",
                line=dict(color=color, width=2, dash="dot" if is_bench else "solid"),
                hovertemplate=f"{prefix}%{{y:{y_format}}}<extra>{col}</extra>",
            )
        )
    fig.update_yaxes(title=y_title, tickformat=y_format, tickprefix=prefix)
    fig.update_layout(showlegend=df.shape[1] > 1)
    return fig


def performance_chart(values: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    """Portfolio value (GBP) against the same cash invested in the benchmark."""
    df = values.rename("Portfolio").to_frame()
    colors = {"Portfolio": SERIES[0]}
    if benchmark is not None and not benchmark.empty:
        name = str(benchmark.name or "Benchmark")
        df[name] = benchmark
        colors[name] = BENCHMARK
    return line_chart(df, "Portfolio value vs benchmark", "Value (GBP)", prefix="£",
                      colors=colors)  # fmt: skip


def allocation_chart(holdings: pd.DataFrame, cash: float) -> go.Figure:
    """Horizontal bars of portfolio weight per holding, plus cash."""
    rows = holdings[["ticker", "market_value_gbp"]].copy() if not holdings.empty else None
    data = pd.concat(
        [rows, pd.DataFrame({"ticker": ["Cash"], "market_value_gbp": [cash]})]
        if rows is not None
        else [pd.DataFrame({"ticker": ["Cash"], "market_value_gbp": [cash]})]
    )
    total = data["market_value_gbp"].sum()
    data["weight"] = data["market_value_gbp"] / total if total else 0.0
    data = data.sort_values("weight")
    fig = _base("Asset allocation", height=max(220, 48 * len(data) + 80))
    fig.add_trace(
        go.Bar(
            x=data["weight"],
            y=data["ticker"],
            orientation="h",
            marker=dict(
                color=[NEUTRAL if t == "Cash" else SERIES[0] for t in data["ticker"]],
                cornerradius=4,
            ),
            text=[f"{w:.1%}" for w in data["weight"]],
            textposition="outside",
            customdata=data["market_value_gbp"],
            hovertemplate="%{y}: %{x:.1%} (£%{customdata:,.2f})<extra></extra>",
        )
    )
    fig.update_xaxes(tickformat=".0%", range=[0, min(1.15, data["weight"].max() * 1.25 + 0.01)])
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35)
    return fig


def pnl_bar_chart(holdings: pd.DataFrame) -> go.Figure:
    """Unrealised P&L per holding (blue gains, red losses)."""
    data = holdings.sort_values("unrealised_pnl_gbp")
    fig = _base("Unrealised P&L by holding", height=max(220, 48 * len(data) + 80))
    fig.add_trace(
        go.Bar(
            x=data["unrealised_pnl_gbp"],
            y=data["ticker"],
            orientation="h",
            marker=dict(
                color=[GAIN if v >= 0 else LOSS for v in data["unrealised_pnl_gbp"]],
                cornerradius=4,
            ),
            customdata=data["unrealised_pct"],
            hovertemplate="%{y}: £%{x:,.2f} (%{customdata:+.1%})<extra></extra>",
        )
    )
    fig.update_xaxes(tickprefix="£", tickformat=",.0f", zeroline=True, zerolinewidth=1)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35)
    return fig


def drawdown_chart(drawdowns: pd.DataFrame | pd.Series, title: str = "Drawdown") -> go.Figure:
    """Drawdown from peak over time (one line per column)."""
    df = drawdowns.to_frame() if isinstance(drawdowns, pd.Series) else drawdowns
    fig = line_chart(df, title, "Drawdown", y_format=".1%")
    if df.shape[1] == 1:
        fig.update_traces(fill="tozeroy", line_color=LOSS, fillcolor="rgba(227,73,72,0.15)")
    return fig


def returns_scatter(
    port: pd.Series, bench: pd.Series, beta: float, alpha_daily: float
) -> go.Figure:
    """Daily portfolio vs benchmark returns with the fitted beta line."""
    df = pd.concat([port.rename("p"), bench.rename("b")], axis=1, join="inner").dropna()
    fig = _base("Daily returns vs benchmark (slope = beta)")
    fig.add_trace(
        go.Scatter(
            x=df["b"],
            y=df["p"],
            mode="markers",
            name="Daily returns",
            marker=dict(color=SERIES[0], size=8, opacity=0.55, line=dict(width=1, color="white")),
            hovertemplate="Benchmark %{x:.2%}<br>Portfolio %{y:.2%}<extra></extra>",
        )
    )
    if not df.empty:
        xs = pd.Series([df["b"].min(), df["b"].max()])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=alpha_daily + beta * xs,
                mode="lines",
                name=f"Fit: beta {beta:.2f}",
                line=dict(color=SERIES[1], width=2),
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(title="Benchmark daily return", tickformat=".1%", showgrid=True)
    fig.update_yaxes(title="Portfolio daily return", tickformat=".1%")
    fig.update_layout(hovermode="closest")
    return fig


def candlestick_chart(bars: pd.DataFrame, ticker: str, currency: str) -> go.Figure:
    """Daily OHLC candles for the trade ticket."""
    fig = _base(f"{ticker} daily prices ({currency})", height=340)
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name=ticker,
            increasing=dict(line=dict(color=UP, width=1)),
            decreasing=dict(line=dict(color=DOWN, width=1)),
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
    return fig
