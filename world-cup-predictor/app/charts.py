"""Plotly figure builders (pure functions: data in, figure out).

Colours come from a CVD-validated categorical palette in fixed slot order;
sequential encodings use a single blue ramp. Text uses neutral ink.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
NEUTRAL = "#898781"
INK = "#52514e"
BLUES = [
    [0.0, "#f4f8fe"],
    [0.15, "#cde2fb"],
    [0.35, "#86b6ef"],
    [0.6, "#3987e5"],
    [0.8, "#1c5cab"],
    [1.0, "#0d366b"],
]
STAGE_COLORS = {
    "GROUP": "#d6d4cd",
    "R32": "#cde2fb",
    "R16": "#9ec5f4",
    "QF": "#5598e7",
    "SF": "#256abf",
    "FINAL": "#184f95",
    "WINNER": "#eb6834",
}
FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif")


def _layout(fig: go.Figure, title: str, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=15)),
        height=height,
        margin=dict(l=8, r=8, t=70 if legend else 44, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=legend,
        font=FONT,
    )
    return fig


def outcome_bar(probs: np.ndarray, team_a: str, team_b: str, title: str) -> go.Figure:
    """100% stacked bar of win / draw / win probabilities."""
    fig = go.Figure()
    for label, p, color in [
        (f"{team_a} win", probs[0], SERIES[0]),
        ("Draw", probs[1], NEUTRAL),
        (f"{team_b} win", probs[2], SERIES[1]),
    ]:
        fig.add_trace(
            go.Bar(
                x=[p],
                y=[""],
                orientation="h",
                name=label,
                marker_color=color,
                text=f"{p:.0%}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.2, legend_traceorder="normal")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    return _layout(fig, title, height=170)


def score_heatmap(matrix: np.ndarray, team_a: str, team_b: str, max_goals: int = 6) -> go.Figure:
    """Scoreline probability heatmap (rows: team A goals)."""
    m = matrix[: max_goals + 1, : max_goals + 1]
    fig = go.Figure(
        go.Heatmap(
            z=m,
            x=list(range(max_goals + 1)),
            y=list(range(max_goals + 1)),
            colorscale=BLUES,
            text=[[f"{v:.1%}" if v >= 0.005 else "" for v in row] for row in m],
            texttemplate="%{text}",
            showscale=False,
            hovertemplate=f"{team_a} %{{y}} – %{{x}} {team_b}: %{{z:.2%}}<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"{team_b} goals", side="top", dtick=1)
    fig.update_yaxes(title=f"{team_a} goals", autorange="reversed", dtick=1)
    return _layout(fig, "Scoreline probabilities (Dixon-Coles)", height=430, legend=False)


def stage_heatmap(stages: pd.DataFrame, stage_cols: list[str], labels: dict[str, str]) -> go.Figure:
    """Team x stage probability grid."""
    z = stages[stage_cols].to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[labels[c[2:]] for c in stage_cols],
            y=list(stages.index),
            colorscale=BLUES,
            zmin=0,
            zmax=1,
            text=[[f"{v:.0%}" if v >= 0.005 else "<1%" for v in r] for r in z],
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="%{y} · %{x}: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig = _layout(
        fig, "Probability of reaching each stage", height=28 * len(stages) + 110, legend=False
    )
    fig.update_layout(margin=dict(t=96))
    return fig


def win_prob_bars(
    stages: pd.DataFrame,
    compare: pd.Series | None = None,
    actual: pd.Series | None = None,
    n: int = 16,
    stage_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Horizontal bars of P(win) with 95% Monte Carlo intervals.

    With ``actual``, bars are coloured by the stage each team really reached
    (one legend entry per stage). ``compare`` adds a second model as markers.
    """
    top = stages.head(n).iloc[::-1]
    fig = go.Figure()
    if actual is None:
        groups = [("Full model", SERIES[0], top)]
    else:
        reached = actual.reindex(top.index)
        labels = stage_labels or {}
        groups = [
            (labels.get(st, st), STAGE_COLORS[st], top[reached == st])
            for st in reversed(list(STAGE_COLORS))
            if (reached == st).any()
        ]
    for name, color, d in groups:
        fig.add_trace(
            go.Bar(
                x=d["p_WINNER"],
                y=d.index,
                orientation="h",
                name=name,
                marker=dict(color=color, cornerradius=4),
                error_x=dict(type="data", array=1.96 * d["se_WINNER"], color=INK, thickness=1),
                hovertemplate="%{y}: %{x:.2%}<extra>" + name + "</extra>",
            )
        )
    if compare is not None:
        fig.add_trace(
            go.Scatter(
                x=compare.reindex(top.index),
                y=top.index,
                mode="markers",
                name="Elo-only model",
                marker=dict(symbol="diamond", size=9, color=INK, line=dict(width=1, color="white")),
                hovertemplate="%{y} (Elo-only): %{x:.2%}<extra></extra>",
            )
        )
    fig.update_yaxes(categoryorder="array", categoryarray=list(top.index))
    fig.update_xaxes(tickformat=".0%", title="P(win tournament)")
    fig.update_layout(barmode="overlay", legend_traceorder="normal")
    fig = _layout(fig, "Chance of winning the tournament", height=24 * len(top) + 130)
    if actual is not None:  # many legend entries: put them under the plot
        fig.update_layout(
            legend=dict(y=-0.14, yanchor="top", title_text="Actually reached: "),
            margin=dict(t=44, b=10),
            height=24 * len(top) + 190,
        )
    return fig


def metric_lines(df: pd.DataFrame, metric: str, models: list[str], title: str) -> go.Figure:
    """One line per model across CV years."""
    fig = go.Figure()
    for k, m in enumerate(models):
        d = df[df["model"] == m].sort_values("year")
        fig.add_trace(
            go.Scatter(
                x=d["year"],
                y=d[metric],
                mode="lines+markers",
                name=m,
                line=dict(color=SERIES[k % len(SERIES)], width=2),
                marker=dict(size=8),
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1)
    return _layout(fig, title)


def reliability(cal: pd.DataFrame, models: list[str], outcome: str, title: str) -> go.Figure:
    """Reliability diagram (predicted vs observed frequency) for one outcome class."""
    fig = go.Figure(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=NEUTRAL, dash="dot", width=1),
        )
    )
    for k, m in enumerate(models):
        d = cal[(cal["model"] == m) & (cal["outcome"] == outcome) & (cal["count"] >= 20)]
        fig.add_trace(
            go.Scatter(
                x=d["mean_predicted"],
                y=d["observed_rate"],
                mode="lines+markers",
                name=m,
                line=dict(color=SERIES[k], width=2),
                marker=dict(size=8),
                customdata=d["count"],
                hovertemplate=(
                    "predicted %{x:.1%} → observed %{y:.1%} (n=%{customdata})<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title="Mean predicted probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Observed frequency", tickformat=".0%", range=[0, 1])
    return _layout(fig, title, height=420)


def shap_bars(imp: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Mean |SHAP| per feature."""
    d = imp.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=d["mean_abs"],
            y=[labels.get(i, i) for i in d.index],
            orientation="h",
            marker=dict(color=SERIES[0], cornerradius=4),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Mean |SHAP value| (log-odds), averaged over outcomes")
    return _layout(fig, "Feature importance (SHAP)", height=24 * len(d) + 110, legend=False)


def shap_strip(sample: pd.DataFrame, features: list[str], labels: dict[str, str]) -> go.Figure:
    """Beeswarm-style strip: SHAP value for a first-team win vs feature value."""
    fig = go.Figure()
    for k, f in enumerate(features):
        d = sample[sample["feature"] == f]
        v = d["value"]
        norm = (v - v.quantile(0.05)) / max(v.quantile(0.95) - v.quantile(0.05), 1e-9)
        jitter = np.random.default_rng(k).uniform(-0.3, 0.3, len(d))
        fig.add_trace(
            go.Scatter(
                x=d["shap_H"],
                y=k + jitter,
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=5,
                    color=norm.clip(0, 1),
                    colorscale=BLUES,
                    opacity=0.7,
                    showscale=k == 0,
                    colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["low", "high"]),
                ),
                customdata=v,
                hovertemplate=f"{labels.get(f, f)} = %{{customdata:.2f}}"
                "<br>SHAP %{x:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(
        tickvals=list(range(len(features))),
        ticktext=[labels.get(f, f) for f in features],
        autorange="reversed",
    )
    fig.update_xaxes(title="SHAP value for 'first team wins' (log-odds)", zeroline=True)
    return _layout(
        fig,
        "How feature values push the win probability",
        height=60 * len(features) + 120,
        legend=False,
    )
