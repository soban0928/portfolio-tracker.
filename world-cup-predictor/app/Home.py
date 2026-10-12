"""Streamlit entry point: project overview and headline results."""

from __future__ import annotations

import streamlit as st
from ui import page, pct, summary

page("World Cup Predictor")
s = summary()

st.markdown(
    "Pre-tournament predictions for the **2018, 2022 and 2026 FIFA World Cups**, each made "
    "using only matches played before kick-off, then checked against what really happened. "
    "Match probabilities come from Elo-based machine-learning models; tournament odds from "
    f"{s['n_sims']:,} Monte Carlo simulations per tournament."
)

cols = st.columns(3)
for col, year in zip(cols, ["2026", "2022", "2018"], strict=True):
    t = s["tournaments"][year]["tournament"]["full"]
    col.metric(
        f"{year} champion: {t['champion']}",
        f"ranked #{t['champion_rank']}",
        f"pre-tournament P(win) {pct(t['champion_p_win'])}",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    col.caption(f"Favourite was {t['favourite']} ({pct(t['favourite_p_win'])}).")

st.subheader("Pages")
st.markdown(
    """
- **Match Predictor**: win/draw/loss probabilities and likely scorelines for any two teams.
- **Tournament Probabilities**: each team's chance of reaching every stage, with Monte Carlo
  standard errors.
- **Simulated Bracket**: who most often fills each knockout slot, next to the real bracket.
- **Model Evaluation**: rolling out-of-sample metrics against baselines, calibration, SHAP.
- **Predicted vs Actual**: how the pre-tournament predictions fared.
"""
)
st.caption(
    f"Data: martj42/international_results (CC0), matches up to {s['current_as_of']}. "
    f"Best ML model by rolling log loss: {s['best_model']}. Seed {s['seed']}."
)
