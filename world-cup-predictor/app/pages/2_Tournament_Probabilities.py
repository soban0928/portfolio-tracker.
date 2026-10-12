"""Stage-by-stage probabilities from 100,000 simulations."""

from __future__ import annotations

import streamlit as st
from charts import stage_heatmap, win_prob_bars
from ui import STAGE_LABELS, csv, page, pct, summary, year_picker

page("Tournament Probabilities", "📊")
year = year_picker()
model = st.radio("Simulation model", ["Full Dixon-Coles", "Elo-only Dixon-Coles"], horizontal=True)
label = "full" if model.startswith("Full") else "elo_only"
stages = csv(f"{year}/stages_{label}.csv", index_col="team")
other = csv(f"{year}/stages_{'elo_only' if label == 'full' else 'full'}.csv", index_col="team")
meta = summary()["tournaments"][str(year)]["sim_meta"]
stage_cols = [c for c in stages.columns if c.startswith("p_")]

st.caption(
    f"Pre-tournament simulation: {meta['n_sims']:,} tournaments in {meta['n_batches']} batches "
    f"(seed {meta['seed']}), goals-model parameters re-sampled per batch. Trained only on "
    f"matches before {summary()['tournaments'][str(year)]['start']}."
)
top = st.slider("Teams shown", 8, len(stages), min(24, len(stages)), step=4)
left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(stage_heatmap(stages.head(top), stage_cols, STAGE_LABELS))
with right:
    st.plotly_chart(win_prob_bars(stages, compare=other["p_WINNER"], n=min(top, 20)))
    st.caption(
        "Bars show the selected model with 95% Monte Carlo intervals "
        "(±1.96 batch-means standard errors); diamonds show the other model."
    )

st.subheader("Full table")
table = stages[["elo", *stage_cols, "se_WINNER"]].copy()
table.columns = ["Elo"] + [STAGE_LABELS[c[2:]] for c in stage_cols] + ["MCSE (champion)"]
st.dataframe(
    table.style.format(
        {c: "{:.1%}" for c in table.columns if c != "Elo"}
        | {"Elo": "{:.0f}", "MCSE (champion)": "{:.2%}"}
    ),
    height=420,
)
fav = stages.index[0]
st.caption(
    f"Favourite: {fav} at {pct(stages.loc[fav, 'p_WINNER'])} "
    f"± {pct(1.96 * stages.loc[fav, 'se_WINNER'], 2)} (95% MC interval)."
)

st.subheader("Group finishing positions")
pos = csv(f"{year}/positions.csv", index_col="team")
groups = summary()["tournaments"][str(year)]["groups"]
cols = st.columns(4)
for g, members in enumerate(groups):
    with cols[g % 4]:
        st.markdown(f"**Group {g + 1}**")
        d = pos.loc[members].sort_values("p_1st", ascending=False)
        d.columns = ["1st", "2nd", "3rd", "4th"]
        st.dataframe(d.style.format("{:.0%}"))
st.caption("Groups are numbered by first kick-off, which may differ from FIFA's letters.")
