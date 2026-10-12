"""Pick two teams; get W/D/L probabilities and likely scorelines."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts import outcome_bar, score_heatmap
from ui import page, pct, predictor

from wcpredict.predict import COMPETITIONS

page("Match Predictor", "🎯")
pred = predictor()
teams = pred.teams

c1, c2, c3, c4 = st.columns([3, 3, 3, 3])
team_a = c1.selectbox("Team A", teams, index=teams.index("Spain") if "Spain" in teams else 0)
team_b = c2.selectbox(
    "Team B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1
)
venue_label = c3.selectbox("Venue", ["Neutral", f"{team_a} at home", f"{team_b} at home"])
comp_label = c4.selectbox("Competition", list(COMPETITIONS))
venue = {"Neutral": 0.0, f"{team_a} at home": 1.0, f"{team_b} at home": -1.0}[venue_label]

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

p = pred.predict(team_a, team_b, venue=venue, competition=COMPETITIONS[comp_label])
st.caption(
    f"Team form and ratings as of {pred.as_of.date()} (latest data). "
    f"{pred.meta['classifier']} for outcomes; Dixon-Coles goals model for scorelines."
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(
        outcome_bar(
            p.classifier_probs, team_a, team_b, f"{pred.meta['classifier']} (W/D/L classifier)"
        )
    )
    st.plotly_chart(outcome_bar(p.goals_probs, team_a, team_b, "Dixon-Coles goals model"))
    st.plotly_chart(score_heatmap(p.score_matrix, team_a, team_b))
with right:
    sa, sb = pred.state(team_a), pred.state(team_b)
    st.subheader("Most likely scorelines")
    st.dataframe(
        pd.DataFrame(
            [
                {"Score": f"{team_a} {h}–{a} {team_b}", "Probability": pct(q)}
                for h, a, q in p.top_scorelines
            ]
        ),
        hide_index=True,
    )
    st.markdown(
        f"**Expected goals:** {team_a} {p.expected_goals[0]:.2f} · "
        f"{team_b} {p.expected_goals[1]:.2f}"
    )
    st.subheader("Inputs")
    st.dataframe(
        pd.DataFrame(
            {
                team_a: [
                    f"{sa.elo:.0f}",
                    f"{sa.pts10:.2f}",
                    f"{sa.gf10:.2f}",
                    f"{sa.ga10:.2f}",
                    str(sa.last_match.date()) if sa.last_match is not None else "–",
                ],
                team_b: [
                    f"{sb.elo:.0f}",
                    f"{sb.pts10:.2f}",
                    f"{sb.gf10:.2f}",
                    f"{sb.ga10:.2f}",
                    str(sb.last_match.date()) if sb.last_match is not None else "–",
                ],
            },
            index=[
                "Elo rating",
                "Points/game (last 10)",
                "Goals for/game (last 10)",
                "Goals against/game (last 10)",
                "Last match",
            ],
        )
    )
    st.caption(
        "Draws are the hardest outcome to predict: no model here ever makes a draw "
        "its single most likely result unless the teams are very evenly matched."
    )
