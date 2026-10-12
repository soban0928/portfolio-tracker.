"""How the pre-tournament predictions fared against real results."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts import win_prob_bars
from ui import STAGE_LABELS, csv, page, pct, summary, year_picker

page("Predicted vs Actual", "📈")
year = year_picker()
t = summary()["tournaments"][str(year)]
full, elo = t["tournament"]["full"], t["tournament"]["elo_only"]

c = st.columns(4)
c[0].metric(
    "Champion",
    full["champion"],
    f"ranked #{full['champion_rank']} pre-tournament",
    delta_color="off",
    delta_arrow="off",
    border=True,
)
c[1].metric(
    "Champion's pre-tournament odds",
    pct(full["champion_p_win"]),
    f"Elo-only model: {pct(elo['champion_p_win'])}",
    delta_color="off",
    delta_arrow="off",
    border=True,
)
c[2].metric(
    "Favourite",
    full["favourite"],
    pct(full["favourite_p_win"]),
    delta_color="off",
    delta_arrow="off",
    border=True,
)
c[3].metric(
    "Exit-stage log score",
    f"{full['exit_log_score']:.3f}",
    f"uniform {full['exit_log_score_uniform']:.3f} · Elo-only {elo['exit_log_score']:.3f}",
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="Mean log-probability the simulation gave each team's actual exit stage. "
    "Higher (closer to 0) is better.",
)

stages = csv(f"{year}/stages_full.csv", index_col="team")
actual = csv(f"{year}/actual_progress.csv", index_col="team")["stage"]
left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(
        win_prob_bars(
            stages,
            compare=csv(f"{year}/stages_elo_only.csv", index_col="team")["p_WINNER"],
            actual=actual,
            n=20,
            stage_labels=STAGE_LABELS,
        )
    )
    st.caption(
        "Bars: full model's pre-tournament P(win), coloured by how far each team really went "
        "(darker blue = further, orange = champion); diamonds: Elo-only model."
    )
with right:
    st.markdown("**Where the real finalists and semi-finalists ranked**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Team": k, "Reached": STAGE_LABELS[actual[k]], "Rank by P(reach SF)": v}
                for k, v in full["semifinalist_ranks"].items()
            ]
        ).sort_values("Rank by P(reach SF)"),
        hide_index=True,
    )
    br = pd.DataFrame(full["stage_brier"]).T
    br.index = [STAGE_LABELS[i] for i in br.index]
    br.columns = ["Model", "Uniform baseline"]
    st.markdown("**Brier score for 'reached this stage'** (lower is better)")
    st.dataframe(br.style.format("{:.3f}"))

st.subheader("Match-level accuracy")
scores = csv(f"{year}/match_scores.csv", index_col=0).sort_values("group_log_loss")
cols = {
    "group_accuracy": "Group acc.",
    "group_log_loss": "Group log loss",
    "group_rps": "Group RPS",
    "ko_log_loss": "KO advance log loss",
    "ko_accuracy": "KO advance acc.",
}
st.dataframe(
    scores[list(cols)]
    .rename(columns=cols)
    .style.format(
        {"Group acc.": "{:.1%}", "KO advance acc.": "{:.1%}"}
        | {v: "{:.3f}" for k, v in cols.items() if "acc" not in k}
    )
)
st.caption(
    f"{int(scores['n_group'].iloc[0])} group matches scored on win/draw/loss; "
    f"{int(scores['n_ko'].iloc[0])} knockout ties scored on who advanced (the data's knockout "
    "scores include extra time). Small samples: differences of a few hundredths are noise."
)

st.subheader("Every match")
preds = csv(f"{year}/match_predictions.csv")
model = st.selectbox(
    "Model",
    list(dict.fromkeys(preds["model"])),
    index=list(dict.fromkeys(preds["model"])).index("Dixon-Coles (goals)"),
)
d = preds[preds["model"] == model].copy()
d["Score"] = (
    d["home_team"]
    + " "
    + d["home_score"].astype(str)
    + "–"
    + d["away_score"].astype(str)
    + " "
    + d["away_team"]
)
d["P(actual result)"] = d.apply(lambda r: r[f"p_{r['result']}"], axis=1)
show = d[["date", "stage", "Score", "p_H", "p_D", "p_A", "P(actual result)", "p_advance"]]
st.dataframe(
    show.style.format(
        {
            "p_H": "{:.0%}",
            "p_D": "{:.0%}",
            "p_A": "{:.0%}",
            "P(actual result)": "{:.0%}",
            "p_advance": "{:.0%}",
        },
        na_rep="–",
    ),
    hide_index=True,
    height=420,
)
upsets = d[d["stage"] == "GROUP"].nsmallest(5, "P(actual result)")
surprises = "; ".join(f"{r['Score']} ({r['P(actual result)']:.0%})" for _, r in upsets.iterrows())
st.markdown(
    f"**Biggest group-stage surprises** (lowest probability given to what happened): {surprises}"
)
